from __future__ import annotations

import os
import re
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from pydantic import BaseModel, Field, validator

# Optional: re-use your existing engine if available
# If you don't want execution here, you can stub it out safely.
from predictflow.engine.parser import parse_workflow  # must accept YAML/BPMN file path
from predictflow.engine.executor import Executor

router = APIRouter(prefix="/processes", tags=["processes"])

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "predictflow.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Auth dependency (simple API key)
# -----------------------------
def require_api_key(x_api_key: Optional[str] = None):
    expected = os.getenv("PREDICTFLOW_API_KEY")
    if not expected:
        # If no key configured, lock down everything but let tests override
        raise HTTPException(status_code=503, detail="API not configured (missing PREDICTFLOW_API_KEY).")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return True

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn

def _init_tables():
    conn = _conn()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS processes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                format TEXT NOT NULL,         -- 'yaml' | 'bpmn'
                content TEXT NOT NULL,        -- raw text
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS workflow_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_id INTEGER,
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                metrics_json TEXT,
                FOREIGN KEY (process_id) REFERENCES processes(id)
            );
            """
        )
    finally:
        conn.close()

_init_tables()

# -----------------------------
# Models & Validators
# -----------------------------
ALLOWED_FORMATS = {"yaml", "yml", "bpmn"}
NAME_RE = re.compile(r"^[A-Za-z0-9_\- ]{1,80}$")

class ProcessCreate(BaseModel):
    name: str = Field(..., max_length=80, description="Human-friendly process name")
    version: str = Field("1.0.0", max_length=20)
    format: str = Field(..., description="'yaml'/'yml' or 'bpmn'")
    content: str = Field(..., description="Raw YAML/BPMN text")

    @validator("format")
    def check_format(cls, v: str) -> str:
        v = v.lower().strip()
        if v == "yml":
            v = "yaml"
        if v not in ALLOWED_FORMATS:
            raise ValueError("format must be one of: yaml, yml, bpmn")
        return v

    @validator("name")
    def check_name(cls, v: str) -> str:
        v = v.strip()
        if not NAME_RE.match(v):
            raise ValueError("name must be 1-80 chars; alnum/_/-/space only")
        return v

    @validator("content")
    def clamp_content(cls, v: str) -> str:
        # Defensive cap (about 2MB)
        if len(v.encode("utf-8")) > 2_000_000:
            raise ValueError("content too large")
        return v


class ProcessOut(BaseModel):
    id: int
    name: str
    version: str
    format: str
    created_at: str


class ExecuteOut(BaseModel):
    run_id: int
    workflow_name: str
    critical_path: List[str]
    metrics: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------
def _row_to_process_out(row: sqlite3.Row) -> ProcessOut:
    return ProcessOut(
        id=row["id"],
        name=row["name"],
        version=row["version"],
        format=row["format"],
        created_at=row["created_at"],
    )


# -----------------------------
# Endpoints
# -----------------------------
@router.get("/", response_model=List[ProcessOut])
def list_processes(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _auth=Depends(require_api_key),
):
    conn = _conn()
    try:
        cur = conn.execute(
            "SELECT id, name, version, format, created_at FROM processes ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [_row_to_process_out(r) for r in cur.fetchall()]
    finally:
        conn.close()


@router.post("/", response_model=ProcessOut, status_code=201)
def create_process(payload: ProcessCreate, _auth=Depends(require_api_key)):
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO processes(name, version, format, content) VALUES (?, ?, ?, ?)",
            (payload.name.strip(), payload.version.strip(), payload.format, payload.content),
        )
        pid = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        row = conn.execute(
            "SELECT id, name, version, format, created_at FROM processes WHERE id = ?",
            (pid,),
        ).fetchone()
        return _row_to_process_out(row)
    finally:
        conn.close()


@router.get("/{process_id}", response_model=ProcessOut)
def get_process(process_id: int, _auth=Depends(require_api_key)):
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT id, name, version, format, created_at FROM processes WHERE id = ?",
            (process_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Process not found")
        return _row_to_process_out(row)
    finally:
        conn.close()


@router.post("/{process_id}/execute", response_model=ExecuteOut)
def execute_process(process_id: int, _auth=Depends(require_api_key)):
    """
    Execute a stored process:
    - Writes content to a safe temp file
    - Calls the internal parser + executor
    - Saves a run record with metrics
    """
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM processes WHERE id = ?", (process_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Process not found")

        fmt = row["format"].lower()
        suffix = ".yaml" if fmt in ("yaml", "yml") else ".bpmn"

        # Safe temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(row["content"].encode("utf-8"))
            tmp.flush()
            tmp_path = tmp.name

        try:
            workflow = parse_workflow(tmp_path)
            executor = Executor(workflow)
            executor.run()

            # Build metrics summary (safe)
            metrics = {}
            if isinstance(executor.metrics, dict):
                for k, v in executor.metrics.items():
                    if isinstance(v, dict):
                        rpn = v.get("rpn")
                        conf = v.get("confidence")
                        if isinstance(rpn, (int, float)) and isinstance(conf, (int, float)):
                            metrics[k] = {"rpn": rpn, "confidence": conf}

            # Critical path
            critical = []
            if metrics:
                sorted_steps = sorted(
                    metrics.items(),
                    key=lambda s: (s[1]["rpn"], -s[1]["confidence"]),
                    reverse=True,
                )
                critical = [s[0] for s in sorted_steps[:3]]

            # Save run
            cur = conn.execute(
                "INSERT INTO workflow_runs(process_id, status, metrics_json) VALUES (?, ?, ?)",
                (process_id, "completed", json.dumps(metrics)),
            )
            run_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

            return ExecuteOut(
                run_id=run_id,
                workflow_name=workflow.get("name", "Unnamed Workflow"),
                critical_path=critical,
                metrics=metrics,
            )

        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    finally:
        conn.close()

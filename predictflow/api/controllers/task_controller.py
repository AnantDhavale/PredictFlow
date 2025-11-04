from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, validator

router = APIRouter(prefix="/tasks", tags=["tasks"])

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "predictflow.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Auth
# -----------------------------
def require_api_key(x_api_key: Optional[str] = None):
    expected = os.getenv("PREDICTFLOW_API_KEY")
    if not expected:
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
            CREATE TABLE IF NOT EXISTS user_tasks (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                lane TEXT,
                status TEXT NOT NULL DEFAULT 'pending', -- pending | done | cancelled
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT
            );
            """
        )
    finally:
        conn.close()

_init_tables()

# -----------------------------
# Models
# -----------------------------
ALLOWED_STATUS = {"pending", "done", "cancelled"}

class TaskCreate(BaseModel):
    id: str = Field(..., min_length=1, max_length=64, description="Unique task identifier")
    description: str = Field(..., min_length=1, max_length=500)
    lane: Optional[str] = Field(default=None, max_length=64)

class TaskUpdate(BaseModel):
    description: Optional[str] = Field(default=None, max_length=500)
    lane: Optional[str] = Field(default=None, max_length=64)
    status: Optional[str] = Field(default=None)

    @validator("status")
    def check_status(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.lower().strip()
        if v not in ALLOWED_STATUS:
            raise ValueError("status must be one of: pending, done, cancelled")
        return v

class TaskOut(BaseModel):
    id: str
    description: str
    lane: Optional[str]
    status: str
    created_at: str
    updated_at: Optional[str]


def _row_to_task(row: sqlite3.Row) -> TaskOut:
    return TaskOut(
        id=row["id"],
        description=row["description"],
        lane=row["lane"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )

# -----------------------------
# Endpoints
# -----------------------------
@router.get("/", response_model=List[TaskOut])
def list_tasks(
    status_filter: Optional[str] = Query(None, description="pending|done|cancelled"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _auth=Depends(require_api_key),
):
    conn = _conn()
    try:
        if status_filter:
            status_filter = status_filter.lower().strip()
            if status_filter not in ALLOWED_STATUS:
                raise HTTPException(400, "Invalid status filter")
            cur = conn.execute(
                "SELECT * FROM user_tasks WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status_filter, limit, offset),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM user_tasks ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        return [_row_to_task(r) for r in cur.fetchall()]
    finally:
        conn.close()


@router.post("/", response_model=TaskOut, status_code=status.HTTP_201_CREATED)
def create_task(payload: TaskCreate, _auth=Depends(require_api_key)):
    conn = _conn()
    try:
        # Fail fast if duplicate ID
        existing = conn.execute("SELECT id FROM user_tasks WHERE id = ?", (payload.id,)).fetchone()
        if existing:
            raise HTTPException(409, "Task id already exists")

        conn.execute(
            "INSERT INTO user_tasks(id, description, lane, status) VALUES (?, ?, ?, 'pending')",
            (payload.id.strip(), payload.description.strip(), (payload.lane or "").strip() or None),
        )
        row = conn.execute("SELECT * FROM user_tasks WHERE id = ?", (payload.id,)).fetchone()
        return _row_to_task(row)
    finally:
        conn.close()


@router.get("/{task_id}", response_model=TaskOut)
def get_task(task_id: str, _auth=Depends(require_api_key)):
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM user_tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Task not found")
        return _row_to_task(row)
    finally:
        conn.close()


@router.patch("/{task_id}", response_model=TaskOut)
def update_task(task_id: str, payload: TaskUpdate, _auth=Depends(require_api_key)):
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM user_tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Task not found")

        # Build dynamic update safely
        fields, params = [], []
        if payload.description is not None:
            fields.append("description = ?")
            params.append(payload.description.strip())
        if payload.lane is not None:
            fields.append("lane = ?")
            params.append(payload.lane.strip() or None)
        if payload.status is not None:
            fields.append("status = ?")
            params.append(payload.status)

        if not fields:
            return _row_to_task(row)

        fields.append("updated_at = datetime('now')")
        q = f"UPDATE user_tasks SET {', '.join(fields)} WHERE id = ?"
        params.append(task_id)
        conn.execute(q, tuple(params))

        row = conn.execute("SELECT * FROM user_tasks WHERE id = ?", (task_id,)).fetchone()
        return _row_to_task(row)
    finally:
        conn.close()


@router.delete("/{task_id}", status_code=204)
def delete_task(task_id: str, _auth=Depends(require_api_key)):
    conn = _conn()
    try:
        cur = conn.execute("DELETE FROM user_tasks WHERE id = ?", (task_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "Task not found")
        return
    finally:
        conn.close()

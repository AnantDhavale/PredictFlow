# predictflow/engine/persistence.py
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "predictflow.db"


# ---------- low-level helpers ----------
def _ensure_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


# ---------- schema + init ----------
def init_db():
    conn = _ensure_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS workflow_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT DEFAULT (datetime('now')),
        completed_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS workflow_state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        join_seen TEXT,
        context TEXT,
        metrics TEXT,
        updated_at TEXT DEFAULT (datetime('now'))
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_tasks (
        id TEXT PRIMARY KEY,
        lane TEXT,
        description TEXT,
        status TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        completed_at TEXT,
        result TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT NOT NULL,
        payload TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    )
    """)
    conn.commit()
    conn.close()


# ---------- workflow run logging ----------
def log_run_start(name: str) -> int:
    conn = _ensure_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO workflow_runs (name, status) VALUES (?, ?)", (name, "running"))
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def log_run_status(run_id: int, status: str):
    conn = _ensure_db()
    conn.execute("UPDATE workflow_runs SET status=? WHERE id=?", (status, run_id))
    conn.commit()
    conn.close()


def log_run_complete(run_id: int):
    conn = _ensure_db()
    conn.execute("""
        UPDATE workflow_runs
        SET status='completed', completed_at=datetime('now')
        WHERE id=?
    """, (run_id,))
    conn.commit()
    conn.close()


def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    """List recent workflow runs."""
    conn = _ensure_db()
    cur = conn.execute("""
        SELECT id, name, status, started_at, completed_at
        FROM workflow_runs
        ORDER BY started_at DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ---------- workflow state ----------
def save_state(name: str, state: Dict[str, Any]):
    """Persist engine state."""
    conn = _ensure_db()
    payload = {
        "join_seen": state.get("join_seen", {}),
        "context": state.get("context", {}),
        "metrics": state.get("metrics", {}),
    }
    conn.execute("""
        INSERT INTO workflow_state (name, join_seen, context, metrics, updated_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        ON CONFLICT(name) DO UPDATE SET
            join_seen=excluded.join_seen,
            context=excluded.context,
            metrics=excluded.metrics,
            updated_at=excluded.updated_at
    """, (
        name,
        json.dumps(payload["join_seen"]),
        json.dumps(payload["context"]),
        json.dumps(payload["metrics"]),
    ))
    conn.commit()
    conn.close()


def load_state(name: str) -> Dict[str, Any]:
    conn = _ensure_db()
    cur = conn.execute("SELECT join_seen, context, metrics FROM workflow_state WHERE name=?", (name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"join_seen": {}, "context": {}, "metrics": {}}
    return {
        "join_seen": json.loads(row["join_seen"] or "{}"),
        "context": json.loads(row["context"] or "{}"),
        "metrics": json.loads(row["metrics"] or "{}"),
    }


# ---------- user tasks ----------
def save_user_task(task: Dict[str, Any]):
    """Create or update a pending user task."""
    conn = _ensure_db()
    conn.execute("""
        INSERT OR REPLACE INTO user_tasks (id, lane, description, status, created_at)
        VALUES (?, ?, ?, COALESCE(?, 'pending'), COALESCE(?, datetime('now')))
    """, (
        task.get("id"),
        task.get("lane"),
        task.get("description"),
        task.get("status", "pending"),
        datetime.utcnow().isoformat(timespec="seconds")
    ))
    conn.commit()
    conn.close()


def list_user_tasks(status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _ensure_db()
    if status:
        cur = conn.execute("SELECT * FROM user_tasks WHERE status=? ORDER BY created_at ASC", (status,))
    else:
        cur = conn.execute("SELECT * FROM user_tasks ORDER BY created_at ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def complete_user_task(task_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
    """Mark a user task as complete and record its result."""
    conn = _ensure_db()
    cur = conn.execute("""
        UPDATE user_tasks
        SET status='completed',
            completed_at=datetime('now'),
            result=?
        WHERE id=? AND status='pending'
    """, (json.dumps(result or {}), task_id))
    conn.commit()
    changed = cur.rowcount > 0
    conn.close()
    return changed


# ---------- messages (correlation) ----------
def correlate_message(key: str, payload: Optional[Dict[str, Any]] = None):
    conn = _ensure_db()
    conn.execute(
        "INSERT INTO messages (key, payload) VALUES (?, ?)",
        (key, json.dumps(payload or {}))
    )
    conn.commit()
    conn.close()


def list_messages(key: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _ensure_db()
    if key:
        cur = conn.execute("SELECT * FROM messages WHERE key=? ORDER BY created_at DESC", (key,))
    else:
        cur = conn.execute("SELECT * FROM messages ORDER BY created_at DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

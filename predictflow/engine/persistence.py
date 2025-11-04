# predictflow/engine/persistence.py
"""
Persistence layer for PredictFlow workflow engine.
Handles all database operations with security hardening.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "predictflow.db"


# ---------- Low-level helpers ----------
def _ensure_db():
    """
    Get a database connection with proper configuration.
    Uses WAL mode for better concurrency.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")  # Enable foreign key constraints
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise


def _close_db(conn):
    """Safely close database connection."""
    try:
        if conn:
            conn.close()
    except sqlite3.Error as e:
        logger.error(f"Error closing database: {e}")


# ---------- Schema + Init ----------
def init_db():
    """Initialize database schema with all required tables."""
    conn = None
    try:
        conn = _ensure_db()
        cur = conn.cursor()

        # Workflow runs table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS workflow_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
            started_at TEXT DEFAULT (datetime('now')),
            completed_at TEXT,
            error_message TEXT,
            created_by TEXT DEFAULT 'system',
            CONSTRAINT valid_dates CHECK(completed_at IS NULL OR completed_at >= started_at)
        )
        """)

        # Workflow state table
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

        # User tasks table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_tasks (
            id TEXT PRIMARY KEY,
            lane TEXT,
            description TEXT,
            status TEXT NOT NULL CHECK(status IN ('pending', 'completed', 'cancelled')),
            assigned_to TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            completed_at TEXT,
            result TEXT,
            CONSTRAINT valid_task_dates CHECK(completed_at IS NULL OR completed_at >= created_at)
        )
        """)

        # Messages table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            payload TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            processed INTEGER DEFAULT 0,
            CONSTRAINT valid_key_length CHECK(length(key) <= 255)
        )
        """)

        # Create indexes for better performance
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_workflow_runs_status 
        ON workflow_runs(status, started_at DESC)
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_tasks_status 
        ON user_tasks(status, created_at ASC)
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_key 
        ON messages(key, created_at DESC)
        """)

        conn.commit()
        logger.info(f"Database initialized successfully at {DB_PATH}")
        
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        _close_db(conn)


# ---------- Workflow Run Logging ----------
def log_run_start(name: str, created_by: str = "system") -> int:
    """
    Log the start of a workflow run.
    
    Args:
        name: Workflow name
        created_by: User who initiated the workflow
        
    Returns:
        int: Run ID
    """
    conn = None
    try:
        # Input validation
        if not name or not isinstance(name, str):
            raise ValueError("Workflow name must be a non-empty string")
        
        if len(name) > 255:
            raise ValueError("Workflow name too long (max 255 characters)")
        
        conn = _ensure_db()
        cur = conn.cursor()
        
        # ✅ SAFE: Using parameterized query
        cur.execute(
            "INSERT INTO workflow_runs (name, status, created_by) VALUES (?, ?, ?)",
            (name, "running", created_by)
        )
        
        run_id = cur.lastrowid
        conn.commit()
        logger.info(f"Started workflow run {run_id}: {name}")
        return run_id
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error logging run start: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        _close_db(conn)


def log_run_status(run_id: int, status: str, error_message: Optional[str] = None):
    """
    Update workflow run status.
    
    Args:
        run_id: Workflow run ID
        status: New status (running, completed, failed, cancelled)
        error_message: Optional error message if failed
    """
    conn = None
    try:
        # Input validation
        if not isinstance(run_id, int) or run_id <= 0:
            raise ValueError("Invalid run_id")
        
        valid_statuses = ['running', 'completed', 'failed', 'cancelled']
        if status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        
        conn = _ensure_db()
        
        # ✅ SAFE: Using parameterized query
        conn.execute(
            "UPDATE workflow_runs SET status=?, error_message=? WHERE id=?",
            (status, error_message, run_id)
        )
        
        conn.commit()
        logger.info(f"Updated run {run_id} status to {status}")
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error updating run status: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        _close_db(conn)


def log_run_complete(run_id: int):
    """Mark a workflow run as completed."""
    conn = None
    try:
        if not isinstance(run_id, int) or run_id <= 0:
            raise ValueError("Invalid run_id")
        
        conn = _ensure_db()
        
        # ✅ SAFE: Using parameterized query
        conn.execute("""
            UPDATE workflow_runs
            SET status='completed', completed_at=datetime('now')
            WHERE id=?
        """, (run_id,))
        
        conn.commit()
        logger.info(f"Completed workflow run {run_id}")
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error completing run: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        _close_db(conn)


def list_runs(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None,
    created_by: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List workflow runs with filtering and pagination.
    
    Args:
        limit: Maximum number of results (max 100)
        offset: Number of results to skip
        status: Filter by status
        created_by: Filter by creator
        
    Returns:
        List of workflow run dictionaries
    """
    conn = None
    try:
        # Input validation
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Limit must be a positive integer")
        
        if limit > 100:
            limit = 100  # Cap at 100 for safety
        
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("Offset must be a non-negative integer")
        
        conn = _ensure_db()
        
        # Build query with optional filters
        query = """
            SELECT id, name, status, started_at, completed_at, error_message, created_by
            FROM workflow_runs
            WHERE 1=1
        """
        params = []
        
        # ✅ SAFE: Using parameterized queries for filters
        if status:
            valid_statuses = ['running', 'completed', 'failed', 'cancelled']
            if status not in valid_statuses:
                raise ValueError(f"Invalid status filter: {status}")
            query += " AND status=?"
            params.append(status)
        
        if created_by:
            query += " AND created_by=?"
            params.append(created_by)
        
        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cur = conn.execute(query, tuple(params))
        rows = [dict(r) for r in cur.fetchall()]
        
        logger.info(f"Listed {len(rows)} workflow runs")
        return rows
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error listing runs: {e}")
        raise
    finally:
        _close_db(conn)


# ---------- Workflow State ----------
def save_state(name: str, state: Dict[str, Any]):
    """
    Persist workflow engine state.
    
    Args:
        name: Workflow name
        state: State dictionary containing join_seen, context, metrics
    """
    conn = None
    try:
        # Input validation
        if not name or not isinstance(name, str):
            raise ValueError("Workflow name must be a non-empty string")
        
        if len(name) > 255:
            raise ValueError("Workflow name too long (max 255 characters)")
        
        if not isinstance(state, dict):
            raise ValueError("State must be a dictionary")
        
        conn = _ensure_db()
        
        # Extract and validate state components
        payload = {
            "join_seen": state.get("join_seen", {}),
            "context": state.get("context", {}),
            "metrics": state.get("metrics", {}),
        }
        
        # Serialize to JSON
        try:
            join_seen_json = json.dumps(payload["join_seen"])
            context_json = json.dumps(payload["context"])
            metrics_json = json.dumps(payload["metrics"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"State data is not JSON serializable: {e}")
        
        # ✅ SAFE: Using parameterized query
        conn.execute("""
            INSERT INTO workflow_state (name, join_seen, context, metrics, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(name) DO UPDATE SET
                join_seen=excluded.join_seen,
                context=excluded.context,
                metrics=excluded.metrics,
                updated_at=excluded.updated_at
        """, (name, join_seen_json, context_json, metrics_json))
        
        conn.commit()
        logger.info(f"Saved state for workflow: {name}")
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error saving state: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        _close_db(conn)


def load_state(name: str) -> Dict[str, Any]:
    """
    Load workflow state from database.
    
    Args:
        name: Workflow name
        
    Returns:
        State dictionary or empty state if not found
    """
    conn = None
    try:
        # Input validation
        if not name or not isinstance(name, str):
            raise ValueError("Workflow name must be a non-empty string")
        
        conn = _ensure_db()
        
        # ✅ SAFE: Using parameterized query
        cur = conn.execute(
            "SELECT join_seen, context, metrics FROM workflow_state WHERE name=?",
            (name,)
        )
        row = cur.fetchone()
        
        if not row:
            logger.info(f"No saved state found for workflow: {name}")
            return {"join_seen": {}, "context": {}, "metrics": {}}
        
        # Parse JSON safely
        try:
            return {
                "join_seen": json.loads(row["join_seen"] or "{}"),
                "context": json.loads(row["context"] or "{}"),
                "metrics": json.loads(row["metrics"] or "{}"),
            }
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted state data for {name}: {e}")
            return {"join_seen": {}, "context": {}, "metrics": {}}
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error loading state: {e}")
        raise
    finally:
        _close_db(conn)


# ---------- User Tasks ----------
def save_user_task(task: Dict[str, Any]):
    """
    Create or update a user task.
    
    Args:
        task: Dictionary with id, lane, description, status, assigned_to
    """
    conn = None
    try:
        # Input validation
        if not isinstance(task, dict):
            raise ValueError("Task must be a dictionary")
        
        task_id = task.get("id")
        if not task_id or not isinstance(task_id, str):
            raise ValueError("Task must have a valid string ID")
        
        if len(task_id) > 255:
            raise ValueError("Task ID too long (max 255 characters)")
        
        conn = _ensure_db()
        
        # ✅ SAFE: Using parameterized query
        conn.execute("""
            INSERT OR REPLACE INTO user_tasks 
            (id, lane, description, status, assigned_to, created_at)
            VALUES (?, ?, ?, COALESCE(?, 'pending'), ?, datetime('now'))
        """, (
            task_id,
            task.get("lane"),
            task.get("description"),
            task.get("status", "pending"),
            task.get("assigned_to", "system")
        ))
        
        conn.commit()
        logger.info(f"Saved user task: {task_id}")
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error saving user task: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        _close_db(conn)


def get_user_task(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single user task by ID.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task dictionary or None if not found
    """
    conn = None
    try:
        if not task_id or not isinstance(task_id, str):
            raise ValueError("Task ID must be a non-empty string")
        
        conn = _ensure_db()
        
        # ✅ SAFE: Using parameterized query
        cur = conn.execute(
            "SELECT * FROM user_tasks WHERE id=?",
            (task_id,)
        )
        row = cur.fetchone()
        
        return dict(row) if row else None
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error getting user task: {e}")
        raise
    finally:
        _close_db(conn)


def list_user_tasks(
    status: Optional[str] = None,
    assigned_to: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    List user tasks with optional filtering.
    
    Args:
        status: Filter by status (pending, completed, cancelled)
        assigned_to: Filter by assigned user
        limit: Maximum results (default 100, max 500)
        
    Returns:
        List of task dictionaries
    """
    conn = None
    try:
        # Input validation
        if limit > 500:
            limit = 500
        
        conn = _ensure_db()
        
        # Build query with filters
        query = "SELECT * FROM user_tasks WHERE 1=1"
        params = []
        
        # ✅ SAFE: Using parameterized queries
        if status:
            valid_statuses = ['pending', 'completed', 'cancelled']
            if status not in valid_statuses:
                raise ValueError(f"Invalid status: {status}")
            query += " AND status=?"
            params.append(status)
        
        if assigned_to:
            query += " AND assigned_to=?"
            params.append(assigned_to)
        
        query += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)
        
        cur = conn.execute(query, tuple(params))
        rows = [dict(r) for r in cur.fetchall()]
        
        logger.info(f"Listed {len(rows)} user tasks")
        return rows
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error listing user tasks: {e}")
        raise
    finally:
        _close_db(conn)


def complete_user_task(
    task_id: str,
    result: Optional[Dict[str, Any]] = None,
    completed_by: Optional[str] = None
) -> bool:
    """
    Mark a user task as completed.
    
    Args:
        task_id: Task ID
        result: Optional result data
        completed_by: User who completed the task
        
    Returns:
        bool: True if task was completed, False if not found or already completed
    """
    conn = None
    try:
        # Input validation
        if not task_id or not isinstance(task_id, str):
            raise ValueError("Task ID must be a non-empty string")
        
        if result is not None and not isinstance(result, dict):
            raise ValueError("Result must be a dictionary")
        
        conn = _ensure_db()
        
        # Serialize result
        result_json = json.dumps(result or {})
        
        # ✅ SAFE: Using parameterized query
        cur = conn.execute("""
            UPDATE user_tasks
            SET status='completed',
                completed_at=datetime('now'),
                result=?
            WHERE id=? AND status='pending'
        """, (result_json, task_id))
        
        conn.commit()
        changed = cur.rowcount > 0
        
        if changed:
            logger.info(f"Completed task {task_id} by {completed_by or 'system'}")
        else:
            logger.warning(f"Task {task_id} not found or already completed")
        
        return changed
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error completing user task: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        _close_db(conn)


def cancel_user_task(task_id: str) -> bool:
    """
    Cancel a pending user task.
    
    Args:
        task_id: Task ID
        
    Returns:
        bool: True if cancelled, False if not found or already completed
    """
    conn = None
    try:
        if not task_id or not isinstance(task_id, str):
            raise ValueError("Task ID must be a non-empty string")
        
        conn = _ensure_db()
        
        # ✅ SAFE: Using parameterized query
        cur = conn.execute("""
            UPDATE user_tasks
            SET status='cancelled'
            WHERE id=? AND status='pending'
        """, (task_id,))
        
        conn.commit()
        changed = cur.rowcount > 0
        
        if changed:
            logger.info(f"Cancelled task {task_id}")
        
        return changed
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error cancelling task: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        _close_db(conn)


# ---------- Messages (Correlation) ----------
def correlate_message(key: str, payload: Optional[Dict[str, Any]] = None):
    """
    Store a correlated message.
    
    Args:
        key: Message correlation key
        payload: Optional message payload
    """
    conn = None
    try:
        # Input validation
        if not key or not isinstance(key, str):
            raise ValueError("Message key must be a non-empty string")
        
        if len(key) > 255:
            raise ValueError("Message key too long (max 255 characters)")
        
        if payload is not None and not isinstance(payload, dict):
            raise ValueError("Payload must be a dictionary")
        
        conn = _ensure_db()
        
        payload_json = json.dumps(payload or {})
        
        # ✅ SAFE: Using parameterized query
        conn.execute(
            "INSERT INTO messages (key, payload) VALUES (?, ?)",
            (key, payload_json)
        )
        
        conn.commit()
        logger.info(f"Correlated message with key: {key}")
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error correlating message: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        _close_db(conn)


def list_messages(
    key: Optional[str] = None,
    limit: int = 100,
    processed: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    List messages with optional filtering.
    
    Args:
        key: Filter by message key
        limit: Maximum results
        processed: Filter by processed status
        
    Returns:
        List of message dictionaries
    """
    conn = None
    try:
        if limit > 500:
            limit = 500
        
        conn = _ensure_db()
        
        # Build query
        query = "SELECT * FROM messages WHERE 1=1"
        params = []
        
        # ✅ SAFE: Using parameterized queries
        if key:
            query += " AND key=?"
            params.append(key)
        
        if processed is not None:
            query += " AND processed=?"
            params.append(1 if processed else 0)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cur = conn.execute(query, tuple(params))
        rows = [dict(r) for r in cur.fetchall()]
        
        logger.info(f"Listed {len(rows)} messages")
        return rows
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error listing messages: {e}")
        raise
    finally:
        _close_db(conn)


def mark_message_processed(message_id: int):
    """Mark a message as processed."""
    conn = None
    try:
        if not isinstance(message_id, int) or message_id <= 0:
            raise ValueError("Invalid message ID")
        
        conn = _ensure_db()
        
        # ✅ SAFE: Using parameterized query
        conn.execute(
            "UPDATE messages SET processed=1 WHERE id=?",
            (message_id,)
        )
        
        conn.commit()
        logger.info(f"Marked message {message_id} as processed")
        
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"Error marking message processed: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        _close_db(conn)


# ---------- Utility Functions ----------
def get_db_stats() -> Dict[str, Any]:
    """Get database statistics."""
    conn = None
    try:
        conn = _ensure_db()
        
        stats = {}
        
        # Count records in each table
        for table in ['workflow_runs', 'workflow_state', 'user_tasks', 'messages']:
            cur = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[f"{table}_count"] = cur.fetchone()["count"]
        
        # Database file size
        stats["db_size_bytes"] = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        
        return stats
        
    except (sqlite3.Error, OSError) as e:
        logger.error(f"Error getting DB stats: {e}")
        raise
    finally:
        _close_db(conn)


def vacuum_db():
    """Vacuum the database to reclaim space."""
    conn = None
    try:
        conn = _ensure_db()
        conn.execute("VACUUM")
        logger.info("Database vacuumed successfully")
        
    except sqlite3.Error as e:
        logger.error(f"Error vacuuming database: {e}")
        raise
    finally:
        _close_db(conn)

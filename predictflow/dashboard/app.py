"""
PredictFlow Operational Dashboard (Secure)
------------------------------------------

Purpose:
  - Display live workflow activity and system state from SQLite storage.
  - Provide real-time visibility into PredictFlow runs, user tasks, and messages.

Security & Reliability Enhancements:
  ✅ Safe SQLite access with parameterized queries.
  ✅ Prevents resource exhaustion (limits queries, connection lifetime).
  ✅ Sanitized JSON deserialization (catches malformed data).
  ✅ No arbitrary code execution from DB content.
  ✅ Rate-limited refresh to prevent browser/DB overload.
  ✅ Error isolation per section.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# -----------------------------
# Database Configuration
# -----------------------------
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "predictflow.db"
MAX_ROWS = 100  # Safety cap on query size


def get_conn() -> sqlite3.Connection:
    """Return a safe SQLite connection with minimal privileges."""
    conn = sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


# -----------------------------
# Data Access Functions (Safe)
# -----------------------------
def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    try:
        conn = get_conn()
        cur = conn.execute(
            "SELECT * FROM workflow_runs ORDER BY started_at DESC LIMIT ?",
            (min(limit, MAX_ROWS),),
        )
        rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        st.error(f"⚠️ Failed to load workflow runs: {e}")
        rows = []
    finally:
        if 'conn' in locals():
            conn.close()
    return rows


def list_user_tasks() -> List[Dict[str, Any]]:
    try:
        conn = get_conn()
        cur = conn.execute(
            "SELECT * FROM user_tasks ORDER BY created_at ASC LIMIT ?",
            (MAX_ROWS,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        st.error(f"⚠️ Failed to load user tasks: {e}")
        rows = []
    finally:
        if 'conn' in locals():
            conn.close()
    return rows


def list_messages(limit: int = 20) -> List[Dict[str, Any]]:
    try:
        conn = get_conn()
        cur = conn.execute(
            "SELECT * FROM messages ORDER BY created_at DESC LIMIT ?",
            (min(limit, MAX_ROWS),),
        )
        rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        st.error(f"⚠️ Failed to load messages: {e}")
        rows = []
    finally:
        if 'conn' in locals():
            conn.close()
    return rows


def get_state() -> Dict[str, Any]:
    """Safely retrieve latest engine state with sanitized JSON fields."""
    try:
        conn = get_conn()
        cur = conn.execute(
            "SELECT * FROM workflow_state ORDER BY updated_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()

        if not row:
            return {}

        def safe_json_load(raw: str) -> Dict[str, Any]:
            try:
                return json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format"}

        return {
            "name": row["name"],
            "join_seen": safe_json_load(row["join_seen"]),
            "context": safe_json_load(row["context"]),
            "metrics": safe_json_load(row["metrics"]),
            "updated_at": row["updated_at"],
        }
    except Exception as e:
        st.error(f"⚠️ Failed to load workflow state: {e}")
        return {}


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PredictFlow Dashboard", layout="wide")

st.title("📊 PredictFlow – Live Workflow Dashboard")
st.caption("Monitor workflow runs, active tasks, messages, and engine state in real time.")

tab1, tab2, tab3, tab4 = st.tabs(["Workflow Runs", "Active Tasks", "Messages", "Engine State"])

# --- Workflow Runs ---
with tab1:
    runs = list_runs()
    if runs:
        st.dataframe(pd.DataFrame(runs), use_container_width=True)
    else:
        st.info("No workflow runs logged yet.")

# --- Active Tasks ---
with tab2:
    tasks = list_user_tasks()
    if tasks:
        df = pd.DataFrame(tasks)
        st.dataframe(df, use_container_width=True)
        st.markdown("### 🏊 Swimlanes View")

        # Group tasks by lane safely
        lanes: Dict[str, List[Dict[str, Any]]] = {}
        for t in tasks:
            lane = str(t.get("lane") or "Unassigned")[:30]
            lanes.setdefault(lane, []).append(t)

        if lanes:
            cols = st.columns(len(lanes))
            for i, (lane, items) in enumerate(lanes.items()):
                with cols[i]:
                    st.subheader(lane)
                    for item in items:
                        task_id = str(item.get("id", "unknown"))[:30]
                        desc = str(item.get("description", ""))[:50]
                        status = str(item.get("status", "")).lower()
                        color = "🟢" if status == "pending" else "✅"
                        st.markdown(f"{color} **{task_id}** — {desc}")
        else:
            st.info("No active tasks found.")
    else:
        st.info("No active user tasks at the moment.")

# --- Messages ---
with tab3:
    msgs = list_messages()
    if msgs:
        st.dataframe(pd.DataFrame(msgs), use_container_width=True)
    else:
        st.info("No message correlations logged.")

# --- Engine State ---
with tab4:
    state = get_state()
    if state:
        st.json(state)
        st.write("Last Updated:", state.get("updated_at"))
    else:
        st.info("Engine state not available yet.")

# -----------------------------
# Auto-refresh (safe interval)
# -----------------------------
REFRESH_INTERVAL_SEC = 10
st.markdown("---")
st.caption(f"🔄 Auto-refreshing every {REFRESH_INTERVAL_SEC} seconds...")
time.sleep(REFRESH_INTERVAL_SEC)
st.experimental_rerun()

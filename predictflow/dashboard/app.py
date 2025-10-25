import streamlit as st
import pandas as pd
import time
from pathlib import Path
import sqlite3
import json

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "predictflow.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# -----------------------------
# Data access
# -----------------------------
def list_runs():
    conn = get_conn()
    cur = conn.execute("SELECT * FROM workflow_runs ORDER BY started_at DESC LIMIT 20")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def list_user_tasks():
    conn = get_conn()
    cur = conn.execute("SELECT * FROM user_tasks ORDER BY created_at ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def list_messages():
    conn = get_conn()
    cur = conn.execute("SELECT * FROM messages ORDER BY created_at DESC LIMIT 20")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def get_state():
    conn = get_conn()
    cur = conn.execute("SELECT * FROM workflow_state ORDER BY updated_at DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if not row:
        return {}
    return {
        "name": row["name"],
        "join_seen": json.loads(row["join_seen"] or "{}"),
        "context": json.loads(row["context"] or "{}"),
        "metrics": json.loads(row["metrics"] or "{}"),
        "updated_at": row["updated_at"]
    }


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="PredictFlow Dashboard", layout="wide")

st.title("📊 PredictFlow – Live Workflow Dashboard")
st.caption("Monitor workflow runs, user tasks, and message correlations in real time.")

tab1, tab2, tab3, tab4 = st.tabs(["Workflow Runs", "Active Tasks", "Messages", "Engine State"])

# --- Workflow Runs ---
with tab1:
    runs = list_runs()
    if runs:
        df = pd.DataFrame(runs)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No workflow runs logged yet.")

# --- Active Tasks ---
with tab2:
    tasks = list_user_tasks()
    if tasks:
        df = pd.DataFrame(tasks)
        st.dataframe(df, use_container_width=True)
        st.markdown("### 🏊 Swimlanes View")

        # Group tasks by lane (like pools/lanes)
        lanes = {}
        for t in tasks:
            lane = t.get("lane", "Unassigned")
            lanes.setdefault(lane, []).append(t)

        cols = st.columns(len(lanes) or 1)
        for i, (lane, items) in enumerate(lanes.items()):
            with cols[i]:
                st.subheader(lane)
                for item in items:
                    color = "🟢" if item["status"] == "pending" else "✅"
                    st.markdown(f"{color} **{item['id']}** — {item['description']}")
    else:
        st.info("No active user tasks at the moment.")

# --- Messages ---
with tab3:
    msgs = list_messages()
    if msgs:
        df = pd.DataFrame(msgs)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No message correlations logged.")

# --- Engine State ---
with tab4:
    state = get_state()
    if state:
        st.json(state)
        st.write("Last Updated:", state["updated_at"])
    else:
        st.info("Engine state not available yet.")

st.markdown("---")
st.caption("Auto-refreshing every 10 seconds...")
time.sleep(10)
st.experimental_rerun()

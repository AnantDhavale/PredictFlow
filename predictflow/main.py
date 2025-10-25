"""
PredictFlow Entry Point
-----------------------
Loads and runs a workflow file (YAML or BPMN).
Provides REST API for user task completion and message correlation.
"""

import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictflow.engine.parser import parse_workflow
from predictflow.engine.executor import Executor
from predictflow.engine.bpmn_parser import parse_bpmn
from predictflow.engine import persistence

# Initialize DB
persistence.init_db()

app = FastAPI(title="PredictFlow API", version="1.0")


# --------------------------
# CLI Runner
# --------------------------
def run_workflow(file_path: str, auto_generate: bool = True):
    """Load and run a workflow YAML or BPMN file."""
    print(f"\n[PredictFlow] Loading workflow from: {file_path}")
    if not auto_generate:
        print("[PredictFlow] Auto-generation disabled (--no-autogen)")

    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".yaml", ".yml"):
        workflow = parse_workflow(file_path)
    elif ext == ".bpmn":
        workflow = parse_bpmn(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .yaml or .bpmn")

    executor = Executor(workflow, yaml_path=file_path, auto_generate=auto_generate)
    executor.run()


# --------------------------
# API MODELS
# --------------------------
class CompleteTaskRequest(BaseModel):
    result: dict = {}
    user: str = "system"


class CorrelateMessageRequest(BaseModel):
    message_key: str


# --------------------------
# API ENDPOINTS
# --------------------------
@app.get("/tasks", summary="List pending user tasks")
def list_tasks():
    try:
        tasks = persistence.list_user_tasks(status="pending")
        return {"tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tasks/{task_id}/complete", summary="Complete a user task")
def complete_task(task_id: str, body: CompleteTaskRequest):
    try:
        ok = persistence.complete_user_task(task_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found or already completed.")
        return {"status": "completed", "task_id": task_id, "by": body.user, "result": body.result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/messages/correlate", summary="Correlate a message and resume waiting tokens")
def correlate_message(body: CorrelateMessageRequest):
    try:
        persistence.correlate_message(body.message_key)
        return {"status": "correlated", "message": body.message_key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs", summary="List workflow runs")
def list_runs():
    try:
        conn = persistence._ensure_db()
        rows = conn.execute("SELECT * FROM workflow_runs ORDER BY started_at DESC").fetchall()
        conn.close()
        return {"workflow_runs": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------
# ENTRY POINT
# --------------------------
if __name__ == "__main__":
    # --- special mode: init DB only ---
    if "--init-db" in sys.argv:
        persistence.init_db()
        print("[PredictFlow] Database initialized at", persistence.DB_PATH)
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  ▶ Run workflow: python -m predictflow.main <file.bpmn|.yaml> [--no-autogen]")
        print("  ▶ Start API server: uvicorn predictflow.main:app --reload")
        print("  ▶ Initialize DB: python -m predictflow.main --init-db")
        sys.exit(1)

    file_path = sys.argv[1]
    auto_generate = "--no-autogen" not in sys.argv
    run_workflow(file_path, auto_generate)

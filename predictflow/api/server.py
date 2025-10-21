from fastapi import FastAPI, UploadFile, File
from predictflow.engine.parser import parse_workflow
from predictflow.engine.executor import Executor
import tempfile
import json

app = FastAPI(
    title="PredictFlow API",
    description="Run workflows and get predictive insights via REST API",
    version="0.1.0"
)


@app.post("/run_workflow/")
async def run_workflow(file: UploadFile = File(...)):
    """
    Upload a YAML workflow file and run it through PredictFlow.
    Returns summary metrics and critical path.
    """
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    # Parse and execute
    workflow = parse_workflow(tmp_path)
    executor = Executor(workflow)
    executor.run()

    # Prepare output
    response = {
        "workflow": workflow.get("name", "Unnamed Workflow"),
        "metrics": executor.metrics,
    }

    # Compute critical path
    sorted_steps = sorted(
        executor.metrics.items(),
        key=lambda s: (s[1].get("rpn", 0), -s[1].get("confidence", 1)),
        reverse=True
    )
    critical = [s[0] for s in sorted_steps[:3]]
    response["critical_path"] = critical

    return response


@app.get("/")
def root():
    return {"message": "Welcome to PredictFlow API — upload a YAML at /run_workflow/."}

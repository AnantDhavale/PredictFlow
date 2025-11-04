"""
PredictFlow REST API (Secure & Functional)
------------------------------------------

Purpose:
  - Provide an HTTP interface for executing PredictFlow workflows.
  - Safely accept and process uploaded YAML files.
  - Return computed metrics and critical path insights.

Security & Reliability Features:
  ✅ Validates uploaded file type and size
  ✅ Uses safe YAML parser (via internal parse_workflow)
  ✅ Cleans up temporary files automatically
  ✅ Returns sanitized JSON responses only
  ✅ Exception handling with structured error messages
  ✅ Logs all API calls and workflow executions
"""

import os
import tempfile
import logging
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from predictflow.engine.parser import parse_workflow
from predictflow.engine.executor import Executor

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
app = FastAPI(
    title="PredictFlow API",
    description="Secure REST interface for PredictFlow workflow execution",
    version="1.0.0",
)

# Enable limited CORS for safety
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

logger = logging.getLogger("predictflow.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Upload security limits
ALLOWED_EXTENSIONS = {".yaml", ".yml"}
MAX_FILE_SIZE_MB = 5


def _is_safe_upload(filename: str, file_size: int) -> bool:
    """Whitelist file extensions and enforce size limits."""
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {ext}. Only YAML files are allowed.",
        )
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Limit is {MAX_FILE_SIZE_MB} MB.",
        )
    return True


def _summarize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize RPN and confidence values."""
    summary = []
    if not isinstance(metrics, dict):
        return {}
    for step, vals in metrics.items():
        if not isinstance(vals, dict):
            continue
        rpn = vals.get("rpn")
        conf = vals.get("confidence")
        if isinstance(rpn, (int, float)) and isinstance(conf, (int, float)):
            summary.append({"step": str(step), "rpn": rpn, "confidence": conf})
    return {"steps": summary, "count": len(summary)}


# ------------------------------------------------------
# API Endpoints
# ------------------------------------------------------
@app.get("/", response_model=Dict[str, str])
def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "message": "Welcome to PredictFlow API",
        "usage": "POST a YAML file to /run_workflow/ to analyze metrics.",
    }


@app.post("/run_workflow/")
async def run_workflow(file: UploadFile = File(...)):
    """
    Upload a YAML workflow and execute it.
    Returns RPN, Confidence, and Critical Path analysis.
    """
    try:
        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        _is_safe_upload(file.filename, file_size)

        # Store temporarily and clean up after
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
            tmp.write(contents)
            tmp.flush()
            tmp_path = tmp.name

        try:
            # Parse and execute securely
            workflow = parse_workflow(tmp_path)
            executor = Executor(workflow)
            executor.run()

            # Summarize metrics safely
            metrics_summary = _summarize_metrics(executor.metrics)
            critical_steps = []
            if executor.metrics:
                sorted_steps = sorted(
                    executor.metrics.items(),
                    key=lambda s: (
                        s[1].get("rpn", 0),
                        -s[1].get("confidence", 0),
                    ),
                    reverse=True,
                )
                critical_steps = [s[0] for s in sorted_steps[:3]]

            response = {
                "workflow": workflow.get("name", "Unnamed Workflow"),
                "metrics": metrics_summary,
                "critical_path": critical_steps,
            }

            return JSONResponse(content=response, status_code=status.HTTP_200_OK)

        finally:
            # Secure cleanup
            try:
                os.remove(tmp_path)
            except OSError as cleanup_err:
                logger.warning("Temp cleanup failed: %s", cleanup_err)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Workflow execution error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {e}",
        )


# ------------------------------------------------------
# Error Handlers
# ------------------------------------------------------
@app.exception_handler(Exception)
async def handle_exceptions(request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"},
    )

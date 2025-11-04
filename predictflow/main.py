"""
PredictFlow Entry Point
-----------------------
Loads and runs a workflow file (YAML or BPMN).
Provides REST API for user task completion and message correlation.

Security Features (v1.1.0):
- Optional API authentication via environment variable
- Input validation on all endpoints
- Proper error handling
- Rate limiting ready
"""

import sys
import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from predictflow.engine.parser import parse_workflow
from predictflow.engine.executor import Executor
from predictflow.engine.bpmn_parser import parse_bpmn
from predictflow.engine import persistence

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize DB
try:
    persistence.init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise

# FastAPI app
app = FastAPI(
    title="PredictFlow API",
    version="1.1.0",
    description="Workflow execution engine with predictive intelligence",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
# Adjust origins for your deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8501",  # Streamlit
        # Add your production domains here
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# --------------------------
# SECURITY CONFIGURATION
# --------------------------

# Optional authentication - set PREDICTFLOW_AUTH_ENABLED=true to enable
AUTH_ENABLED = os.getenv("PREDICTFLOW_AUTH_ENABLED", "false").lower() == "true"

# Simple token auth (for production, use JWT or OAuth2)
# Set PREDICTFLOW_API_TOKENS as comma-separated tokens
VALID_TOKENS = set(
    os.getenv("PREDICTFLOW_API_TOKENS", "").split(",")
) if os.getenv("PREDICTFLOW_API_TOKENS") else set()

security = HTTPBearer(auto_error=False)


def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[str]:
    """
    Verify API token if authentication is enabled.
    
    To enable authentication:
    1. Set environment variable: PREDICTFLOW_AUTH_ENABLED=true
    2. Set valid tokens: PREDICTFLOW_API_TOKENS=token1,token2,token3
    
    Returns:
        str: User identifier (token prefix) or None if auth disabled
    """
    if not AUTH_ENABLED:
        return None  # Auth disabled, allow all requests
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Set PREDICTFLOW_AUTH_ENABLED=false to disable auth.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    if not VALID_TOKENS or token not in VALID_TOKENS:
        logger.warning(f"Invalid authentication attempt with token: {token[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Return token prefix as user identifier
    return f"user_{token[:8]}"


# --------------------------
# CLI RUNNER
# --------------------------

def run_workflow(file_path: str, auto_generate: bool = True):
    """
    Load and run a workflow YAML or BPMN file.
    
    Args:
        file_path: Path to workflow file
        auto_generate: Enable auto-generation features
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported
    """
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Workflow file not found: {file_path}")
        
        # Validate file size (max 10MB)
        file_size = os.path.getsize(file_path)
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            raise ValueError(f"File too large: {file_size} bytes (max {max_size})")
        
        logger.info(f"Loading workflow from: {file_path}")
        if not auto_generate:
            logger.info("Auto-generation disabled (--no-autogen)")
        
        # Parse based on extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in (".yaml", ".yml"):
            workflow = parse_workflow(file_path)
        elif ext == ".bpmn":
            workflow = parse_bpmn(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {ext}. Supported types: .yaml, .yml, .bpmn"
            )
        
        # Execute workflow
        executor = Executor(workflow, yaml_path=file_path, auto_generate=auto_generate)
        executor.run()
        
        logger.info("Workflow execution completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        print(f"\n❌ Workflow execution failed: {e}")
        sys.exit(1)


# --------------------------
# API MODELS
# --------------------------

class CompleteTaskRequest(BaseModel):
    """Request body for completing a user task."""
    result: dict = Field(default_factory=dict, description="Task completion result data")
    user: str = Field(default="system", max_length=255, description="User completing the task")
    
    @validator('user')
    def validate_user(cls, v):
        if not v or not v.strip():
            raise ValueError("User cannot be empty")
        return v.strip()


class CorrelateMessageRequest(BaseModel):
    """Request body for message correlation."""
    message_key: str = Field(..., max_length=255, description="Message correlation key")
    payload: Optional[dict] = Field(default_factory=dict, description="Optional message payload")
    
    @validator('message_key')
    def validate_message_key(cls, v):
        if not v or not v.strip():
            raise ValueError("Message key cannot be empty")
        return v.strip()


class TaskFilter(BaseModel):
    """Query parameters for filtering tasks."""
    status: Optional[str] = Field(None, description="Filter by status: pending, completed, cancelled")
    limit: int = Field(50, ge=1, le=100, description="Maximum number of results")
    
    @validator('status')
    def validate_status(cls, v):
        if v and v not in ['pending', 'completed', 'cancelled']:
            raise ValueError("Status must be: pending, completed, or cancelled")
        return v


class RunFilter(BaseModel):
    """Query parameters for filtering workflow runs."""
    status: Optional[str] = Field(None, description="Filter by status")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")


# --------------------------
# API ENDPOINTS
# --------------------------

@app.get("/", summary="API root")
def root():
    """Welcome endpoint with API information."""
    auth_status = "enabled" if AUTH_ENABLED else "disabled"
    return {
        "service": "PredictFlow API",
        "version": "1.1.0",
        "status": "running",
        "authentication": auth_status,
        "docs": "/docs",
        "endpoints": {
            "tasks": "/tasks",
            "runs": "/runs",
            "messages": "/messages/correlate",
            "health": "/health"
        }
    }


@app.get("/health", summary="Health check")
def health_check():
    """Check if service is healthy."""
    try:
        stats = persistence.get_db_stats()
        return {
            "status": "healthy",
            "database": "connected",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.get("/tasks", summary="List user tasks")
def list_tasks(
    status: Optional[str] = None,
    limit: int = 50,
    user: Optional[str] = Depends(verify_token)
):
    """
    List user tasks with optional filtering.
    
    Query Parameters:
    - status: Filter by task status (pending, completed, cancelled)
    - limit: Maximum number of results (1-100, default 50)
    
    Authentication: Optional (configure via PREDICTFLOW_AUTH_ENABLED)
    """
    try:
        # Validate parameters
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be between 1 and 100"
            )
        
        if status and status not in ['pending', 'completed', 'cancelled']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid status. Must be: pending, completed, or cancelled"
            )
        
        # List tasks
        tasks = persistence.list_user_tasks(
            status=status,
            limit=limit
        )
        
        logger.info(f"Listed {len(tasks)} tasks (status={status}, user={user})")
        
        return {
            "tasks": tasks,
            "count": len(tasks),
            "filters": {
                "status": status,
                "limit": limit
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing tasks: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list tasks"
        )


@app.get("/tasks/{task_id}", summary="Get a specific task")
def get_task(
    task_id: str,
    user: Optional[str] = Depends(verify_token)
):
    """
    Get details of a specific task.
    
    Path Parameters:
    - task_id: Task identifier
    """
    try:
        # Validate task_id
        if not task_id or len(task_id) > 255:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID"
            )
        
        task = persistence.get_user_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return {"task": task}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get task"
        )


@app.post("/tasks/{task_id}/complete", summary="Complete a user task")
def complete_task(
    task_id: str,
    body: CompleteTaskRequest,
    user: Optional[str] = Depends(verify_token)
):
    """
    Mark a user task as completed.
    
    Path Parameters:
    - task_id: Task identifier
    
    Request Body:
    - result: Task completion result data (optional)
    - user: User completing the task (default: system)
    """
    try:
        # Validate task_id
        if not task_id or len(task_id) > 255:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID"
            )
        
        # Complete the task
        completed_by = user or body.user
        ok = persistence.complete_user_task(
            task_id=task_id,
            result=body.result,
            completed_by=completed_by
        )
        
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found or already completed"
            )
        
        logger.info(f"Task {task_id} completed by {completed_by}")
        
        return {
            "status": "completed",
            "task_id": task_id,
            "completed_by": completed_by,
            "result": body.result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing task {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete task"
        )


@app.post("/messages/correlate", summary="Correlate a message")
def correlate_message(
    body: CorrelateMessageRequest,
    user: Optional[str] = Depends(verify_token)
):
    """
    Correlate a message and resume waiting workflow tokens.
    
    Request Body:
    - message_key: Message correlation key (required)
    - payload: Optional message payload data
    """
    try:
        # Correlate message
        persistence.correlate_message(
            key=body.message_key,
            payload=body.payload
        )
        
        logger.info(f"Message correlated: {body.message_key} by {user or 'system'}")
        
        return {
            "status": "correlated",
            "message_key": body.message_key,
            "payload": body.payload
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error correlating message: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to correlate message"
        )


@app.get("/runs", summary="List workflow runs")
def list_runs(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    user: Optional[str] = Depends(verify_token)
):
    """
    List workflow runs with pagination and filtering.
    
    Query Parameters:
    - status: Filter by status (running, completed, failed, cancelled)
    - limit: Maximum results (1-100, default 20)
    - offset: Number of results to skip (default 0)
    """
    try:
        # Validate parameters
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be between 1 and 100"
            )
        
        if offset < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Offset must be non-negative"
            )
        
        # List runs using the proper persistence method
        runs = persistence.list_runs(
            limit=limit,
            offset=offset,
            status=status
        )
        
        logger.info(f"Listed {len(runs)} workflow runs (user={user})")
        
        return {
            "workflow_runs": runs,
            "count": len(runs),
            "limit": limit,
            "offset": offset,
            "has_more": len(runs) == limit
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing runs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list workflow runs"
        )


# --------------------------
# ADMIN ENDPOINTS (Optional)
# --------------------------

@app.get("/admin/stats", summary="Get database statistics")
def get_stats(user: Optional[str] = Depends(verify_token)):
    """
    Get database statistics.
    Note: Consider adding admin-only check in production.
    """
    try:
        stats = persistence.get_db_stats()
        return {"statistics": stats}
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get statistics"
        )


# --------------------------
# ERROR HANDLERS
# --------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(exc)
    )


# --------------------------
# ENTRY POINT
# --------------------------

if __name__ == "__main__":
    # Special mode: init DB only
    if "--init-db" in sys.argv:
        try:
            persistence.init_db()
            print(f"✅ [PredictFlow] Database initialized at {persistence.DB_PATH}")
            stats = persistence.get_db_stats()
            print(f"📊 Database stats: {stats}")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
            sys.exit(1)
    
    # Show help if no arguments
    if len(sys.argv) < 2:
        print("\n🚀 PredictFlow - Workflow Engine with Predictive Intelligence\n")
        print("Usage:")
        print("  ▶️  Run workflow:")
        print("      python -m predictflow.main <file.bpmn|.yaml> [--no-autogen]")
        print("\n  🌐 Start API server:")
        print("      uvicorn predictflow.main:app --reload")
        print("\n  🔧 Initialize database:")
        print("      python -m predictflow.main --init-db")
        print("\n  🔒 Enable authentication:")
        print("      export PREDICTFLOW_AUTH_ENABLED=true")
        print("      export PREDICTFLOW_API_TOKENS=token1,token2")
        print("      uvicorn predictflow.main:app --reload")
        print("\n  📚 API documentation:")
        print("      http://localhost:8000/docs")
        print()
        sys.exit(1)
    
    # Run workflow from CLI
    file_path = sys.argv[1]
    auto_generate = "--no-autogen" not in sys.argv
    run_workflow(file_path, auto_generate)

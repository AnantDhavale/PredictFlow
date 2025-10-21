"""
PredictFlow Entry Point
-----------------------
Loads and runs a workflow YAML file.
Automatically generates missing action stubs before execution.
"""

from predictflow.engine.parser import parse_workflow
from predictflow.engine.executor import Executor


def run_workflow(file_path: str):
    """Load and run a workflow YAML file."""
    print(f"\n[PredictFlow] Loading workflow from: {file_path}")

    # Parse workflow YAML
    workflow = parse_workflow(file_path)

    # Initialize executor with YAML path (for auto-generation)
    executor = Executor(workflow, yaml_path=file_path, auto_generate=True)

    # Run the workflow
    executor.run()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m predictflow.main <workflow_file.yaml>")
    else:
        run_workflow(sys.argv[1])

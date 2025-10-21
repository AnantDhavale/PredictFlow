"""
PredictFlow Entry Point
-----------------------
Loads and runs a workflow YAML file.
Automatically generates missing action stubs before execution,
unless --no-autogen is specified.
"""

import sys
from predictflow.engine.parser import parse_workflow
from predictflow.engine.executor import Executor


def run_workflow(file_path: str, auto_generate: bool = True):
    """Load and run a workflow YAML file."""
    print(f"\n[PredictFlow] Loading workflow from: {file_path}")
    if not auto_generate:
        print("[PredictFlow] Auto-generation disabled (--no-autogen)")

    # Parse workflow YAML
    workflow = parse_workflow(file_path)

    # Initialize executor with or without autogen
    executor = Executor(workflow, yaml_path=file_path, auto_generate=auto_generate)

    # Run workflow
    executor.run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m predictflow.main <workflow_file.yaml> [--no-autogen]")
        sys.exit(1)

    yaml_path = sys.argv[1]
    auto_generate = "--no-autogen" not in sys.argv

    run_workflow(yaml_path, auto_generate)

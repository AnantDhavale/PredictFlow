from predictflow.engine.parser import parse_workflow
from predictflow.engine.executor import Executor

def run_workflow(file_path: str):
    """Load and run a workflow YAML file."""
    workflow = parse_workflow(file_path)
    executor = Executor(workflow)
    executor.run()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m predictflow.main <workflow_file.yaml>")
    else:
        run_workflow(sys.argv[1])

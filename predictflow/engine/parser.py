import yaml

def parse_workflow(file_path: str):
    """Reads a YAML workflow definition and returns it as a Python dict."""
    with open(file_path, "r") as f:
        workflow = yaml.safe_load(f)
    return workflow

import os
import yaml
from pathlib import Path

ACTIONS_PATH = Path(__file__).resolve().parent.parent / "actions"


def generate_actions_from_yaml(yaml_file: str):
    """
    Parses a YAML workflow file and generates corresponding Python action stubs
    under the predictflow/actions/ directory.

    Example:
      - YAML Step:
          id: collect_data
          description: Gather quarterly revenue
          action: collect_data

      → creates predictflow/actions/collect_data.py
    """
    with open(yaml_file, "r") as f:
        workflow = yaml.safe_load(f)

    steps = workflow.get("steps", [])
    if not steps:
        print("No steps found in YAML.")
        return

    os.makedirs(ACTIONS_PATH, exist_ok=True)

    for step in steps:
        action_name = step.get("action") or step.get("id")
        description = step.get("description", "No description provided.")
        filename = f"{action_name}.py"
        filepath = ACTIONS_PATH / filename

        if filepath.exists():
            print(f"Action '{action_name}' already exists. Skipping.")
            continue

        content = f'''def run(context, step):
    """
    Auto-generated from YAML.
    Description: {description}
    """
    print("Executing auto-generated action: {action_name}")
    # TODO: Implement logic for '{action_name}'
    return {{"status": "success"}}
'''

        with open(filepath, "w") as f:
            f.write(content)

        print(f"Created action file: {filepath}")

    print("\Action generation complete.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m predictflow.engine.autogen <workflow.yaml>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    generate_actions_from_yaml(yaml_path)

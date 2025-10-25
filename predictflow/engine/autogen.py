import os
import re
import yaml
from pathlib import Path

ACTIONS_PATH = Path(__file__).resolve().parent.parent / "actions"


def generate_actions_from_yaml(file_path: str):
    """
    Parses a YAML or BPMN workflow file and generates corresponding Python action stubs
    under the predictflow/actions/ directory.

    - For YAML:
        Reads workflow['steps'] and creates Python stubs for each action.
    - For BPMN:
        Extracts <task> and <event> IDs from the XML and creates corresponding stubs.

    Example (YAML Step):
      id: collect_data
      description: Gather quarterly revenue
      action: collect_data

      → creates predictflow/actions/collect_data.py
    """
    if not os.path.exists(file_path):
        print(f"[PredictFlow] Autogen skipped — file not found: {file_path}")
        return

    with open(file_path, "r") as f:
        content = f.read()

    os.makedirs(ACTIONS_PATH, exist_ok=True)
    action_names = []

    # --- Detect format ---
    is_bpmn = "<definitions" in content and "</definitions>" in content

    if is_bpmn:
        print("[PredictFlow] Detected BPMN — extracting task IDs for action generation...")
        # Extract all task/event IDs from BPMN XML
        ids = re.findall(r'id="([A-Za-z0-9_]+)"', content)
        for item in ids:
            # Ignore flow lines and diagram definitions
            if item.lower().startswith("flow"):
                continue
            action_names.append(item)
    else:
        # --- YAML workflow ---
        try:
            workflow = yaml.safe_load(content)
            steps = workflow.get("steps", []) if isinstance(workflow, dict) else []
            if not steps:
                print("No steps found in YAML.")
                return

            for step in steps:
                action_name = step.get("action") or step.get("id")
                if action_name:
                    action_names.append(action_name)
        except Exception as e:
            print(f"Warning: Could not parse YAML for autogen: {e}")
            return

    # --- Create stub files ---
    for action_name in sorted(set(action_names)):
        if not action_name:
            continue

        filename = f"{action_name}.py"
        filepath = ACTIONS_PATH / filename

        if filepath.exists():
            print(f"Action '{action_name}' already exists. Skipping.")
            continue

        description = f"Auto-generated action stub for '{action_name}'."

        content = f'''def run(context, step):
    """
    {description}
    """
    print("Executing auto-generated action: {action_name}")
    # TODO: Implement logic for '{action_name}'
    return {{"status": "success"}}
'''

        with open(filepath, "w") as f:
            f.write(content)

        print(f"Created action file: {filepath}")

    print("Action generation complete.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m predictflow.engine.autogen <workflow.yaml or .bpmn>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    generate_actions_from_yaml(yaml_path)

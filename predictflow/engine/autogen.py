import os
import re
import yaml
from pathlib import Path

ACTIONS_PATH = Path(__file__).resolve().parent.parent / "actions"


def generate_actions_from_yaml(file_path: str):
    """
    ✅ SECURITY HARDENED: Generate action stubs with validation
    """
    if not os.path.exists(file_path):
        print(f"[PredictFlow] Autogen skipped – file not found: {file_path}")
        return

    with open(file_path, "r") as f:
        content = f.read()

    os.makedirs(ACTIONS_PATH, exist_ok=True)
    action_names = []

    is_bpmn = "<definitions" in content and "</definitions>" in content

    if is_bpmn:
        print("[PredictFlow] Detected BPMN – extracting task IDs for action generation...")
        ids = re.findall(r'id="([A-Za-z0-9_]+)"', content)
        for item in ids:
            if item.lower().startswith("flow"):
                continue
            action_names.append(item)
    else:
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

    # ✅ SECURITY FIX: Validate action names
    for action_name in sorted(set(action_names)):
        if not action_name:
            continue

        # Validate: only alphanumeric, underscore, hyphen
        if not re.match(r'^[a-zA-Z0-9_-]+$', action_name):
            print(f"❌ Skipping invalid action name: {action_name}")
            continue
        
        # Prevent path traversal
        if '/' in action_name or '\\' in action_name or '..' in action_name:
            print(f"❌ Skipping action with path characters: {action_name}")
            continue
        
        # Length limit
        if len(action_name) > 100:
            print(f"❌ Action name too long: {action_name}")
            continue

        filename = f"{action_name}.py"
        filepath = ACTIONS_PATH / filename
        
        # ✅ Verify filepath is actually inside ACTIONS_PATH
        try:
            filepath = filepath.resolve()
            if not str(filepath).startswith(str(ACTIONS_PATH.resolve())):
                print(f"❌ Path traversal attempt blocked: {action_name}")
                continue
        except Exception as e:
            print(f"❌ Invalid path for action {action_name}: {e}")
            continue

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

        try:
            with open(filepath, "w") as f:
                f.write(content)
            print(f"✅ Created action file: {filepath}")
        except Exception as e:
            print(f"❌ Failed to create action {action_name}: {e}")

    print("Action generation complete.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m predictflow.engine.autogen <workflow.yaml or .bpmn>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    generate_actions_from_yaml(yaml_path)

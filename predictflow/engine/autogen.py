"""
This module securely auto-generates Python action stubs from YAML or BPMN workflow definitions.
It scans a workflow file for defined actions, sanitizes their names, and creates corresponding
boilerplate .py files inside the 'actions/' directory. Existing actions are not overwritten,
ensuring safe, one-time generation of reusable action modules.
"""

import os
import re
import yaml
from pathlib import Path

ACTIONS_PATH = Path(__file__).resolve().parent.parent / "actions"


def _sanitize_action_name(name: str) -> str:
    """
    Clean and validate an action name to prevent path traversal or unsafe file writes.
    Returns None if invalid.
    """
    if not name or not isinstance(name, str):
        return None

    # Remove any leading/trailing whitespace
    name = name.strip()

    # Disallow dangerous patterns (slashes, traversal, etc.)
    if "/" in name or "\\" in name or ".." in name:
        return None

    # Restrict to alphanumeric, underscore, and hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        return None

    # Enforce reasonable length
    if len(name) > 100:
        return None

    return name


def generate_actions_from_yaml(file_path: str):
    """
    ✅ SECURITY HARDENED: Generate action stubs safely from YAML or BPMN definitions.
    Prevents arbitrary file writes by validating and sanitizing action names.
    """
    if not os.path.exists(file_path):
        print(f"[PredictFlow] Autogen skipped – file not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    os.makedirs(ACTIONS_PATH, exist_ok=True)
    action_names = []

    # Detect BPMN vs YAML
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

    # Process and validate each unique action
    for raw_name in sorted(set(action_names)):
        safe_name = _sanitize_action_name(raw_name)
        if not safe_name:
            print(f"❌ Skipping invalid or unsafe action name: {raw_name}")
            continue

        filename = f"{safe_name}.py"
        filepath = ACTIONS_PATH / filename

        # Resolve canonical path and confirm it's under ACTIONS_PATH
        try:
            filepath = filepath.resolve(strict=False)
            if not str(filepath).startswith(str(ACTIONS_PATH.resolve())):
                print(f"❌ Path traversal attempt blocked: {raw_name}")
                continue
        except Exception as e:
            print(f"❌ Invalid path for action {raw_name}: {e}")
            continue

        if filepath.exists():
            print(f"Action '{safe_name}' already exists. Skipping.")
            continue

        description = f"Auto-generated action stub for '{safe_name}'."

        file_content = f'''def run(context, step):
    """
    {description}
    """
    print("Executing auto-generated action: {safe_name}")
    # TODO: Implement logic for '{safe_name}'
    return {{"status": "success"}}
'''

        try:
            # Safe file creation
            with open(filepath, "x", encoding="utf-8") as f:
                f.write(file_content)
            print(f"✅ Created action file: {filepath}")
        except FileExistsError:
            print(f"Action '{safe_name}' already exists (race condition). Skipping.")
        except Exception as e:
            print(f"❌ Failed to create action '{safe_name}': {e}")

    print("Action generation complete.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m predictflow.engine.autogen <workflow.yaml or .bpmn>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    generate_actions_from_yaml(yaml_path)

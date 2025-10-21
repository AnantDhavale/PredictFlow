import json
from datetime import datetime

class StateManager:
    """Handles workflow state persistence."""
    def __init__(self, file_path="workflow_state.json"):
        self.file_path = file_path

    def save_state(self, state):
        state["last_saved"] = datetime.utcnow().isoformat()
        with open(self.file_path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

import json
import os
import tempfile
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class StateManager:
    """Handles workflow state persistence with validation and safe I/O."""

    def __init__(self, file_path="workflow_state.json"):
        self.file_path = file_path

    def save_state(self, state):
        """Safely persist state to disk with atomic write."""
        if not isinstance(state, dict):
            raise ValueError("State must be a dictionary.")

        state = dict(state)  # make a copy
        state["last_saved"] = datetime.utcnow().isoformat()

        # Write atomically to avoid corruption if interrupted mid-write
        dir_name = os.path.dirname(self.file_path) or "."
        os.makedirs(dir_name, exist_ok=True)

        temp_fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix=".tmp_state_", text=True)
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # ensure write to disk
            os.replace(temp_path, self.file_path)
            logger.debug(f"State saved successfully to {self.file_path}")
        except Exception as e:
            logger.exception("Error saving state")
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise e

    def load_state(self):
        """Load and validate persisted state safely."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate structure — must be a dictionary
            if not isinstance(data, dict):
                logger.warning("Invalid state file format: not a JSON object. Resetting state.")
                return {}

            # Optional: sanity-check known keys
            if "last_saved" in data and not isinstance(data["last_saved"], str):
                logger.warning("Corrupted 'last_saved' value detected. Removing key.")
                data.pop("last_saved", None)

            return data

        except FileNotFoundError:
            logger.info("No existing state file found; starting fresh.")
            return {}
        except json.JSONDecodeError:
            logger.warning(f"State file '{self.file_path}' is not valid JSON; resetting.")
            return {}
        except Exception as e:
            logger.exception(f"Unexpected error reading state file: {e}")
            return {}

from typing import Callable, Dict, Any, List

class HookManager:
    """
    PredictFlow Hook Management Module

    Purpose:
      - Manage and execute user-defined or system hooks during workflow execution
      - Support pre-step and post-step hook registration
      - Allow external modules (e.g., logging, alerts, metrics) to plug into workflow events
      - Provide a clean extension point for custom behaviors without modifying the core engine
    """

    def __init__(self):
        # Maintain separate registries for different hook timings
        self.hooks = {
            "before_step": [],
            "after_step": [],
            "on_error": [],
            "on_complete": []
        }

    # ----------------------------
    # 1. Hook Registration
    # ----------------------------
    def register(self, event: str, func: Callable):
        """
        Register a new hook for a specific event.
        Supported events:
            - 'before_step'
            - 'after_step'
            - 'on_error'
            - 'on_complete'
        """
        if event not in self.hooks:
            raise ValueError(f"Invalid hook event '{event}'. Must be one of {list(self.hooks.keys())}.")
        self.hooks[event].append(func)

    # ----------------------------
    # 2. Hook Execution
    # ----------------------------
    def run(self, event: str, context: Dict[str, Any], step: Dict[str, Any] = None, error: Exception = None):
        """
        Execute all hooks for a given event.
        Each hook function should accept parameters:
            func(context, step) or func(context, step, error)
        """
        if event not in self.hooks:
            return

        for hook in self.hooks[event]:
            try:
                if event == "on_error":
                    hook(context, step, error)
                else:
                    hook(context, step)
            except Exception as e:
                print(f"Hook '{hook.__name__}' failed during '{event}': {e}")

    # ----------------------------
    # 3. Hook Management Utilities
    # ----------------------------
    def list_hooks(self) -> Dict[str, List[str]]:
        """Return a dictionary of all registered hooks by event name."""
        return {event: [f.__name__ for f in funcs] for event, funcs in self.hooks.items()}

    def clear_hooks(self):
        """Remove all registered hooks."""
        for event in self.hooks:
            self.hooks[event] = []

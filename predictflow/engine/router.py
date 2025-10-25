"""
Router module for PredictFlow
-----------------------------

Enables conditional and parallel routing between workflow steps.
Works alongside the Executor, which calls route_next() after each step.

Supports:
- Sequential flow (default)
- Conditional branching using 'when' clauses
- Parallel execution using 'parallel' arrays
"""

import threading
import traceback

class Router:
    def __init__(self, context):
        self.context = context

    def route_next(self, current_step, workflow_steps):
        """Determine the next step(s) after current_step."""
        next_steps = []

        # If current step defines conditional next steps
        if "next" in current_step:
            next_def = current_step["next"]

            # Case 1: Parallel execution
            if isinstance(next_def, dict) and "parallel" in next_def:
                for s in next_def["parallel"]:
                    next_steps.append(self._find_step(s, workflow_steps))

            # Case 2: Conditional routing
            elif isinstance(next_def, list):
                for branch in next_def:
                    cond = branch.get("when")
                    target = branch.get("step")
                    if self._evaluate_condition(cond):
                        next_steps.append(self._find_step(target, workflow_steps))
                        break

            # Case 3: Simple direct transition
            elif isinstance(next_def, str):
                next_steps.append(self._find_step(next_def, workflow_steps))

        return [s for s in next_steps if s]

    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------
    def _find_step(self, step_id, workflow_steps):
        for s in workflow_steps:
            if s.get("id") == step_id:
                return s
        print(f"⚠️  Step '{step_id}' not found in workflow.")
        return None

    def _evaluate_condition(self, expr: str):
        """Very simple safe evaluator for conditions."""
        if not expr:
            return False
        try:
            return bool(eval(expr, {}, self.context))
        except Exception:
            traceback.print_exc()
            return False

    def execute_parallel(self, steps, executor_func):
        """Run steps in parallel threads."""
        threads = []
        for step in steps:
            t = threading.Thread(target=executor_func, args=(step,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

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
import concurrent.futures
import math
import logging
import os

logger = logging.getLogger(__name__)

class Router:
    def __init__(self, context, *, max_workers=None, max_total_tasks=1000):
        """
        context: execution context dict
        max_workers: maximum number of threads for parallel execution
        max_total_tasks: hard cap on allowed parallel tasks (DoS protection)
        """
        self.context = context
        try:
            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1

        if max_workers is None:
            # Conservative default — no more than 32 threads
            self.max_workers = max(1, min(32, cpu_count * 5))
        else:
            self.max_workers = int(max(1, max_workers))

        self.max_total_tasks = int(max_total_tasks)

    # ----------------------------------------------------
    # Routing logic
    # ----------------------------------------------------
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

    # ----------------------------------------------------
    # Safe Parallel Execution
    # ----------------------------------------------------
    def execute_parallel(self, steps, executor_func, *, max_workers=None, max_total_tasks=None):
        """
        Run steps in parallel with resource limits.

        Args:
            steps (list): Steps to execute in parallel.
            executor_func (callable): Function to execute for each step.
            max_workers (int, optional): Override concurrency limit.
            max_total_tasks (int, optional): Override total step cap.
        """
        if not steps:
            return

        # Resolve limits
        max_workers = int(max_workers or self.max_workers)
        max_total_tasks = int(max_total_tasks or self.max_total_tasks)

        # Hard cap to avoid DoS attacks
        if len(steps) > max_total_tasks:
            raise RuntimeError(
                f"Refusing to execute {len(steps)} parallel steps — "
                f"exceeds safe maximum of {max_total_tasks}."
            )

        # Process in bounded chunks to avoid huge queue buildup
        chunk_size = max(1, max_workers * 4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for start in range(0, len(steps), chunk_size):
                chunk = steps[start:start + chunk_size]
                futures = []

                for step in chunk:
                    futures.append(executor.submit(executor_func, step))

                done, not_done = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_EXCEPTION
                )

                # If any task raises an exception, cancel remaining and re-raise
                for f in done:
                    exc = f.exception()
                    if exc:
                        for nf in not_done:
                            try:
                                nf.cancel()
                            except Exception:
                                pass
                        logger.exception("Parallel step execution failed")
                        raise exc

        logger.debug(f"Executed {len(steps)} steps in parallel safely.")

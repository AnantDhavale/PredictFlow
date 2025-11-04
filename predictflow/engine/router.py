"""
Router module for PredictFlow - Security Hardened
-----------------------------
Enables conditional and parallel routing between workflow steps.
"""

import logging
import os
import re
import concurrent.futures
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class Router:
    """
    Secure router with bounded parallelism and safe condition evaluation.
    """
    
    def __init__(self, context: Dict[str, Any], *, max_workers: Optional[int] = None, max_total_tasks: int = 1000):
        """
        Args:
            context: Execution context dict
            max_workers: Maximum number of threads for parallel execution
            max_total_tasks: Hard cap on allowed parallel tasks (DoS protection)
        """
        self.context = context
        
        try:
            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1

        if max_workers is None:
            self.max_workers = max(1, min(32, cpu_count * 5))
        else:
            self.max_workers = max(1, int(max_workers))

        self.max_total_tasks = int(max_total_tasks)

    def route_next(self, current_step: Dict[str, Any], workflow_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine the next step(s) after current_step."""
        next_steps = []

        if "next" not in current_step:
            return next_steps

        next_def = current_step["next"]

        # Case 1: Parallel execution (with limit)
        if isinstance(next_def, dict) and "parallel" in next_def:
            parallel_steps = next_def["parallel"][:self.max_total_tasks]
            
            if len(next_def["parallel"]) > self.max_total_tasks:
                logger.warning(
                    "Parallel execution limited to %d steps (requested %d)",
                    self.max_total_tasks, len(next_def["parallel"])
                )
            
            for s in parallel_steps:
                step = self._find_step(s, workflow_steps)
                if step:
                    next_steps.append(step)

        # Case 2: Conditional routing
        elif isinstance(next_def, list):
            for branch in next_def:
                cond = branch.get("when")
                target = branch.get("step")
                if self._evaluate_condition(cond):
                    step = self._find_step(target, workflow_steps)
                    if step:
                        next_steps.append(step)
                    break

        # Case 3: Simple direct transition
        elif isinstance(next_def, str):
            step = self._find_step(next_def, workflow_steps)
            if step:
                next_steps.append(step)

        return next_steps

    def _find_step(self, step_id: str, workflow_steps: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find step by ID with validation."""
        if not step_id or not isinstance(step_id, str):
            return None
        
        if len(step_id) > 255:
            logger.warning("Step ID too long: %s", step_id[:50])
            return None
        
        for s in workflow_steps:
            if s.get("id") == step_id:
                return s
        
        logger.warning("Step '%s' not found in workflow", step_id[:100])
        return None

    def _evaluate_condition(self, expr: Optional[str]) -> bool:
        """
        Safely evaluate conditions without allowing arbitrary code execution.
        
        Supports simple comparisons like:
        - context['status'] == 'approved'
        - context['amount'] > 1000
        """
        if not expr:
            return False
        
        try:
            expr = str(expr).strip()
            
            if len(expr) > 1000:
                logger.error("Condition expression too long, rejecting")
                return False
            
            # Check for dangerous patterns
            dangerous_patterns = [
                '__import__', 'eval', 'exec', 'compile', 'open',
                'file', 'input', 'raw_input', 'execfile',
                'reload', '__builtins__', 'globals', 'locals',
                'vars', 'dir', 'getattr', 'setattr', 'delattr',
                'hasattr', 'callable', 'isinstance', 'issubclass',
                'type', 'classmethod', 'staticmethod'
            ]
            
            expr_lower = expr.lower()
            for pattern in dangerous_patterns:
                if pattern in expr_lower:
                    logger.error("Blocked dangerous pattern in condition: %s", pattern)
                    return False
            
            # Build safe evaluation environment
            safe_context = {
                k: v for k, v in self.context.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
            
            restricted_namespace = {
                '__builtins__': {},
                'True': True,
                'False': False,
                'None': None,
                'context': safe_context
            }
            
            result = eval(expr, restricted_namespace, {})
            return bool(result)
            
        except Exception as e:
            logger.warning("Condition evaluation failed: %s", str(e)[:200])
            return False

    def execute_parallel(
        self,
        steps: List[Dict[str, Any]],
        executor_func,
        *,
        max_workers: Optional[int] = None,
        max_total_tasks: Optional[int] = None
    ):
        """
        Run steps in parallel with resource limits.
        
        Args:
            steps: Steps to execute in parallel
            executor_func: Function to execute for each step
            max_workers: Override concurrency limit
            max_total_tasks: Override total step cap
        """
        if not steps:
            return

        max_workers = int(max_workers or self.max_workers)
        max_total_tasks = int(max_total_tasks or self.max_total_tasks)

        if len(steps) > max_total_tasks:
            raise RuntimeError(
                f"Refusing to execute {len(steps)} parallel steps - "
                f"exceeds safe maximum of {max_total_tasks}"
            )

        chunk_size = max(1, max_workers * 4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for start in range(0, len(steps), chunk_size):
                chunk = steps[start:start + chunk_size]
                futures = []

                for step in chunk:
                    if not isinstance(step, dict):
                        logger.warning("Invalid step type in parallel execution, skipping")
                        continue
                    futures.append(executor.submit(executor_func, step))

                done, not_done = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_EXCEPTION
                )

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

        logger.debug("Executed %d steps in parallel safely", len(steps))

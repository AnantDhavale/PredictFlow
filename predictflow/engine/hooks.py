"""
HookManager Module for PredictFlow
----------------------------------

Purpose:
    - Manage and execute user-defined or system hooks during workflow execution
    - Support pre-step and post-step hook registration
    - Allow external modules (logging, alerts, metrics) to plug into workflow events
    - Provide a safe extension point for custom behaviors without modifying the core engine

Security/Resilience features:
    - Validates hook event names and function callables
    - Prevents double-registration
    - Thread-safe registration/execution
    - Isolates hook errors (one bad hook can’t crash workflow)
    - Optional timeout and limited execution for untrusted hooks
    - Sanitized logging (no code injection)
"""

import logging
import inspect
import threading
import traceback
from typing import Callable, Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = logging.getLogger(__name__)

# Configurable security controls
MAX_HOOKS_PER_EVENT = 20           # prevent over-registration (DoS guard)
HOOK_TIMEOUT_SEC = 5.0             # maximum time allowed per hook execution
ALLOWED_EVENTS = {"before_step", "after_step", "on_error", "on_complete"}

class HookManager:
    """Secure, thread-safe hook manager for PredictFlow."""

    def __init__(self):
        # Registry: {event_name: [callables]}
        self._hooks: Dict[str, List[Callable]] = {event: [] for event in ALLOWED_EVENTS}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=5)

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

        Raises:
            ValueError if event invalid or registration exceeds cap.
        """
        if not callable(func):
            raise ValueError("Hook must be callable.")

        if event not in self._hooks:
            raise ValueError(f"Invalid hook event '{event}'. Must be one of {sorted(ALLOWED_EVENTS)}.")

        with self._lock:
            if func in self._hooks[event]:
                logger.warning("Hook '%s' already registered for event '%s'.", func.__name__, event)
                return

            if len(self._hooks[event]) >= MAX_HOOKS_PER_EVENT:
                raise RuntimeError(f"Too many hooks registered for '{event}'. Cap is {MAX_HOOKS_PER_EVENT}.")

            self._hooks[event].append(func)
            logger.info("✅ Hook '%s' registered for event '%s'.", func.__name__, event)

    # ----------------------------
    # 2. Hook Execution
    # ----------------------------
    def run(
        self,
        event: str,
        context: Dict[str, Any],
        step: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ):
        """
        Execute all hooks for a given event.

        Each hook must have a compatible signature:
            - func(context, step)
            - func(context, step, error)  (for on_error)
        """
        if event not in self._hooks:
            logger.debug("Unknown event '%s' — skipping hook execution.", event)
            return

        hooks = list(self._hooks[event])  # copy for thread safety
        if not hooks:
            return

        logger.debug("Executing %d hook(s) for event '%s'.", len(hooks), event)

        for hook in hooks:
            self._execute_hook_safe(event, hook, context, step, error)

    def _execute_hook_safe(
        self,
        event: str,
        hook: Callable,
        context: Dict[str, Any],
        step: Optional[Dict[str, Any]],
        error: Optional[Exception]
    ):
        """Execute a single hook safely and with timeout."""
        try:
            sig = inspect.signature(hook)
            params = len(sig.parameters)
            args = (context, step) if params < 3 else (context, step, error)

            # Run hook with timeout (thread pool)
            future = self._executor.submit(hook, *args)
            future.result(timeout=HOOK_TIMEOUT_SEC)
        except TimeoutError:
            logger.error("⏱️ Hook '%s' timed out after %.1f sec.", hook.__name__, HOOK_TIMEOUT_SEC)
        except Exception as e:
            msg = str(e).replace("\n", " ").strip()
            logger.error("❌ Hook '%s' failed during '%s': %s", hook.__name__, event, msg)
            traceback.print_exc()

    # ----------------------------
    # 3. Hook Management Utilities
    # ----------------------------
    def list_hooks(self) -> Dict[str, List[str]]:
        """Return a dictionary of all registered hooks by event name."""
        with self._lock:
            return {event: [h.__name__ for h in funcs] for event, funcs in self._hooks.items()}

    def clear_hooks(self, event: Optional[str] = None):
        """Remove all registered hooks for a specific event or all events."""
        with self._lock:
            if event:
                if event not in self._hooks:
                    raise ValueError(f"Invalid event '{event}'.")
                self._hooks[event].clear()
                logger.info("Cleared hooks for event '%s'.", event)
            else:
                for ev in self._hooks:
                    self._hooks[ev].clear()
                logger.info("Cleared all registered hooks.")

    def shutdown(self, wait: bool = True):
        """Gracefully shut down internal executor."""
        self._executor.shutdown(wait=wait)
        logger.info("HookManager executor shutdown complete.")

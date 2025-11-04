"""
Scheduler Module for PredictFlow
--------------------------------
Adds time-based step handling to workflows.

Supported YAML keys:
- wait: "10s", "2m", "1h"   → pauses step for given duration
- retry: 3                  → retries the step on failure
- start_after: "30s"        → schedule step to begin in the future
- cron: "0 9 * * *"         → (future) schedule for periodic workflows

Security/Resilience features:
- Bounded and validated durations (cap long waits)
- Bounded retries
- ThreadPoolExecutor with max workers (no unbounded threads)
- Rate limiter to prevent DoS from rapid scheduling
- Validation of step schema/keys
- Safe logging (avoid log injection)
- Graceful shutdown of scheduled tasks
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Event, Lock
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# ------------------------------
# Configurable safety limits
# ------------------------------
MAX_SCHEDULER_THREADS = 10           # Max concurrent scheduler worker threads
MAX_WAIT_SECONDS = 6 * 3600          # 6 hours cap on any single wait or start_after
MAX_RETRIES = 10                     # Cap retries for safety
MAX_STEPS_PER_WORKFLOW = 200         # Optional: sanity limit per workflow instance
MAX_PENDING_SCHEDULED = 100          # Max futures held/scheduled at once

# Rate limiter defaults (token-bucket)
DEFAULT_RATE_PER_SEC = 5.0           # allow 5 schedule operations/sec
DEFAULT_BURST_CAPACITY = 10          # allow short bursts up to 10 tokens


# ------------------------------
# Helpers
# ------------------------------
def _sanitize_log_val(val: Any) -> str:
    """Remove control characters to reduce log injection/noise."""
    s = str(val)
    return s.replace("\n", " ").replace("\r", " ").strip()


def parse_duration(duration_str: Optional[str]) -> float:
    """
    Convert duration strings ('10s','2m','1h') to seconds with validation.
    Caps to MAX_WAIT_SECONDS. Returns 0.0 on invalid.
    """
    if not duration_str:
        return 0.0

    try:
        duration_str = duration_str.strip().lower()
        if len(duration_str) < 2:
            raise ValueError("too short")

        unit = duration_str[-1]
        value = float(duration_str[:-1])

        if value < 0:
            raise ValueError("negative durations are not allowed")

        if unit == "s":
            seconds = value
        elif unit == "m":
            seconds = value * 60
        elif unit == "h":
            seconds = value * 3600
        else:
            raise ValueError(f"unsupported duration unit: {unit}")

        if seconds > MAX_WAIT_SECONDS:
            logger.warning(
                "Wait duration %ss exceeds cap; clamping to %ss",
                seconds, MAX_WAIT_SECONDS
            )
            seconds = float(MAX_WAIT_SECONDS)

        return float(seconds)
    except Exception as e:
        logger.error("Invalid duration string '%s': %s", duration_str, e)
        return 0.0


def _validate_step_schema(step: Dict[str, Any]) -> None:
    """
    Validate that a step contains only expected keys and that values are sane.
    Raises ValueError on violations.
    """
    allowed_keys = {
        "id", "action", "wait", "retry", "start_after", "cron",
        # add other keys your engine supports here (e.g., "next", "parallel")
    }
    unexpected = [k for k in step.keys() if k not in allowed_keys]
    if unexpected:
        raise ValueError(f"Unexpected keys in step: {unexpected}")

    # Validate wait/start_after if present
    for key in ("wait", "start_after"):
        if key in step and step[key] is not None:
            if not isinstance(step[key], str):
                raise ValueError(f"'{key}' must be a string like '10s', '2m', '1h'")
            # parse to ensure format is valid
            _ = parse_duration(step[key])  # will log/return 0 for invalid

    # Validate retry
    if "retry" in step and step["retry"] is not None:
        try:
            retries = int(step["retry"])
        except Exception:
            raise ValueError("'retry' must be an integer")
        if retries < 0:
            raise ValueError("'retry' must be >= 0")
        if retries > MAX_RETRIES:
            # Don't error; clamp later in run_with_retry
            logger.warning(
                "Retry count %s exceeds cap; it will be clamped to %s",
                retries, MAX_RETRIES
            )


# ------------------------------
# Token-bucket Rate Limiter
# ------------------------------
class RateLimiter:
    """
    Simple token-bucket rate limiter.
    rate_per_sec: average tokens refilled per second
    capacity: max burst size
    """
    def __init__(self, rate_per_sec: float, capacity: int):
        if rate_per_sec <= 0 or capacity <= 0:
            raise ValueError("rate_per_sec and capacity must be > 0")
        self.rate = float(rate_per_sec)
        self.capacity = int(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        delta = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.capacity, self._tokens + delta * self.rate)

    def acquire(self, n: float = 1.0, timeout: float = 0.0) -> bool:
        """
        Attempt to take n tokens. If not enough tokens are available,
        wait up to 'timeout' seconds for tokens to refill.
        Returns True if acquired, False otherwise.
        """
        end = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    return True
            if time.monotonic() >= end:
                return False
            # Sleep a small amount before trying again
            time.sleep(0.01)


# ------------------------------
# Scheduler
# ------------------------------
class Scheduler:
    def __init__(
        self,
        context: Dict[str, Any],
        *,
        max_workers: int = MAX_SCHEDULER_THREADS,
        rate_per_sec: float = DEFAULT_RATE_PER_SEC,
        burst_capacity: int = DEFAULT_BURST_CAPACITY,
        max_pending_scheduled: int = MAX_PENDING_SCHEDULED,
    ):
        self.context = context
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._shutdown_event = Event()
        self._futures: List[Future] = []
        self._futures_lock = Lock()
        self._rate_limiter = RateLimiter(rate_per_sec, burst_capacity)
        self._max_pending = int(max_pending_scheduled)

    # ------------------------------------------------------------
    # WAIT HANDLING
    # ------------------------------------------------------------
    def handle_wait(self, step: Dict[str, Any]) -> None:
        """Pause workflow if 'wait' key is present (bounded and validated)."""
        _validate_step_schema(step)
        wait_val = (step.get("wait") or "").strip()
        if not wait_val:
            return

        seconds = parse_duration(wait_val)
        if seconds <= 0:
            logger.debug("No valid wait specified; continuing.")
            return

        display = _sanitize_log_val(wait_val)
        logger.info("⏳ Waiting for %s (%.2fs) before continuing...", display, seconds)

        # Respect shutdown signal during long waits
        remaining = float(seconds)
        while remaining > 0 and not self._shutdown_event.is_set():
            chunk = min(1.0, remaining)  # sleep in 1s chunks to allow quick shutdown
            time.sleep(chunk)
            remaining -= chunk

        if self._shutdown_event.is_set():
            logger.info("Wait interrupted due to shutdown signal.")

    # ------------------------------------------------------------
    # RETRY LOGIC
    # ------------------------------------------------------------
    def run_with_retry(
        self,
        func: Callable[[], Any],
        retries: int = 0,
        delay: float = 0.0
    ) -> Any:
        """Execute a function with retry support and capped attempts."""
        try:
            retries = int(retries)
        except Exception:
            retries = 0
        retries = max(0, min(retries, MAX_RETRIES))
        delay = max(0.0, float(delay))

        attempt = 0
        while True:
            try:
                return func()
            except Exception as e:
                attempt += 1
                logger.warning("Retry %d/%d failed: %s", attempt, retries, e)
                traceback.print_exc()
                if attempt > retries or self._shutdown_event.is_set():
                    logger.error("❌ Max retries reached or shutdown signaled; raising.")
                    raise
                # bounded sleep that respects shutdown
                slept = 0.0
                while slept < delay and not self._shutdown_event.is_set():
                    chunk = min(0.5, delay - slept)
                    time.sleep(chunk)
                    slept += chunk

    # ------------------------------------------------------------
    # FUTURE EXECUTION
    # ------------------------------------------------------------
    def schedule_future(
        self,
        step: Dict[str, Any],
        executor_func: Callable[[Dict[str, Any]], Any],
        *,
        acquire_timeout: float = 0.1
    ) -> None:
        """
        Schedule step for future execution after 'start_after' delay.
        Uses a rate limiter to avoid abuse and a bounded thread pool.
        """
        _validate_step_schema(step)

        start_after = (step.get("start_after") or "").strip()
        if not start_after:
            return

        # Rate-limit scheduling attempts (non-blocking or short wait)
        if not self._rate_limiter.acquire(1.0, timeout=acquire_timeout):
            sid = _sanitize_log_val(step.get("id", "<unknown>"))
            logger.warning(
                "Rate limit exceeded; refusing to schedule step '%s' right now.",
                sid
            )
            return

        seconds = parse_duration(start_after)
        if seconds <= 0:
            return

        # Cap overall number of pending scheduled tasks
        with self._futures_lock:
            # Prune completed futures
            self._futures = [f for f in self._futures if not f.done()]
            if len(self._futures) >= self._max_pending:
                sid = _sanitize_log_val(step.get("id", "<unknown>"))
                logger.warning(
                    "Too many pending scheduled tasks (%d); skipping step '%s'.",
                    len(self._futures), sid
                )
                return

        step_id = _sanitize_log_val(step.get("id", "<unknown>"))
        logger.info("🕓 Step '%s' scheduled to run in %.2fs.", step_id, seconds)

        def delayed_run():
            try:
                # Sleep in small increments to allow fast shutdown
                remaining = float(seconds)
                while remaining > 0 and not self._shutdown_event.is_set():
                    chunk = min(1.0, remaining)
                    time.sleep(chunk)
                    remaining -= chunk

                if self._shutdown_event.is_set():
                    logger.info("Scheduled step '%s' cancelled due to shutdown.", step_id)
                    return

                logger.debug("Executing scheduled step '%s'.", step_id)
                executor_func(step)
            except Exception as e:
                logger.exception("Error executing scheduled step '%s': %s", step_id, e)

        try:
            fut = self._executor.submit(delayed_run)
            with self._futures_lock:
                self._futures.append(fut)
        except Exception as e:
            logger.error("Failed to schedule step '%s': %s", step_id, e)

    # ------------------------------------------------------------
    # BULK VALIDATION (optional utility)
    # ------------------------------------------------------------
    def validate_workflow_steps(self, steps: Iterable[Dict[str, Any]]) -> None:
        """
        Optional helper to validate a collection of steps prior to execution.
        Enforces a soft cap on number of steps.
        """
        steps = list(steps)
        if len(steps) > MAX_STEPS_PER_WORKFLOW:
            logger.warning(
                "Workflow has %d steps; exceeds soft cap of %d.",
                len(steps), MAX_STEPS_PER_WORKFLOW
            )
        for s in steps:
            _validate_step_schema(s)

    # ------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------
    def shutdown(self, wait: bool = True) -> None:
        """
        Gracefully shut down scheduler:
        - Signal shutdown to stop waits and scheduled sleeps
        - Wait for in-flight tasks if requested
        - Shutdown thread pool
        """
        self._shutdown_event.set()

        # Optionally wait for scheduled tasks to finish/cancel
        if wait:
            with self._futures_lock:
                futures = list(self._futures)
            for f in futures:
                try:
                    f.result(timeout=2.0)  # don't block forever
                except Exception:
                    # Ignore errors; they were already logged in worker
                    pass

        self._executor.shutdown(wait=wait)
        logger.info("Scheduler shutdown complete.")

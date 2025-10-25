"""
Scheduler Module for PredictFlow
--------------------------------
Adds time-based step handling to workflows.

Supported YAML keys:
- wait: "10s", "2m", "1h"  → pauses step for given duration
- retry: 3                 → retries the step on failure
- cron: "0 9 * * *"        → (future) schedule for periodic workflows
"""

import time
import traceback
import threading
from datetime import datetime, timedelta


def parse_duration(duration_str: str) -> float:
    """Convert '10s', '2m', '1h' into seconds."""
    if not duration_str:
        return 0
    try:
        unit = duration_str[-1]
        value = float(duration_str[:-1])
        if unit == "s":
            return value
        elif unit == "m":
            return value * 60
        elif unit == "h":
            return value * 3600
    except Exception:
        traceback.print_exc()
    return 0


class Scheduler:
    def __init__(self, context):
        self.context = context
        self.scheduled_tasks = []

    # ------------------------------------------------------------
    # WAIT HANDLING
    # ------------------------------------------------------------
    def handle_wait(self, step):
        """Pause workflow if 'wait' key is present."""
        wait_val = step.get("wait")
        if not wait_val:
            return
        duration = parse_duration(wait_val)
        if duration > 0:
            print(f"⏳ Waiting for {wait_val} before continuing...")
            time.sleep(duration)

    # ------------------------------------------------------------
    # RETRY LOGIC
    # ------------------------------------------------------------
    def run_with_retry(self, func, retries: int = 0, delay: float = 0):
        """Execute a function with retry support."""
        attempt = 0
        while True:
            try:
                return func()
            except Exception as e:
                attempt += 1
                print(f"Retry {attempt}/{retries}: {e}")
                traceback.print_exc()
                if attempt >= retries:
                    print("❌ Max retries reached.")
                    raise
                time.sleep(delay)

    # ------------------------------------------------------------
    # ASYNC / FUTURE EXECUTION
    # ------------------------------------------------------------
    def schedule_future(self, step, executor_func):
        """For steps with a 'start_after' key — schedule execution in future."""
        start_after = step.get("start_after")
        if not start_after:
            return

        duration = parse_duration(start_after)
        if duration <= 0:
            return

        def delayed_run():
            time.sleep(duration)
            executor_func(step)

        t = threading.Thread(target=delayed_run)
        t.start()
        self.scheduled_tasks.append(t)
        print(f"🕓 Step '{step.get('id')}' scheduled to run in {start_after}.")


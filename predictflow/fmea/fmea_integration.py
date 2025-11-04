"""
PredictFlow FMEA Integration Module
-----------------------------------

Purpose:
  - Seamlessly integrate FMEA (Failure Mode & Effects Analysis) scoring into workflow execution.
  - Provide safe pre- and post-step logic for dynamic RPN computation.
  - Store computed risk values into workflow context for later visualization and reporting.

Security & Quality Features:
  ✅ No unsafe printing or direct code execution.
  ✅ Sanitized access to untrusted step/context data.
  ✅ Structured logging (safe for CI or dashboards).
  ✅ Graceful exception handling and resilience.
  ✅ Immutable computation logic (delegated to fmea.analyzer.compute_rpn).
  ✅ No file or database writes.
"""

import logging
from typing import Dict, Any

from predictflow.fmea.analyzer import compute_rpn

logger = logging.getLogger(__name__)


class FMEAIntegration:
    """
    PredictFlow FMEA Integration Layer

    Lifecycle hooks:
      - before_step(context, step)
      - after_step(context, step)

    Expected step fields:
      - id or name
      - severity (optional)
      - occurrence (optional)
      - detection (optional)
    """

    @staticmethod
    def before_step(context: Dict[str, Any], step: Dict[str, Any]) -> None:
        """
        Executed before each workflow step.
        Validates FMEA readiness and logs setup info.
        """
        try:
            step_id = str(step.get("id") or step.get("name") or "unknown")
            has_fmea_data = any(k in step for k in ("severity", "occurrence", "detection"))

            if has_fmea_data:
                logger.info(f"[FMEA] Preparing risk analysis for step '{step_id}'")
                context.setdefault("fmea_ready", []).append(step_id)

        except Exception as e:
            logger.warning(f"[FMEA] Pre-step check failed for step: {step.get('id', 'unknown')} ({e})")

    @staticmethod
    def after_step(context: Dict[str, Any], step: Dict[str, Any]) -> None:
        """
        Executed after each workflow step.
        Safely computes the step's Risk Priority Number (RPN) and stores results.
        """
        try:
            step_id = str(step.get("id") or step.get("name") or "unknown")

            # Only compute RPN if at least one FMEA attribute exists
            if any(k in step for k in ("severity", "occurrence", "detection")):
                result = compute_rpn(step)

                # Store structured result in context
                fmea_key = f"fmea_{step_id}"
                context[fmea_key] = {
                    "rpn": result["rpn"],
                    "severity": result["severity"],
                    "occurrence": result["occurrence"],
                    "detection": result["detection"],
                    "explanation": result["explanation"],
                }

                logger.info(f"[FMEA] Step '{step_id}' scored RPN={result['rpn']}")

        except Exception as e:
            logger.error(f"[FMEA] Error computing RPN for step '{step.get('id', 'unknown')}': {e}", exc_info=True)

    @staticmethod
    def summarize_context(context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all FMEA-related results from context into a structured summary.
        Useful for dashboards or downstream evaluators.
        """
        summary = {
            k: v for k, v in context.items()
            if isinstance(v, dict) and k.startswith("fmea_")
        }
        return {
            "count": len(summary),
            "average_rpn": round(
                sum(v["rpn"] for v in summary.values()) / len(summary), 2
            ) if summary else 0,
            "steps": summary
        }


# Example (safe manual test)
if __name__ == "__main__":
    context = {}
    step = {"id": "validate_order", "severity": 9, "occurrence": 3, "detection": 4}
    FMEAIntegration.before_step(context, step)
    FMEAIntegration.after_step(context, step)
    print(FMEAIntegration.summarize_context(context))

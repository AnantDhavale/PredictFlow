"""
PredictFlow Evaluator Module (Hardened)
--------------------------------------

Purpose:
  - Analyze workflow execution results
  - Compute overall health score and performance metrics
  - Identify critical path and anomalies
  - Quantify process reliability using FMEA + NLP confidence
  - Provide structured insights for dashboards or APIs

Security/Resilience features:
  - Validates and clamps numeric inputs from untrusted context
  - Ignores non-finite numbers (NaN/Inf)
  - Caps items processed to mitigate DoS on oversized contexts
  - Sanitizes printed output to avoid log noise/injection
  - Robust error handling; never raises from evaluate()
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# ------------------------------
# Tunables / Safety Caps
# ------------------------------
MAX_CONTEXT_ITEMS = 50_000      # hard upper bound on items to scan from context
MAX_RISK_RPN = 1000.0           # theoretical FMEA RPN upper bound (10*10*10)
MAX_CONFIDENCE = 100.0          # expected confidence upper bound (percentage)
MIN_CONFIDENCE = 0.0
CRITICAL_TOP_N = 3              # how many "critical path" items to report


def _is_str(x: Any) -> bool:
    return isinstance(x, str)


def _sanitize_for_log(s: Any) -> str:
    """Remove control chars to keep logs tidy and avoid multi-line injections."""
    text = str(s)
    return text.replace("\n", " ").replace("\r", " ").strip()


def _to_finite_float(v: Any) -> float | None:
    """
    Convert a value to a finite float or return None if invalid.
    Rejects NaN/Inf to avoid poisoning statistics.
    """
    try:
        f = float(v)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class Evaluator:
    """
    PredictFlow Evaluator Module

    Notes on security:
      - Treats all values in `context` as untrusted input.
      - Only processes keys that end with '_confidence' or '_rpn' AND have finite numeric values.
      - Applies clamping to reasonable ranges to avoid skew/overflow attacks.
      - Caps the number of processed items so that a massive context cannot cause excessive CPU/memory use.
    """

    def __init__(self, context: Dict[str, Any]):
        """
        Initialize evaluator with workflow execution context.
        The context usually comes from Executor after a workflow run.
        """
        if not isinstance(context, dict):
            logger.warning("Evaluator initialized with non-dict context; substituting empty dict.")
            context = {}
        self.context = context
        self.results: Dict[str, Any] = {
            "workflow_health": None,
            "average_confidence": None,
            "average_risk": None,
            "critical_path": [],
            "alerts": []
        }

    # ----------------------------
    # 1. Main entry point
    # ----------------------------
    def evaluate(self) -> Dict[str, Any]:
        """
        Compute key metrics from the workflow context.

        This method is fault-tolerant and will not raise; on error it logs
        and returns best-effort results with safe defaults.
        """
        try:
            confidence_scores = self._extract_confidence_scores()
            rpn_values = self._extract_rpn_values()

            avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0
            avg_risk = statistics.mean(rpn_values) if rpn_values else 0.0

            # Compute overall health (higher is better)
            health_score = self._compute_health(avg_confidence, avg_risk)

            # Identify top N risky steps (critical path)
            critical_path = self._identify_critical_path()

            # Check for any alert conditions
            alerts = self._detect_alerts(avg_confidence, avg_risk)

            self.results.update({
                "workflow_health": round(health_score, 2),
                "average_confidence": round(avg_confidence, 2),
                "average_risk": round(avg_risk, 2),
                "critical_path": critical_path,
                "alerts": alerts
            })

        except Exception as e:
            # Never propagate; keep API stable and safe
            logger.exception("Evaluator encountered an error: %s", e)
            # keep defaults already in self.results

        self._print_results_safe()
        return self.results

    # ----------------------------
    # 2. Metric Computations
    # ----------------------------
    def _extract_confidence_scores(self) -> List[float]:
        """
        Extract all confidence scores from context keys (*_confidence).
        Sanitizes and clamps to [MIN_CONFIDENCE, MAX_CONFIDENCE].
        """
        scores: List[float] = []
        count = 0
        for k, v in self.context.items():
            count += 1
            if count > MAX_CONTEXT_ITEMS:
                logger.warning(
                    "Context item cap reached (%d). Remaining keys are ignored for safety.",
                    MAX_CONTEXT_ITEMS
                )
                break
            if not (_is_str(k) and k.endswith("_confidence")):
                continue
            f = _to_finite_float(v)
            if f is None:
                continue
            scores.append(_clamp(f, MIN_CONFIDENCE, MAX_CONFIDENCE))
        return scores

    def _extract_rpn_values(self) -> List[float]:
        """
        Extract all FMEA RPN values from context keys (*_rpn).
        Sanitizes and clamps to [0, MAX_RISK_RPN].
        """
        rpns: List[float] = []
        count = 0
        for k, v in self.context.items():
            count += 1
            if count > MAX_CONTEXT_ITEMS:
                logger.warning(
                    "Context item cap reached (%d). Remaining keys are ignored for safety.",
                    MAX_CONTEXT_ITEMS
                )
                break
            if not (_is_str(k) and k.endswith("_rpn")):
                continue
            f = _to_finite_float(v)
            if f is None:
                continue
            rpns.append(_clamp(f, 0.0, MAX_RISK_RPN))
        return rpns

    def _compute_health(self, confidence: float, risk: float) -> float:
        """
        Compute a composite 'workflow health' metric.
        Formula (tunable):
            health = confidence * (1 - normalized_risk)

        Risk normalization:
            We normalize risk onto 0..1 scale using MAX_RISK_RPN.
        """
        conf = _clamp(confidence, MIN_CONFIDENCE, MAX_CONFIDENCE)
        risk = _clamp(risk, 0.0, MAX_RISK_RPN)
        norm_risk = min(risk / MAX_RISK_RPN, 1.0)  # 0..1
        health = (conf / MAX_CONFIDENCE) * (1.0 - norm_risk) * 100.0
        return max(0.0, min(100.0, health))

    def _identify_critical_path(self) -> List[str]:
        """
        Find top N risky steps based on highest *_rpn values.

        Returns a list of step IDs (key base names without the '_rpn' suffix).
        """
        risky: List[Tuple[str, float]] = []
        count = 0
        for k, v in self.context.items():
            count += 1
            if count > MAX_CONTEXT_ITEMS:
                logger.warning(
                    "Context item cap reached (%d). Remaining keys are ignored for safety.",
                    MAX_CONTEXT_ITEMS
                )
                break
            if not (_is_str(k) and k.endswith("_rpn")):
                continue
            f = _to_finite_float(v)
            if f is None:
                continue
            f = _clamp(f, 0.0, MAX_RISK_RPN)
            base = k[:-4]  # strip "_rpn"
            risky.append((base, f))

        # Stable, deterministic sort: by risk desc, then name asc
        risky.sort(key=lambda x: (-x[1], x[0]))
        return [name for name, _ in risky[:CRITICAL_TOP_N]]

    def _detect_alerts(self, avg_confidence: float, avg_risk: float) -> List[str]:
        """
        Generate warnings based on process health conditions.
        Thresholds can be tuned as needed.
        """
        alerts: List[str] = []
        if avg_confidence < 60:
            alerts.append("Low average confidence across steps.")
        if avg_risk > 150:
            alerts.append("High overall risk detected. Review critical path steps.")
        if not alerts:
            alerts.append("Process stability within acceptable thresholds.")
        return alerts

    # ----------------------------
    # 3. Output Formatting
    # ----------------------------
    def _print_results_safe(self) -> None:
        """
        Print a clean summary of evaluation results with basic sanitization.
        Avoid leaking arbitrary context data; only print computed fields.
        """
        try:
            health = self.results.get("workflow_health")
            avg_conf = self.results.get("average_confidence")
            avg_risk = self.results.get("average_risk")
            crit = self.results.get("critical_path") or []
            alerts = self.results.get("alerts") or []

            crit_s = " -> ".join(_sanitize_for_log(x) for x in crit)

            print("\nPredictFlow Evaluation Summary")
            print("---------------------------------")
            print(f"Workflow Health Score : {health}")
            print(f"Average Confidence     : {avg_conf}")
            print(f"Average Risk (RPN)     : {avg_risk}")
            print(f"Critical Path (Top {CRITICAL_TOP_N})  : {crit_s}")
            for alert in alerts:
                print(_sanitize_for_log(alert))
            print("---------------------------------\n")
        except Exception as e:
            logger.exception("Failed to print evaluation summary: %s", e)

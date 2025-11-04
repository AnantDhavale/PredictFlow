"""
PredictFlow Confidence Metrics (Secure & Explainable)
-----------------------------------------------------

Purpose:
  - Compute workflow-level confidence and text similarity metrics
  - Aggregate adapter-based embedding scores into structured results
  - Provide interpretable measures for Evaluator and dashboards

Security & Quality Features:
  ✅ Deterministic local computation
  ✅ Bounded numeric outputs (0–100)
  ✅ Sanitized text and context inputs
  ✅ No code execution or network dependencies
  ✅ Graceful fallback for invalid or missing data
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from predictflow.confidence.adapters import ConfidenceAdapterFactory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Core Confidence Metric Engine
# ---------------------------------------------------------------------------
class ConfidenceMetrics:
    """
    PredictFlow Confidence Metric Computation Engine

    Responsibilities:
      - Compute text similarity-based confidence scores for workflow steps
      - Aggregate and normalize metrics for downstream evaluation
      - Provide structured, explainable results for dashboards
    """

    def __init__(self, adapter=None):
        """
        Initialize with a chosen confidence adapter (local or remote).
        Defaults to LocalConfidenceAdapter for offline safety.
        """
        self.adapter = adapter or ConfidenceAdapterFactory.get_adapter(prefer_remote=False)

    # -------------------------------------------------------------------
    # 2. Step-Level Metrics
    # -------------------------------------------------------------------
    def compute_step_confidence(self, step: Dict[str, Any]) -> float:
        """
        Compute confidence for a single workflow step.
        Uses text similarity between 'expected' and 'actual' or self-consistency.
        Returns a score ∈ [0, 100].
        """
        try:
            text_a = str(step.get("description") or step.get("action") or "").strip()
            text_b = str(step.get("output") or step.get("notes") or "").strip()

            if not text_a:
                return 50.0  # Neutral baseline when text is missing

            # Self-consistency confidence if no output to compare
            if not text_b:
                vector = self.adapter.embed(text_a)
                conf = float(np.mean(vector)) * 100
            else:
                score = self.adapter.compare(text_a, text_b)
                conf = score * 100

            return round(min(max(conf, 0), 100), 2)
        except Exception as e:
            logger.warning(f"[Confidence] Step confidence failed: {e}")
            return 0.0

    # -------------------------------------------------------------------
    # 3. Workflow-Level Aggregates
    # -------------------------------------------------------------------
    def compute_workflow_confidence(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregate confidence metrics for an entire workflow.
        Returns structured dict with averages and distribution.
        """
        if not isinstance(steps, list) or not steps:
            return {"average_confidence": 0.0, "distribution": [], "low_confidence_steps": []}

        scores = []
        low_conf_steps = []

        for step in steps:
            score = self.compute_step_confidence(step)
            scores.append(score)
            if score < 60:
                low_conf_steps.append(step.get("id", "unknown"))

        avg_conf = round(float(np.mean(scores)) if scores else 0.0, 2)
        std_dev = round(float(np.std(scores)) if len(scores) > 1 else 0.0, 2)

        return {
            "average_confidence": avg_conf,
            "std_dev": std_dev,
            "distribution": scores,
            "low_confidence_steps": low_conf_steps
        }

    # -------------------------------------------------------------------
    # 4. Pairwise Step Similarity Matrix
    # -------------------------------------------------------------------
    def build_similarity_matrix(self, steps: List[Dict[str, Any]]) -> np.ndarray:
        """
        Build a pairwise similarity matrix (n×n) across step descriptions.
        Useful for identifying redundant or inconsistent workflow actions.
        """
        try:
            texts = [str(s.get("description") or s.get("id") or "") for s in steps]
            n = len(texts)
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    score = self.adapter.compare(texts[i], texts[j])
                    matrix[i, j] = matrix[j, i] = round(score, 3)
            return matrix
        except Exception as e:
            logger.error(f"[Confidence] Failed to build similarity matrix: {e}")
            return np.zeros((1, 1))

    # -------------------------------------------------------------------
    # 5. Risk-Confidence Correlation
    # -------------------------------------------------------------------
    def correlate_with_rpn(self, steps: List[Dict[str, Any]]) -> Optional[float]:
        """
        Compute correlation (Pearson) between RPN and Confidence.
        Helps measure inverse relationship (expected: higher risk → lower confidence).
        """
        try:
            confidences = []
            rpns = []

            for step in steps:
                if "rpn" in step:
                    rpns.append(float(step["rpn"]))
                    confidences.append(self.compute_step_confidence(step))

            if len(rpns) < 2:
                return None

            corr = np.corrcoef(rpns, confidences)[0, 1]
            return round(float(corr), 3)
        except Exception as e:
            logger.warning(f"[Confidence] Failed correlation: {e}")
            return None


# ---------------------------------------------------------------------------
# 6. Example Standalone Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    steps_example = [
        {"id": "validate_invoice", "description": "Validate invoice details", "output": "Invoice validation successful"},
        {"id": "approve_payment", "description": "Approve payment order", "output": "Payment approved"},
        {"id": "notify_user", "description": "Send notification", "output": "Notification sent to user"},
        {"id": "risky_step", "description": "Handle exception", "output": "", "rpn": 240},
    ]

    metrics = ConfidenceMetrics()
    report = metrics.compute_workflow_confidence(steps_example)
    corr = metrics.correlate_with_rpn(steps_example)
    matrix = metrics.build_similarity_matrix(steps_example)

    print("=== Confidence Report ===")
    print(report)
    print("\nCorrelation (RPN ↔ Confidence):", corr)
    print("\nSimilarity Matrix:\n", matrix)

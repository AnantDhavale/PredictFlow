"""
PredictFlow Confidence Scorer (Secure & Deterministic)
------------------------------------------------------

Purpose:
  - Generate reproducible and explainable confidence scores for workflow steps
  - Support both simulated (deterministic pseudo-random) and adapter-based modes
  - Ensure no untrusted randomness or insecure seed leakage

Security & Design Features:
  ✅ Deterministic SHA-256 seeding for reproducible scoring
  ✅ Bounded numeric outputs (0.0–1.0)
  ✅ Optional adapter integration for semantic confidence
  ✅ No untrusted code execution or external dependency by default
"""

import hashlib
import logging
import numpy as np
from typing import Dict, Any, Optional

from predictflow.confidence.adapters import ConfidenceAdapterFactory

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    PredictFlow Confidence Scorer

    Modes:
      - "deterministic": secure hash-based pseudo-random (default)
      - "semantic": real NLP similarity via ConfidenceAdapter
    """

    def __init__(self, mode: str = "deterministic", adapter=None):
        self.mode = mode.lower().strip()
        self.adapter = adapter or ConfidenceAdapterFactory.get_adapter(prefer_remote=False)

    # ------------------------------------------------------------------
    # 1. Deterministic hash-based pseudo-random confidence
    # ------------------------------------------------------------------
    def _hash_confidence(self, text: str) -> float:
        """
        Generate a reproducible pseudo-random confidence score from text.
        Uses SHA-256 → float mapping in [0.4, 0.95].
        """
        try:
            clean_text = str(text or "").strip()
            if not clean_text:
                return 0.5  # neutral baseline

            h = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
            seed_int = int(h[:12], 16)
            rng = np.random.default_rng(seed_int)
            value = rng.random()
            # Map to a realistic range
            scaled = 0.4 + (value * 0.55)
            return round(float(min(max(scaled, 0.0), 1.0)), 2)
        except Exception as e:
            logger.error(f"[ConfidenceScorer] Hash-based confidence failed: {e}")
            return 0.5

    # ------------------------------------------------------------------
    # 2. Semantic similarity confidence (if enabled)
    # ------------------------------------------------------------------
    def _semantic_confidence(self, step: Dict[str, Any]) -> float:
        """
        Compute semantic confidence using text similarity.
        Requires adapter with embedding capability.
        """
        try:
            text_a = str(step.get("description") or step.get("action") or "")
            text_b = str(step.get("output") or step.get("notes") or "")
            if not text_a:
                return 0.5
            if not text_b:
                return self._hash_confidence(text_a)
            score = self.adapter.compare(text_a, text_b)
            return round(float(min(max(score, 0.0), 1.0)), 2)
        except Exception as e:
            logger.warning(f"[ConfidenceScorer] Semantic confidence failed: {e}")
            return 0.5

    # ------------------------------------------------------------------
    # 3. Main entry point
    # ------------------------------------------------------------------
    def compute_confidence(self, step: Dict[str, Any]) -> float:
        """
        Compute the confidence score for a workflow step.
        Secure, bounded, and reproducible across runs.
        """
        if self.mode == "semantic":
            return self._semantic_confidence(step)
        return self._hash_confidence(step.get("description") or step.get("id") or "")


# ----------------------------------------------------------------------
# 4. Standalone Utility Function
# ----------------------------------------------------------------------
def compute_confidence(step: Dict[str, Any], mode: str = "deterministic") -> float:
    """
    Lightweight wrapper for direct module-level use.
    """
    scorer = ConfidenceScorer(mode=mode)
    return scorer.compute_confidence(step)


# ----------------------------------------------------------------------
# 5. Example Usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_steps = [
        {"id": "validate_invoice", "description": "Validate invoice details"},
        {"id": "approve_payment", "description": "Approve payment order", "output": "Payment approved"},
        {"id": "send_email", "description": "Send notification email", "notes": "Email sent"},
    ]

    print("=== PredictFlow Confidence Scores ===")
    scorer_det = ConfidenceScorer(mode="deterministic")
    scorer_sem = ConfidenceScorer(mode="semantic")

    for step in test_steps:
        print(f"Step: {step['id']}")
        print(f"  Deterministic: {scorer_det.compute_confidence(step)}")
        print(f"  Semantic:      {scorer_sem.compute_confidence(step)}")
        print()

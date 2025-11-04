"""
PredictFlow FMEA Analyzer (Secure, Deterministic & Calibrated)
==============================================================

Purpose
--------
- Compute Failure Mode and Effects Analysis (FMEA) Risk Priority Numbers (RPN)
- Provide normalized scores to support cross-workflow comparisons
- Optionally calibrate deterministic scores against historical failure data

Security & Quality Features
---------------------------
✅ No random or non-deterministic elements — fully reproducible by default  
✅ Input validation and strict type enforcement  
✅ Safe handling of untrusted `step` data (prevents code injection)  
✅ Output bounded to known safe numeric ranges (1–10)  
✅ Graceful fallback if ML dependencies are unavailable  
✅ Compatible with PredictFlow auditing, dashboards, and reports
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

# Optional dependency: scikit-learn (declared in requirements.txt)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False
    LogisticRegression = Pipeline = StandardScaler = None  # type: ignore

logger = logging.getLogger(__name__)

# ----------------------------
# Constants
# ----------------------------
MIN_SCORE = 1
MAX_SCORE = 10
MIN_RPN = MIN_SCORE ** 3
MAX_RPN = MAX_SCORE ** 3


# ----------------------------
# Data Model
# ----------------------------
@dataclass(frozen=True)
class FMEAScore:
    """Structured container for FMEA results."""

    severity: int
    occurrence: int
    detection: int
    rpn: int
    normalized_rpn: float
    calibrated_risk: Optional[float]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable representation."""
        return asdict(self)


# ----------------------------
# Utility functions
# ----------------------------
def _safe_int(value: Any, default: int = 5) -> int:
    """Safely cast a value to an integer within 1–10 range."""
    try:
        num = int(value)
        if not (MIN_SCORE <= num <= MAX_SCORE):
            return default
        return num
    except (ValueError, TypeError):
        return default


def _hash_to_scale(text: str, low: int = MIN_SCORE, high: int = MAX_SCORE) -> int:
    """Map text deterministically to a numeric score in the given range."""
    if not isinstance(text, str):
        text = str(text)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    scaled = low + (int(h, 16) % (high - low + 1))
    return max(low, min(high, scaled))


def normalize_rpn(rpn: int, minimum: int = MIN_RPN, maximum: int = MAX_RPN) -> float:
    """Normalize RPN value onto a 0.0–1.0 scale."""
    try:
        value = int(rpn)
    except (TypeError, ValueError):
        value = minimum
    if maximum <= minimum:
        return 0.0
    clamped = max(minimum, min(maximum, value))
    normalized = (clamped - minimum) / (maximum - minimum)
    return round(float(normalized), 4)


# ----------------------------
# Calibrator
# ----------------------------
class FMEACalibrator:
    """Optional ML-assisted calibrator for deterministic FMEA scores."""

    def __init__(self, model: Optional[Pipeline] = None):
        self._model: Optional[Pipeline] = model
        self._is_fitted = False

        if self._model is None and SKLEARN_AVAILABLE:
            self._model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "logreg",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=1000,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )
        elif self._model is None:
            logger.warning(
                "[FMEA] ML calibration unavailable — scikit-learn not installed."
            )

    @property
    def is_ready(self) -> bool:
        return bool(self._model is not None and self._is_fitted)

    def fit(self, records: Sequence[Dict[str, Any]]) -> bool:
        """Fit the calibration model using historical data."""
        if not records:
            logger.warning("[FMEA] No records supplied for calibration.")
            return False
        if self._model is None:
            return False

        features, targets = [], []
        for record in records:
            if not isinstance(record, dict):
                continue
            s = _safe_int(record.get("severity"))
            o = _safe_int(record.get("occurrence"))
            d = _safe_int(record.get("detection"))
            rpn = record.get("rpn", s * o * d)
            normalized = normalize_rpn(rpn)
            outcome = self._extract_outcome(record)
            if outcome is None:
                continue
            features.append([s, o, d, normalized, float(rpn)])
            targets.append(outcome)

        if len(set(targets)) < 2:
            logger.warning("[FMEA] Need both success and failure samples.")
            return False

        try:
            self._model.fit(features, targets)
            self._is_fitted = True
            logger.info("[FMEA] Calibration model fitted (%d samples).", len(targets))
            return True
        except Exception as exc:  # pragma: no cover
            logger.error("[FMEA] Calibration failed: %s", exc)
            return False

    def predict_probability(
        self,
        *,
        severity: int,
        occurrence: int,
        detection: int,
        rpn: int,
        normalized_rpn: Optional[float] = None,
    ) -> Optional[float]:
        """Predict probability of failure for a scored step."""
        if not self.is_ready:
            return None
        normalized = (
            normalize_rpn(rpn)
            if normalized_rpn is None
            else max(0.0, min(1.0, float(normalized_rpn)))
        )
        vector = [[severity, occurrence, detection, normalized, float(rpn)]]
        try:
            proba = float(self._model.predict_proba(vector)[0][1])  # type: ignore
            return round(max(0.0, min(1.0, proba)), 4)
        except Exception as exc:  # pragma: no cover
            logger.error("[FMEA] Probability prediction failed: %s", exc)
            return None

    @staticmethod
    def _extract_outcome(record: Dict[str, Any]) -> Optional[int]:
        """Extract binary outcome (1=failure, 0=success) from record."""
        labels = (
            record.get("failure"),
            record.get("failed"),
            record.get("incident"),
            record.get("outcome"),
        )
        for value in labels:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(float(value) > 0.5)
        return None


# ----------------------------
# Main computation
# ----------------------------
def compute_rpn(step: Dict[str, Any], calibrator: Optional[FMEACalibrator] = None) -> Dict[str, Any]:
    """Compute deterministic RPN for a given workflow step."""
    safe_step = dict(step or {})
    step_key = str(safe_step.get("id") or safe_step.get("name") or "unknown")[:100]

    severity = (
        _safe_int(safe_step.get("severity"))
        if "severity" in safe_step
        else _hash_to_scale(step_key + "_s")
    )
    occurrence = (
        _safe_int(safe_step.get("occurrence"))
        if "occurrence" in safe_step
        else _hash_to_scale(step_key + "_o")
    )
    detection = (
        _safe_int(safe_step.get("detection"))
        if "detection" in safe_step
        else _hash_to_scale(step_key + "_d")
    )

    rpn = int(min(severity * occurrence * detection, MAX_RPN))
    normalized = normalize_rpn(rpn)
    calibrated: Optional[float] = None

    if calibrator and calibrator.is_ready:
        calibrated = calibrator.predict_probability(
            severity=severity,
            occurrence=occurrence,
            detection=detection,
            rpn=rpn,
            normalized_rpn=normalized,
        )

    explanation = (
        f"Step '{step_key}': S={severity}, O={occurrence}, D={detection} → "
        f"RPN={rpn}, normalized={normalized}, calibrated={calibrated}"
    )

    score = FMEAScore(
        severity=severity,
        occurrence=occurrence,
        detection=detection,
        rpn=rpn,
        normalized_rpn=normalized,
        calibrated_risk=calibrated,
        explanation=explanation,
    )

    return score.to_dict()


# ----------------------------
# Manual test
# ----------------------------
if __name__ == "__main__":
    example_step = {"id": "payment_approval", "severity": 9, "occurrence": 3, "detection": 4}
    print(compute_rpn(example_step))

    unknown_step = {"id": "auto_validate"}  # Missing numeric fields
    print(compute_rpn(unknown_step))

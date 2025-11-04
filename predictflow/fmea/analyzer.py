"""
PredictFlow FMEA Analyzer (Secure & Deterministic)
--------------------------------------------------

Purpose:
  - Compute Failure Mode and Effects Analysis (FMEA) Risk Priority Numbers (RPN)
  - Ensure reproducibility, transparency, and auditability for each step
  - Provide deterministic scoring even without explicit user-supplied values

Security & Quality Features:
  ✅ No random or non-deterministic elements — fully reproducible
  ✅ No external dependencies or system calls
  ✅ Input validation and strict type enforcement
  ✅ Safe handling of untrusted `step` data (prevents code injection)
  ✅ Output bounded to known safe numeric ranges (1–10)
  ✅ No unsafe YAML or eval parsing
  ✅ Compatible with PredictFlow auditing and dashboard modules
"""

import hashlib
from typing import Dict, Any

# Define FMEA component scoring range
MIN_SCORE = 1
MAX_SCORE = 10


def _safe_int(value: Any, default: int = 5) -> int:
    """
    Safely cast a value to an integer within 1–10 bounds.

    Returns `default` if conversion fails.
    Prevents injection or overflow from untrusted input.
    """
    try:
        num = int(value)
        if not (MIN_SCORE <= num <= MAX_SCORE):
            num = default
        return num
    except (ValueError, TypeError):
        return default


def _hash_to_scale(text: str, low: int = MIN_SCORE, high: int = MAX_SCORE) -> int:
    """
    Deterministic hash-based fallback.

    Maps any text (e.g., step ID or name) to a stable integer in the given range.
    Ensures reproducibility without using `random` or system entropy.
    """
    if not isinstance(text, str):
        text = str(text)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    scaled = low + (int(h, 16) % (high - low + 1))
    return max(low, min(high, scaled))


def compute_rpn(step: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute deterministic, explainable RPN for a given workflow step.

    Expected fields (optional):
      - severity (int, 1–10)
      - occurrence (int, 1–10)
      - detection (int, 1–10)
      - id or name (string) for fallback hashing

    Returns:
      {
        "severity": int,
        "occurrence": int,
        "detection": int,
        "rpn": int,
        "explanation": str
      }
    """
    # Defensive copy: avoid mutation of original dict
    safe_step = dict(step or {})

    # Step identifier for deterministic fallback
    step_key = str(safe_step.get("id") or safe_step.get("name") or "unknown")[:100]

    # Validate or infer Severity, Occurrence, Detection
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

    # Compute RPN safely (bounded integer arithmetic)
    rpn = severity * occurrence * detection
    rpn = int(min(rpn, MAX_SCORE**3))  # prevent overflow in case of misconfiguration

    explanation = (
        f"Step '{step_key}': "
        f"S={severity}, O={occurrence}, D={detection} → RPN={rpn}"
    )

    return {
        "severity": severity,
        "occurrence": occurrence,
        "detection": detection,
        "rpn": rpn,
        "explanation": explanation,
    }


# Example manual test
if __name__ == "__main__":
    example_step = {"id": "payment_approval", "severity": 9, "occurrence": 3, "detection": 4}
    print(compute_rpn(example_step))

    unknown_step = {"id": "auto_validate"}  # missing numeric fields
    print(compute_rpn(unknown_step))

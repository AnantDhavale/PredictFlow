"""
PredictFlow FMEA Report Generator (Secure & Auditable)
------------------------------------------------------

Purpose:
  - Summarize and export structured FMEA results from workflow context
  - Produce safe, validated JSON and text reports
  - Prevent arbitrary file writes and injection risks
  - Provide clear, auditable outputs for dashboards or APIs

Security Features:
  ✅ Strict context type checking and key filtering
  ✅ Sanitized step names and bounded output size
  ✅ Controlled file writing (safe path validation)
  ✅ Exception-safe JSON serialization
  ✅ No arbitrary file system access or code execution
"""

import json
import os
import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FMEAReport:
    """
    PredictFlow FMEA Report Generator

    Collects, sanitizes, and exports FMEA results post workflow execution.
    """

    MAX_STEPS = 5000  # prevent DoS or massive log flooding
    MAX_FILENAME_LEN = 100

    def __init__(self, context: Dict[str, Any]):
        """
        Initialize with a workflow context dictionary.
        Expected context keys: "<step>_rpn", "<step>_severity", etc.
        """
        if not isinstance(context, dict):
            raise TypeError("FMEAReport context must be a dictionary.")
        self.context = context
        self.results: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------------
    # 1. Report Builder
    # ----------------------------------------------------------------------
    def build_report(self) -> List[Dict[str, Any]]:
        """Create a list of step-level FMEA summaries with validation."""
        results = []
        for key, value in self.context.items():
            if not isinstance(key, str) or not key.endswith("_rpn"):
                continue

            step_name = key.replace("_rpn", "")
            if not re.match(r"^[\w\-]{1,50}$", step_name):
                logger.warning(f"[FMEA] Skipping invalid step name: {step_name}")
                continue

            severity = self.context.get(f"{step_name}_severity")
            occurrence = self.context.get(f"{step_name}_occurrence")
            detection = self.context.get(f"{step_name}_detection")
            normalized = self.context.get(f"{step_name}_normalized_rpn")
            calibrated = self.context.get(f"{step_name}_calibrated_risk")

            # Ensure numeric RPN value
            try:
                rpn_value = int(value)
            except (TypeError, ValueError):
                logger.warning(f"[FMEA] Invalid RPN for step {step_name}: {value}")
                continue

            results.append(
                {
                    "step": step_name,
                    "rpn": rpn_value,
                    "severity": self._safe_int(severity),
                    "occurrence": self._safe_int(occurrence),
                    "detection": self._safe_int(detection),
                    "normalized_rpn": self._safe_float(normalized),
                    "calibrated_risk": self._safe_float(calibrated),
                }
            )

            if len(results) >= self.MAX_STEPS:
                logger.warning("[FMEA] Step limit reached, truncating report.")
                break

        self.results = results
        return results

    # ----------------------------------------------------------------------
    # 2. JSON Export
    # ----------------------------------------------------------------------
    def to_json(self, file_path: Optional[str] = None) -> str:
        """
        Return the report as a JSON string.
        Optionally save to a file after safe path validation.
        """
        if not self.results:
            self.build_report()

        try:
            report_json = json.dumps(self.results, indent=2)
        except Exception as e:
            logger.error(f"[FMEA] Failed to serialize report: {e}")
            report_json = json.dumps({"error": "serialization_failed"})

        if file_path:
            if not self._is_safe_path(file_path):
                logger.error(f"[FMEA] Unsafe file path: {file_path}")
                return report_json

            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(report_json)
                logger.info(f"[FMEA] Report written to {file_path}")
            except Exception as e:
                logger.error(f"[FMEA] Failed to write report: {e}")

        return report_json

    # ----------------------------------------------------------------------
    # 3. Text Export
    # ----------------------------------------------------------------------
    def to_text(self) -> str:
        """Return a plain-text formatted report (for logs or console output)."""
        if not self.results:
            self.build_report()

        lines = ["FMEA Step Report", "----------------"]
        for r in self.results:
            lines.append(f"Step: {self._sanitize_name(r['step'])}")
            lines.append(f"  RPN: {r['rpn']}")
            lines.append(f"  Severity: {r.get('severity')}")
            lines.append(f"  Occurrence: {r.get('occurrence')}")
            lines.append(f"  Detection: {r.get('detection')}")
            if r.get("normalized_rpn") is not None:
                lines.append(f"  Normalized RPN: {r.get('normalized_rpn')}")
            if r.get("calibrated_risk") is not None:
                lines.append(f"  Calibrated Risk: {r.get('calibrated_risk')}")
            lines.append("")
        return "\n".join(lines)

    # ----------------------------------------------------------------------
    # 4. Utility Methods
    # ----------------------------------------------------------------------
    @staticmethod
    def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        """Safely cast numeric fields to int within 1–10 range."""
        try:
            num = int(value)
            if 1 <= num <= 10:
                return num
            return default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(
        value: Any,
        default: Optional[float] = None,
        minimum: float = 0.0,
        maximum: float = 1.0,
        precision: int = 4,
    ) -> Optional[float]:
        """Safely cast numeric fields to float within an expected range."""
        try:
            num = float(value)
        except (ValueError, TypeError):
            return default

        if minimum is not None and num < minimum:
            return default
        if maximum is not None and num > maximum:
            return default

        return round(num, precision)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Remove unsafe characters from step names."""
        if not isinstance(name, str):
            return "unknown"
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", name[:50])

    @staticmethod
    def _is_safe_path(file_path: str) -> bool:
        """Prevent directory traversal or unsafe file access."""
        if not isinstance(file_path, str):
            return False

        normalized = os.path.abspath(file_path)
        base_dir = os.path.abspath(os.getcwd())

        # Enforce writing only inside current working directory
        if not normalized.startswith(base_dir):
            logger.warning(f"[FMEA] Path traversal attempt: {file_path}")
            return False

        if len(os.path.basename(file_path)) > FMEAReport.MAX_FILENAME_LEN:
            logger.warning(f"[FMEA] File name too long: {file_path}")
            return False

        return True


# ----------------------------------------------------------------------
# Manual test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    context = {
        "step1_rpn": 120,
        "step1_severity": 8,
        "step1_occurrence": 5,
        "step1_detection": 3,
        "step1_normalized_rpn": 0.25,
        "step1_calibrated_risk": 0.65,
        "invalid_step$#_rpn": "bad",  # should be ignored
    }

    report = FMEAReport(context)
    print(report.to_text())

    print("\nJSON OUTPUT:")
    print(report.to_json())

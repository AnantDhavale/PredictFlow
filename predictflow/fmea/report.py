import json
from typing import Dict, Any, List

class FMEAReport:
    """
    FMEA Report Generator

    Purpose:
      - Summarize FMEA results after workflow execution
      - Produce a structured report with severity, occurrence, detection, and RPN per step
      - Export the report to JSON or text formats
    """

    def __init__(self, context: Dict[str, Any]):
        """
        Initialize the report generator with the workflow context.
        The context should include step-level RPN values, typically from the Executor.
        """
        self.context = context
        self.results = []

    def build_report(self) -> List[Dict[str, Any]]:
        """Create a list of step-level FMEA summaries."""
        for key, value in self.context.items():
            if key.endswith("_rpn"):
                step_name = key.replace("_rpn", "")
                severity = self.context.get(f"{step_name}_severity", None)
                occurrence = self.context.get(f"{step_name}_occurrence", None)
                detection = self.context.get(f"{step_name}_detection", None)

                self.results.append({
                    "step": step_name,
                    "rpn": value,
                    "severity": severity,
                    "occurrence": occurrence,
                    "detection": detection
                })

        return self.results

    def to_json(self, file_path: str = None) -> str:
        """Return the report as JSON, optionally saving to a file."""
        if not self.results:
            self.build_report()

        report_json = json.dumps(self.results, indent=2)
        if file_path:
            with open(file_path, "w") as f:
                f.write(report_json)
        return report_json

    def to_text(self) -> str:
        """Return a plain-text formatted report."""
        if not self.results:
            self.build_report()

        lines = ["FMEA Step Report", "----------------"]
        for r in self.results:
            lines.append(f"Step: {r['step']}")
            lines.append(f"  RPN: {r['rpn']}")
            lines.append(f"  Severity: {r.get('severity')}")
            lines.append(f"  Occurrence: {r.get('occurrence')}")
            lines.append(f"  Detection: {r.get('detection')}")
            lines.append("")

        return "\n".join(lines)

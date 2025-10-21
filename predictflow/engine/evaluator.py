import statistics
from typing import Dict, Any, List

class Evaluator:
    """
    PredictFlow Evaluator Module

    Purpose:
      - Analyze workflow execution results
      - Compute overall health score and performance metrics
      - Identify critical path and anomalies
      - Quantify process reliability using FMEA + NLP confidence
      - Provide structured insights for dashboards or APIs
    """

    def __init__(self, context: Dict[str, Any]):
        """
        Initialize evaluator with workflow execution context.
        The context usually comes from Executor after a workflow run.
        """
        self.context = context
        self.results = {
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
        """Compute key metrics from the workflow context."""
        confidence_scores = self._extract_confidence_scores()
        rpn_values = self._extract_rpn_values()

        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        avg_risk = statistics.mean(rpn_values) if rpn_values else 0

        # Compute overall health (higher is better)
        health_score = self._compute_health(avg_confidence, avg_risk)

        # Identify top 3 risky steps (critical path)
        critical_path = self._identify_critical_path(rpn_values)

        # Check for any alert conditions
        alerts = self._detect_alerts(avg_confidence, avg_risk)

        self.results.update({
            "workflow_health": round(health_score, 2),
            "average_confidence": round(avg_confidence, 2),
            "average_risk": round(avg_risk, 2),
            "critical_path": critical_path,
            "alerts": alerts
        })

        self._print_results()
        return self.results

    # ----------------------------
    # 2. Metric Computations
    # ----------------------------
    def _extract_confidence_scores(self) -> List[float]:
        """Extract all confidence scores from context keys."""
        return [
            float(v)
            for k, v in self.context.items()
            if k.endswith("_confidence") and isinstance(v, (float, int))
        ]

    def _extract_rpn_values(self) -> List[float]:
        """Extract all FMEA RPN values from context keys."""
        return [
            float(v)
            for k, v in self.context.items()
            if k.endswith("_rpn") and isinstance(v, (float, int))
        ]

    def _compute_health(self, confidence: float, risk: float) -> float:
        """
        Compute a composite 'workflow health' metric.
        Formula (tunable):
            health = confidence * (1 - normalized_risk)
        Normalizes risk to 0–1 scale.
        """
        norm_risk = min(risk / 300, 1.0)  # since max RPN = 10*10*10 = 1000, normalized to ~300 scale
        return max(0, (confidence / 100) * (1 - norm_risk) * 100)

    def _identify_critical_path(self, rpn_values: List[float]) -> List[str]:
        """Find top 3 risky steps based on highest RPN values."""
        risky_steps = sorted(
            [(k.replace("_rpn", ""), v)
             for k, v in self.context.items()
             if k.endswith("_rpn")],
            key=lambda x: x[1],
            reverse=True
        )
        return [step for step, _ in risky_steps[:3]]

    def _detect_alerts(self, avg_confidence: float, avg_risk: float) -> List[str]:
        """Generate warnings based on process health conditions."""
        alerts = []
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
    def _print_results(self):
        """Print a clean summary of evaluation results."""
        print("\nPredictFlow Evaluation Summary")
        print("---------------------------------")
        print(f"Workflow Health Score : {self.results['workflow_health']}")
        print(f"Average Confidence     : {self.results['average_confidence']}")
        print(f"Average Risk (RPN)     : {self.results['average_risk']}")
        print(f"Critical Path (Top 3)  : {' -> '.join(self.results['critical_path'])}")
        for alert in self.results["alerts"]:
            print(alert)
        print("---------------------------------\n")

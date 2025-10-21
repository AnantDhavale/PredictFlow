from typing import Dict, Any
from predictflow.fmea.analyzer import compute_rpn

class FMEAHooks:
    """
    FMEA Hook Integrations

    Purpose:
      - Integrate FMEA logic into the workflow execution cycle
      - Provide pre- and post-step analysis hooks for dynamic risk scoring
      - Automatically calculate RPN values if FMEA attributes are defined in a workflow step
    """

    @staticmethod
    def before_step(context: Dict[str, Any], step: Dict[str, Any]):
        """
        Hook executed before each workflow step.
        Can be used for setup or initial validation.
        """
        step_id = step.get("id", "unknown")
        if all(k in step for k in ("severity", "occurrence", "detection")):
            print(f"Preparing FMEA analysis for step '{step_id}'")

    @staticmethod
    def after_step(context: Dict[str, Any], step: Dict[str, Any]):
        """
        Hook executed after each workflow step.
        Automatically computes and stores the step's RPN.
        """
        try:
            if all(k in step for k in ("severity", "occurrence", "detection")):
                rpn = compute_rpn(step)
                step_id = step.get("id")
                context[f"{step_id}_rpn"] = rpn
                print(f"Computed FMEA RPN for '{step_id}': {rpn}")
        except Exception as e:
            print(f"Error computing FMEA for step {step.get('id')}: {e}")

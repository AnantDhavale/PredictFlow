import importlib
import time
import traceback
from typing import Dict, Any


class Executor:
    """
    PredictFlow Workflow Executor
    --------------------------------
    - Runs workflow steps sequentially
    - Automatically computes Confidence + FMEA + Embedding vectors
    - Builds Critical Path (based on risk/confidence)
    """

    def __init__(self, workflow: Dict[str, Any]):
        self.workflow = workflow
        self.context = {}
        self.metrics = {}  # stores RPN, confidence, embedding per step
        self.hooks = {"before_step": [], "after_step": []}

    # -------------------------
    # Main Execution Loop
    # -------------------------
    def run(self):
        print(f"🔷 Running workflow: {self.workflow.get('name', 'Unnamed Workflow')}")

        for step in self.workflow.get("steps", []):
            step_id = step.get("id")
            action_name = step.get("action")
            print(f"\n➡️  Executing step: {step_id} ({action_name})")

            # Hooks before
            self._run_hooks("before_step", step)

            try:
                result = self._execute_action(action_name, step)
                if isinstance(result, dict):
                    self.context.update(result)

                # --- Predictive analytics ---
                rpn = self._compute_fmea(step)
                conf = self._compute_confidence(step)
                emb = self._compute_embedding(step)

                self.metrics[step_id] = {
                    "rpn": rpn,
                    "confidence": conf,
                    "embedding": emb,
                }

            except Exception as e:
                print(f"💥 Error in step {step_id}: {e}")
                traceback.print_exc()
                self.context["last_error"] = str(e)
                break

            self._run_hooks("after_step", step)
            time.sleep(0.3)

        print("\n✅ Workflow completed.")
        self._show_summary()
        self._compute_critical_path()

    # -------------------------
    # Action Execution
    # -------------------------
    def _execute_action(self, action_name, step):
        """Load and execute an action module from predictflow.actions"""
        try:
            module_name = f"predictflow.actions.{action_name}"
            action_module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            print(f"⚠️  Action not found: {module_name}")
            return None

        if not hasattr(action_module, "run"):
            print(f"⚠️  No run() function in {module_name}")
            return None

        print(f"⚙️  Running action: {action_name}")
        return action_module.run(self.context, step)

    # -------------------------
    # Predictive Scoring
    # -------------------------
    def _compute_fmea(self, step):
        """Compute RPN risk using FMEA module"""
        try:
            from predictflow.fmea.analyzer import compute_rpn
            rpn = compute_rpn(step)
            print(f"🧮 FMEA Risk (RPN): {rpn}")
            return rpn
        except Exception as e:
            print(f"⚠️ FMEA error: {e}")
            return 0

    def _compute_confidence(self, step):
        """Compute NLP-based confidence score"""
        try:
            from predictflow.confidence.scorer import compute_confidence
            conf = compute_confidence(step)
            print(f"🤖 Confidence Score: {conf}")
            return conf
        except Exception as e:
            print(f"⚠️ Confidence error: {e}")
            return 0.5

    def _compute_embedding(self, step):
        """Compute a vector embedding for the step description (for semantic clustering)"""
        try:
            from predictflow.confidence.embedding import get_vector
            desc = step.get("description", "")
            if not desc:
                return None
            vector = get_vector(desc)
            print(f"🧠 Embedding computed for '{step.get('id')}' ({len(vector)} dims)")
            return vector
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
            return None

    # -------------------------
    # Hooks
    # -------------------------
    def add_hook(self, when: str, func):
        if when not in self.hooks:
            raise ValueError("Hook must be 'before_step' or 'after_step'")
        self.hooks[when].append(func)

    def _run_hooks(self, when: str, step):
        for hook in self.hooks.get(when, []):
            try:
                hook(self.context, step)
            except Exception as e:
                print(f"⚠️ Hook '{when}' failed: {e}")

    # -------------------------
    # Reporting
    # -------------------------
    def _show_summary(self):
        print("\n📊 Workflow Metrics Summary:")
        for step, data in self.metrics.items():
            rpn = data.get("rpn", "-")
            conf = data.get("confidence", "-")
            print(f"  • {step}: RPN={rpn}, Confidence={conf}")

    def _compute_critical_path(self):
        """Find steps with highest RPN or lowest confidence."""
        if not self.metrics:
            print("⚠️ No metrics available for critical path analysis.")
            return

        sorted_steps = sorted(
            self.metrics.items(),
            key=lambda s: (s[1].get("rpn", 0), -s[1].get("confidence", 1)),
            reverse=True,
        )
        critical = [s[0] for s in sorted_steps[:3]]
        print(f"\n Critical Path (highest risk): {' → '.join(critical)}")

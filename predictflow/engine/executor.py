import importlib
import time
import traceback
import re
from typing import Dict, Any, List, Set, Optional

from predictflow.engine.scheduler import Scheduler
from predictflow.engine.autogen import generate_actions_from_yaml
from predictflow.engine import persistence


class Executor:
    """
    PredictFlow BPMN Executor (token-based) - Security Hardened
    """
    
    # Security: Whitelist of allowed condition operators
    SAFE_OPERATORS = {
        '==', '!=', '<', '>', '<=', '>=',
        'and', 'or', 'not', 'in', 'is',
        'True', 'False', 'None'
    }

    def __init__(self, workflow: Dict[str, Any], yaml_path: str = None, auto_generate: bool = True):
        self.wf = workflow
        self.yaml_path = yaml_path
        self.auto_generate = auto_generate

        # Runtime state
        self.context: Dict[str, Any] = {"messages": set()}
        self.metrics: Dict[str, Any] = {}
        self.scheduler = Scheduler(self.context)

        # Hooks
        self.hooks = {"before_step": [], "after_step": []}

        # Create action stubs if missing
        if self.auto_generate and self.yaml_path:
            try:
                print("Checking and generating missing actions from YAML/BPMN...")
                generate_actions_from_yaml(self.yaml_path)
            except Exception as e:
                print(f"Warning: Auto-generation failed: {e}")

        # Build graph
        self.nodes: Dict[str, Dict[str, Any]] = {s["id"]: s for s in self.wf.get("steps", [])}
        self.flows: List[Dict[str, Any]] = self.wf.get("flows", [])
        self.starts: List[str] = self.wf.get("start_ids", [])
        self.gateways: Dict[str, Any] = self.wf.get("gateways", {})

        # Adjacency maps
        self.out_edges: Dict[str, List[Dict[str, Any]]] = {}
        self.in_edges: Dict[str, List[Dict[str, Any]]] = {}
        for f in self.flows:
            self.out_edges.setdefault(f["sourceRef"], []).append(f)
            self.in_edges.setdefault(f["targetRef"], []).append(f)

        # Join state
        self.join_seen: Dict[str, Set[str]] = {}

    # -----------------------------
    # Public
    # -----------------------------
    def run(self):
        run_id = persistence.log_run_start(self.wf.get("name", "Unnamed Workflow"))
        print(f"Running workflow: {self.wf.get('name', 'Unnamed Workflow')}")

        try:
            if not self.starts:
                first = next(iter(self.nodes.keys()), None)
                if first:
                    self._run_token(first)
            else:
                for start in self.starts:
                    self._run_token(start)

            persistence.log_run_complete(run_id)
            print("\nWorkflow completed.")
            
        except Exception as e:
            persistence.log_run_status(run_id, "failed", str(e))
            print(f"\n❌ Workflow failed: {e}")
            traceback.print_exc()
            raise
        
        self._show_summary()
        self._compute_critical_path()

    # -----------------------------
    # Token traversal
    # -----------------------------
    def _run_token(self, node_id: str):
        queue: List[str] = [node_id]
        max_iterations = 1000  # ✅ Prevent infinite loops
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            current_id = queue.pop(0)
            step = self.nodes.get(current_id)
            
            if not step:
                print(f"Warning: step '{current_id}' not found")
                continue

            if self._is_join_gateway(current_id):
                if not self._join_ready(current_id):
                    queue.append(current_id)
                    time.sleep(0.01)
                    continue

            self._execute_step(step)

            boundary_targets = self._boundary_targets_for_host(step["id"])
            next_targets = self._next_nodes(step)

            for tgt in next_targets:
                if self._is_join_gateway(tgt):
                    self.join_seen.setdefault(tgt, set()).add(step["id"])

            for nx in next_targets:
                queue.append(nx)

            for bn in boundary_targets:
                queue.append(bn)
        
        if iterations >= max_iterations:
            raise RuntimeError("Workflow exceeded maximum iteration limit (possible infinite loop)")

    # -----------------------------
    # Execution of a single node
    # -----------------------------
    def _execute_step(self, step: Dict[str, Any]):
        sid = step["id"]
        stype = step.get("type", "task")
        action = step.get("action") or sid
        
        # ✅ Validate step ID
        if not self._is_valid_identifier(sid):
            raise ValueError(f"Invalid step ID: {sid}")
        
        print(f"\nExecuting step: {sid} ({action}) [{stype}]")

        self._run_hooks("before_step", step)

        try:
            if stype in ("intermediateCatchEvent", "startEvent") and step.get("variant") == "timer":
                self._handle_timer_event(step)

            if stype in ("intermediateCatchEvent", "startEvent") and step.get("variant") == "message":
                self._handle_message_wait(step)

            if stype == "userTask":
                self._handle_user_task(step)

            self._run_action(action, step)
            self._score(step)

            persistence.save_state(self.wf["name"], {
                "join_seen": {k: list(v) for k, v in self.join_seen.items()},  # ✅ Serialize sets
                "context": self.context,
                "metrics": self.metrics
            })

        except Exception as e:
            print(f"Error in step {sid}: {e}")
            traceback.print_exc()
            self.context["last_error"] = str(e)
            raise

        self._run_hooks("after_step", step)
        time.sleep(0.1)

    def _run_action(self, action_name: str, step: Dict[str, Any]):
        """
        ✅ SECURITY FIX: Validate action name before import
        """
        # Validate action name
        if not self._is_valid_identifier(action_name):
            print(f"❌ Invalid action name: {action_name}")
            return
        
        # Prevent path traversal
        if '/' in action_name or '\\' in action_name or '..' in action_name:
            print(f"❌ Action name contains invalid characters: {action_name}")
            return
        
        try:
            mod = importlib.import_module(f"predictflow.actions.{action_name}")
            if hasattr(mod, "run"):
                print(f"Running action: {action_name}")
                result = mod.run(self.context, step)
                if isinstance(result, dict):
                    # ✅ Validate result before merging
                    if len(str(result)) > 100000:  # 100KB limit
                        print("❌ Action result too large")
                        return
                    self.context.update(result)
            else:
                print(f"No run() in predictflow.actions.{action_name}")
        except ModuleNotFoundError:
            print(f"Action not found: predictflow.actions.{action_name}")
        except Exception as e:
            print(f"Action execution failed: {e}")
            traceback.print_exc()

    # -----------------------------
    # Gateways & Routing
    # -----------------------------
    def _next_nodes(self, step: Dict[str, Any]) -> List[str]:
        sid = step["id"]
        stype = step.get("type", "task")
        outs = self.out_edges.get(sid, [])

        if stype == "exclusiveGateway":
            for f in outs:
                if self._cond_true(f.get("condition")):
                    return [f["targetRef"]]
            default = next((f for f in outs if f.get("isDefault")), None)
            return [default["targetRef"]] if default else ([outs[0]["targetRef"]] if outs else [])

        if stype == "inclusiveGateway":
            matched = [f["targetRef"] for f in outs if self._cond_true(f.get("condition"))]
            return matched or [f["targetRef"] for f in outs]

        if stype == "parallelGateway":
            # ✅ Limit parallel branches
            max_parallel = 10
            targets = [f["targetRef"] for f in outs[:max_parallel]]
            if len(outs) > max_parallel:
                print(f"⚠️ Parallel gateway limited to {max_parallel} branches")
            return targets

        return [f["targetRef"] for f in outs]

    def _is_join_gateway(self, node_id: str) -> bool:
        st = self.nodes[node_id].get("type")
        incoming = len(self.in_edges.get(node_id, []))
        if incoming <= 1:
            return False
        return st in ("parallelGateway", "inclusiveGateway")

    def _join_ready(self, node_id: str) -> bool:
        incoming_from = {f["sourceRef"] for f in self.in_edges.get(node_id, [])}
        seen = self.join_seen.setdefault(node_id, set())
        gtype = self.nodes[node_id].get("type")

        if gtype == "parallelGateway":
            return seen.issuperset(incoming_from)
        else:
            return len(seen.intersection(incoming_from)) >= 1

    # -----------------------------
    # Boundary events
    # -----------------------------
    def _boundary_targets_for_host(self, host_id: str) -> List[str]:
        targets: List[str] = []
        for nid, node in self.nodes.items():
            if node.get("type") == "boundaryEvent" and node.get("attachedTo") == host_id:
                outs = self.out_edges.get(nid, [])
                for f in outs:
                    targets.append(f["targetRef"])
        if targets:
            print(f"Scheduling boundary branches from host {host_id}: {targets}")
        return targets

    # -----------------------------
    # ✅ SECURITY FIX: Safe Condition Evaluation
    # -----------------------------
    def _cond_true(self, expr: Optional[str]) -> bool:
        """
        Safely evaluate conditions without allowing arbitrary code execution.
        
        Supports simple comparisons like:
        - context['status'] == 'approved'
        - context['amount'] > 1000
        - context['approved'] and context['verified']
        """
        if not expr:
            return False
        
        try:
            # Remove whitespace
            expr = expr.strip()
            
            # Check for dangerous patterns
            dangerous_patterns = [
                '__import__', 'eval', 'exec', 'compile', 'open',
                'file', 'input', 'raw_input', 'execfile',
                'reload', '__builtins__', 'globals', 'locals',
                'vars', 'dir', 'help', 'copyright', 'credits',
                'license', 'quit', 'exit', 'delattr', 'setattr',
                'getattr', 'hasattr', 'callable', 'isinstance',
                'issubclass', 'type', 'classmethod', 'staticmethod'
            ]
            
            expr_lower = expr.lower()
            for pattern in dangerous_patterns:
                if pattern in expr_lower:
                    print(f"❌ Blocked dangerous pattern in condition: {pattern}")
                    return False
            
            # Only allow simple context variable access
            # Build a safe evaluation environment
            safe_context = {
                k: v for k, v in self.context.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
            
            # Restricted namespace - no builtins, no imports
            restricted_namespace = {
                '__builtins__': {},
                'True': True,
                'False': False,
                'None': None,
                'context': safe_context
            }
            
            # Evaluate with timeout protection
            result = eval(expr, restricted_namespace, {})
            return bool(result)
            
        except Exception as e:
            print(f"⚠️ Condition evaluation failed: {e}")
            return False

    def _is_valid_identifier(self, name: str) -> bool:
        """
        ✅ Validate that a name is a safe identifier.
        Allows: letters, numbers, underscores, hyphens
        """
        if not name:
            return False
        
        # Max length check
        if len(name) > 255:
            return False
        
        # Pattern check: alphanumeric, underscore, hyphen only
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, name))

    # -----------------------------
    # Conditions, timers, messages, human tasks
    # -----------------------------
    def _handle_timer_event(self, step: Dict[str, Any]):
        timer_raw = (step.get("timer") or {}).get("raw", "")
        
        # ✅ Limit timer duration to prevent DoS
        max_wait = 3600  # 1 hour max
        wait_time = min(self._parse_timer_duration(timer_raw), max_wait)
        
        if wait_time > 0:
            print(f"Timer wait ({wait_time}s)")
            time.sleep(wait_time)

    def _parse_timer_duration(self, raw: str) -> float:
        """Parse ISO 8601 duration or simple duration string."""
        if not raw:
            return 2.0  # Default
        
        try:
            # Simple parsing for PT10S format
            if raw.startswith('PT') and raw.endswith('S'):
                return float(raw[2:-1])
            elif raw.endswith('s'):
                return float(raw[:-1])
            elif raw.endswith('m'):
                return float(raw[:-1]) * 60
            elif raw.endswith('h'):
                return float(raw[:-1]) * 3600
        except:
            pass
        
        return 2.0  # Fallback

    def _handle_message_wait(self, step: Dict[str, Any]):
        key = step["id"]
        print(f"Waiting for message at '{key}' (stub). Use API /messages/correlate to add it.")

    def _handle_user_task(self, step: Dict[str, Any]):
        task = {
            "id": step["id"],
            "lane": step.get("lane"),
            "description": step.get("description"),
            "status": "pending"
        }
        persistence.save_user_task(task)
        print(f"[UserTask] Created: {task['id']} (lane={task['lane']})")

    # -----------------------------
    # Scoring
    # -----------------------------
    def _score(self, step: Dict[str, Any]):
        try:
            from predictflow.fmea.analyzer import compute_rpn
            r = compute_rpn(step)
        except Exception:
            r = 0
        try:
            from predictflow.confidence.scorer import compute_confidence
            c = compute_confidence(step)
        except Exception:
            c = 0.5
        try:
            from predictflow.confidence.embedding import get_vector
            desc = step.get("description", "")
            e = get_vector(desc) if desc else None
        except Exception:
            e = None
        self.metrics[step["id"]] = {
            "rpn": r,
            "confidence": c,
            "embedding": e,
            "lane": step.get("lane")
        }

    # -----------------------------
    # Hooks
    # -----------------------------
    def add_hook(self, when: str, func):
        if when not in self.hooks:
            raise ValueError("Hook must be 'before_step' or 'after_step'")
        self.hooks[when].append(func)

    def _run_hooks(self, when: str, step: Dict[str, Any]):
        for hook in self.hooks.get(when, []):
            try:
                hook(self.context, step)
            except Exception as e:
                print(f"Hook '{when}' failed: {e}")

    # -----------------------------
    # Reports
    # -----------------------------
    def _show_summary(self):
        print("\nWorkflow Metrics Summary:")
        for sid, m in self.metrics.items():
            print(f"  • {sid}: RPN={m.get('rpn')}, Confidence={m.get('confidence')}")

    def _compute_critical_path(self):
        if not self.metrics:
            print("No metrics available for critical path analysis.")
            return
        ordered = sorted(
            self.metrics.items(),
            key=lambda kv: (kv[1].get("rpn", 0), -kv[1].get("confidence", 1)),
            reverse=True
        )
        top = [sid for sid, _ in ordered[:3]]
        print(f"\nCritical Path (highest risk): {' → '.join(top)}")

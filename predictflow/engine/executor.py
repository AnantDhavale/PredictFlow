import importlib
import time
import traceback
from typing import Dict, Any, List, Set, Optional

from predictflow.engine.scheduler import Scheduler
from predictflow.engine.autogen import generate_actions_from_yaml
from predictflow.engine import persistence


class Executor:
    """
    PredictFlow BPMN Executor (token-based)
    --------------------------------------
    • Traverses BPMN sequenceFlow graph from Start → End using a token queue
    • Gateways:
        - exclusiveGateway: first matching condition, else default
        - inclusiveGateway: all matching conditions (join waits for any; can be tightened)
        - parallelGateway: fork all; join waits for all incoming
    • Events:
        - boundary events (timer/message) attached to a task → schedule their side branches
        - intermediateCatch timer → waits (simulated)
        - message catch → placeholder wait (API can correlate)
    • Lanes/Pools captured in step metadata
    • Persistence: run logs + state + user tasks
    """

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

        # Join state: for gateway joins, record which predecessors have arrived
        self.join_seen: Dict[str, Set[str]] = {}   # gwId -> set(sourceNodeIds that arrived)

    # -----------------------------
    # Public
    # -----------------------------
    def run(self):
        # Run start logging
        run_id = persistence.log_run_start(self.wf.get("name", "Unnamed Workflow"))
        print(f"Running workflow: {self.wf.get('name', 'Unnamed Workflow')}")

        # Traverse from each start node
        if not self.starts:
            # Fallback: if parser didn't provide starts, run first step found
            first = next(iter(self.nodes.keys()), None)
            if first:
                self._run_token(first)
        else:
            for start in self.starts:
                self._run_token(start)

        # Mark completion
        persistence.log_run_complete(run_id)

        print("\nWorkflow completed.")
        self._show_summary()
        self._compute_critical_path()

    # -----------------------------
    # Token traversal
    # -----------------------------
    def _run_token(self, node_id: str):
        queue: List[str] = [node_id]

        while queue:
            current_id = queue.pop(0)
            step = self.nodes.get(current_id)
            if not step:
                print(f"Warning: step '{current_id}' not found")
                continue

            # JOIN logic: if current node is a join gateway, ensure ready
            if self._is_join_gateway(current_id):
                if not self._join_ready(current_id):
                    # Not ready yet; requeue to check later
                    queue.append(current_id)
                    # avoid tight loop
                    time.sleep(0.01)
                    continue

            # Execute node
            self._execute_step(step)

            # Collect boundary branches attached to this activity
            boundary_targets = self._boundary_targets_for_host(step["id"])

            # Determine primary outgoing targets based on gateway/task logic
            next_targets = self._next_nodes(step)

            # Mark arrivals to downstream joins (for join readiness)
            for tgt in next_targets:
                if self._is_join_gateway(tgt):
                    self.join_seen.setdefault(tgt, set()).add(step["id"])

            # Enqueue main flow
            for nx in next_targets:
                queue.append(nx)

            # Enqueue boundary branches (side paths)
            for bn in boundary_targets:
                queue.append(bn)

    # -----------------------------
    # Execution of a single node
    # -----------------------------
    def _execute_step(self, step: Dict[str, Any]):
        sid = step["id"]
        stype = step.get("type", "task")
        action = step.get("action") or sid
        print(f"\nExecuting step: {sid} ({action}) [{stype}]")

        # Hooks before
        self._run_hooks("before_step", step)

        # Event handling (basic)
        try:
            if stype in ("intermediateCatchEvent", "startEvent") and step.get("variant") == "timer":
                self._handle_timer_event(step)

            if stype in ("intermediateCatchEvent", "startEvent") and step.get("variant") == "message":
                self._handle_message_wait(step)

            # User tasks go to a queue (persisted)
            if stype == "userTask":
                self._handle_user_task(step)

            # Service/Script/Generic tasks → run action module
            self._run_action(action, step)

            # scoring
            self._score(step)

            # Persist engine state after every step
            persistence.save_state(self.wf["name"], {
                "join_seen": self.join_seen,
                "context": self.context,
                "metrics": self.metrics
            })

        except Exception as e:
            print(f"Error in step {sid}: {e}")
            traceback.print_exc()
            self.context["last_error"] = str(e)

        # Hooks after
        self._run_hooks("after_step", step)

        # Small pacing
        time.sleep(0.1)

    def _run_action(self, action_name: str, step: Dict[str, Any]):
        try:
            mod = importlib.import_module(f"predictflow.actions.{action_name}")
            if hasattr(mod, "run"):
                print(f"Running action: {action_name}")
                result = mod.run(self.context, step)
                if isinstance(result, dict):
                    self.context.update(result)
            else:
                print(f"No run() in predictflow.actions.{action_name}")
        except ModuleNotFoundError:
            print(f"Action not found: predictflow.actions.{action_name}")

    # -----------------------------
    # Gateways & Routing
    # -----------------------------
    def _next_nodes(self, step: Dict[str, Any]) -> List[str]:
        sid = step["id"]
        stype = step.get("type", "task")
        outs = self.out_edges.get(sid, [])

        if stype == "exclusiveGateway":
            # First matching condition, else default (by flow isDefault)
            for f in outs:
                if self._cond_true(f.get("condition")):
                    return [f["targetRef"]]
            default = next((f for f in outs if f.get("isDefault")), None)
            return [default["targetRef"]] if default else ([outs[0]["targetRef"]] if outs else [])

        if stype == "inclusiveGateway":
            # All matching conditions; if none explicitly match, take all
            matched = [f["targetRef"] for f in outs if self._cond_true(f.get("condition"))]
            return matched or [f["targetRef"] for f in outs]

        if stype == "parallelGateway":
            # Fork all
            return [f["targetRef"] for f in outs]

        # Default for tasks/events: all outs (usually 1)
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
            # all predecessors must have arrived
            return seen.issuperset(incoming_from)
        else:
            # inclusive: at least one predecessor (can be tightened with path-analysis)
            return len(seen.intersection(incoming_from)) >= 1

    # -----------------------------
    # Boundary events
    # -----------------------------
    def _boundary_targets_for_host(self, host_id: str) -> List[str]:
        """
        Return the target node IDs of boundary events attached to host task.
        """
        targets: List[str] = []
        # Find boundary event nodes attached to this host
        for nid, node in self.nodes.items():
            if node.get("type") == "boundaryEvent" and node.get("attachedTo") == host_id:
                outs = self.out_edges.get(nid, [])
                for f in outs:
                    targets.append(f["targetRef"])
        if targets:
            print(f"Scheduling boundary branches from host {host_id}: {targets}")
        return targets

    # -----------------------------
    # Conditions, timers, messages, human tasks
    # -----------------------------
    def _cond_true(self, expr: Optional[str]) -> bool:
        if not expr:
            return False
        try:
            # Evaluate condition in context (sandboxed: no builtins)
            return bool(eval(expr, {"__builtins__": {}}, self.context))
        except Exception:
            return False

    def _handle_timer_event(self, step: Dict[str, Any]):
        timer_raw = (step.get("timer") or {}).get("raw", "")
        print(f"Timer wait (raw='{timer_raw}') → simulating 2s sleep")
        time.sleep(2)

    def _handle_message_wait(self, step: Dict[str, Any]):
        key = step["id"]
        print(f"Waiting for message at '{key}' (stub). Use API /messages/correlate to add it.")
        # Real systems would park token; here we just continue.

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
        self.metrics[step["id"]] = {"rpn": r, "confidence": c, "embedding": e, "lane": step.get("lane")}

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

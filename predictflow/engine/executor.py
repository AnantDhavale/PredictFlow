"""
PredictFlow BPMN Executor - Enhanced with Intelligent Routing
------------------------------------------------------------
Integrates context-aware resource assignment with workflow execution.
"""

import importlib
import time
import traceback
import re
from typing import Dict, Any, List, Set, Optional

from predictflow.engine.scheduler import Scheduler
from predictflow.engine.autogen import generate_actions_from_yaml
from predictflow.engine import persistence

# NEW: Import intelligent routing components
try:
    from predictflow.routing.resource_router import ResourceRouter, ResourceProfile, RoutingStrategy
    from predictflow.context.context_engine import ContextEngine
    ROUTING_AVAILABLE = True
except ImportError:
    ROUTING_AVAILABLE = False
    print("⚠️  Intelligent routing not available - install routing module")


class Executor:
    """
    PredictFlow BPMN Executor (token-based) with Intelligent Routing
    
    Features:
      - Token-based BPMN execution (existing)
      - Context-aware resource assignment (NEW)
      - Risk-based routing and escalation (NEW)
    """
    
    SAFE_OPERATORS = {
        '==', '!=', '<', '>', '<=', '>=',
        'and', 'or', 'not', 'in', 'is',
        'True', 'False', 'None'
    }

    def __init__(
        self, 
        workflow: Dict[str, Any], 
        yaml_path: str = None, 
        auto_generate: bool = True,
        enable_intelligent_routing: bool = True,  # NEW
        routing_strategy: str = "hybrid"  # NEW
    ):
        self.wf = workflow
        self.yaml_path = yaml_path
        self.auto_generate = auto_generate

        # Runtime state
        self.context: Dict[str, Any] = {
            "messages": set(),
            "workflow_state": {}  # NEW: for context engine
        }
        self.metrics: Dict[str, Any] = {}
        self.scheduler = Scheduler(self.context)

        # Hooks
        self.hooks = {"before_step": [], "after_step": []}

        # NEW: Intelligent Routing Components
        self.enable_intelligent_routing = enable_intelligent_routing and ROUTING_AVAILABLE
        if self.enable_intelligent_routing:
            self._init_routing(routing_strategy)
        else:
            self.resource_router = None
            self.context_engine = None

        # Auto-generate actions
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
    # NEW: Routing Initialization
    # -----------------------------
    def _init_routing(self, strategy: str):
        """Initialize intelligent routing components."""
        try:
            strategy_map = {
                "hybrid": RoutingStrategy.HYBRID,
                "risk": RoutingStrategy.RISK_BASED,
                "skill": RoutingStrategy.SKILL_MATCH,
                "load": RoutingStrategy.LOAD_BALANCE
            }
            
            self.resource_router = ResourceRouter(
                strategy=strategy_map.get(strategy.lower(), RoutingStrategy.HYBRID)
            )
            self.context_engine = ContextEngine()
            
            print(f"✓ Intelligent routing enabled (strategy: {strategy})")
        except Exception as e:
            print(f"⚠️  Failed to initialize routing: {e}")
            self.resource_router = None
            self.context_engine = None

    # NEW: Resource Management
    def register_resource(
        self, 
        resource_id: str, 
        seniority: float = 0.5,
        authority: float = 0.5,
        skills: List[str] = None,
        workload: float = 0.5
    ):
        """
        Register an assignee/resource for intelligent routing.
        
        Args:
            resource_id: Unique identifier (e.g., "alice", "bob")
            seniority: 0-1, experience level
            authority: 0-1, decision-making power
            skills: List of skill tags
            workload: 0-1, current load
        """
        if not self.resource_router:
            print("⚠️  Routing not enabled, resource registration ignored")
            return
        
        profile = ResourceProfile(
            resource_id=resource_id,
            seniority_level=seniority,
            authority_level=authority,
            skills=skills or [],
            current_workload=workload
        )
        
        self.resource_router.register_resource(resource_id, profile)
        print(f"✓ Registered resource: {resource_id}")

    def set_workflow_state(self, **kwargs):
        """
        Set workflow-level context for intelligent routing.
        
        Example:
            executor.set_workflow_state(
                customer_tier='enterprise',
                deadline='2024-12-01T10:00:00Z',
                system_health=0.95
            )
        """
        self.context['workflow_state'].update(kwargs)

    # -----------------------------
    # Public Execution
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
        
        # NEW: Show routing analytics
        if self.resource_router:
            self._show_routing_summary()

    # -----------------------------
    # Token Traversal
    # -----------------------------
    def _run_token(self, node_id: str):
        queue: List[str] = [node_id]
        max_iterations = 1000
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
            raise RuntimeError("Workflow exceeded maximum iteration limit")

    # -----------------------------
    # Execution of Single Node (ENHANCED)
    # -----------------------------
    def _execute_step(self, step: Dict[str, Any]):
        sid = step["id"]
        stype = step.get("type", "task")
        action = step.get("action") or sid
        
        if not self._is_valid_identifier(sid):
            raise ValueError(f"Invalid step ID: {sid}")
        
        # NEW: Intelligent resource assignment
        if self.enable_intelligent_routing:
            step = self._assign_resource(step)
        
        assignee = step.get('assignee', 'unassigned')
        print(f"\nExecuting step: {sid} ({action}) [{stype}] → Assignee: {assignee}")

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

            # NEW: Update resource workload after execution
            if self.resource_router and assignee != 'unassigned':
                routing_data = step.get('routing_decision', {})
                current_load = routing_data.get('workload_score', 0.5)
                self.resource_router.update_workload(assignee, current_load + 0.1)

            persistence.save_state(self.wf["name"], {
                "join_seen": {k: list(v) for k, v in self.join_seen.items()},
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

    # -----------------------------
    # NEW: Intelligent Resource Assignment
    # -----------------------------
    def _assign_resource(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply intelligent routing to determine step assignee.
        
        Returns:
            Step dict enriched with assignee and routing metadata
        """
        # If step has explicit assignee (not 'auto'), respect it
        if step.get('assignee') and step.get('assignee') != 'auto':
            return step
        
        # Get available assignees
        available = list(self.resource_router.resource_pool.keys())
        if not available:
            print("⚠️  No resources registered, using default assignee")
            step['assignee'] = 'default'
            return step
        
        try:
            # Collect context
            context_snapshot = self.context_engine.collect_context(
                step, 
                self.context.get('workflow_state', {})
            )
            
            # Compute FMEA risk
            from predictflow.fmea.analyzer import compute_rpn
            fmea_result = compute_rpn(step)
            
            # Context-aware risk adjustment
            adjusted_risk = self.context_engine.adjust_rpn_with_context(
                fmea_result['rpn'],
                context_snapshot
            )
            
            # Intelligent routing decision
            routing_decision = self.resource_router.route_step(
                step,
                context_snapshot,
                adjusted_risk,
                available
            )
            
            # Enrich step
            step['assignee'] = routing_decision.assignee
            step['routing_decision'] = routing_decision.to_dict()
            step['context_snapshot'] = context_snapshot.to_dict()
            step['fmea_adjusted'] = adjusted_risk
            
            # Log routing decision
            if routing_decision.escalated:
                print(f"⚠️  ESCALATED: {routing_decision.explanation}")
            else:
                print(f"ℹ️  Routing: {routing_decision.explanation}")
        
        except Exception as e:
            print(f"⚠️  Routing failed: {e}, using default assignee")
            step['assignee'] = available[0] if available else 'default'
        
        return step

    # -----------------------------
    # Action Execution
    # -----------------------------
    def _run_action(self, action_name: str, step: Dict[str, Any]):
        """Execute action with security validation."""
        if not self._is_valid_identifier(action_name):
            print(f"❌ Invalid action name: {action_name}")
            return
        
        if '/' in action_name or '\\' in action_name or '..' in action_name:
            print(f"❌ Action name contains invalid characters: {action_name}")
            return
        
        try:
            mod = importlib.import_module(f"predictflow.actions.{action_name}")
            if hasattr(mod, "run"):
                print(f"Running action: {action_name}")
                result = mod.run(self.context, step)
                if isinstance(result, dict):
                    if len(str(result)) > 100000:
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
    # Gateways & Routing (Workflow Graph)
    # -----------------------------
    def _next_nodes(self, step: Dict[str, Any]) -> List[str]:
        """Determine next nodes in workflow graph."""
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
    # Boundary Events
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
    # Safe Condition Evaluation
    # -----------------------------
    def _cond_true(self, expr: Optional[str]) -> bool:
        """Safely evaluate conditions."""
        if not expr:
            return False
        
        try:
            expr = expr.strip()
            
            dangerous_patterns = [
                '__import__', 'eval', 'exec', 'compile', 'open',
                'file', 'input', 'raw_input', 'execfile',
                'reload', '__builtins__', 'globals', 'locals',
                'vars', 'dir', 'help', 'delattr', 'setattr',
                'getattr', 'hasattr'
            ]
            
            expr_lower = expr.lower()
            for pattern in dangerous_patterns:
                if pattern in expr_lower:
                    print(f"❌ Blocked dangerous pattern: {pattern}")
                    return False
            
            safe_context = {
                k: v for k, v in self.context.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
            
            restricted_namespace = {
                '__builtins__': {},
                'True': True,
                'False': False,
                'None': None,
                'context': safe_context
            }
            
            result = eval(expr, restricted_namespace, {})
            return bool(result)
            
        except Exception as e:
            print(f"⚠️ Condition evaluation failed: {e}")
            return False

    def _is_valid_identifier(self, name: str) -> bool:
        """Validate identifier safety."""
        if not name or len(name) > 255:
            return False
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, name))

    # -----------------------------
    # Event Handlers
    # -----------------------------
    def _handle_timer_event(self, step: Dict[str, Any]):
        timer_raw = (step.get("timer") or {}).get("raw", "")
        max_wait = 3600
        wait_time = min(self._parse_timer_duration(timer_raw), max_wait)
        if wait_time > 0:
            print(f"Timer wait ({wait_time}s)")
            time.sleep(wait_time)

    def _parse_timer_duration(self, raw: str) -> float:
        if not raw:
            return 2.0
        try:
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
        return 2.0

    def _handle_message_wait(self, step: Dict[str, Any]):
        key = step["id"]
        print(f"Waiting for message at '{key}' (stub)")

    def _handle_user_task(self, step: Dict[str, Any]):
        task = {
            "id": step["id"],
            "lane": step.get("lane"),
            "assignee": step.get("assignee", "unassigned"),  # NEW
            "description": step.get("description"),
            "status": "pending"
        }
        persistence.save_user_task(task)
        print(f"[UserTask] Created: {task['id']} (assignee={task['assignee']})")

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
        
        # NEW: Include routing metadata in metrics
        self.metrics[step["id"]] = {
            "rpn": r,
            "confidence": c,
            "embedding": e,
            "lane": step.get("lane"),
            "assignee": step.get("assignee"),
            "routing_confidence": step.get("routing_decision", {}).get("confidence", 0),
            "escalated": step.get("routing_decision", {}).get("escalated", False)
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
            assignee = m.get('assignee', 'N/A')
            print(f"  • {sid} [{assignee}]: RPN={m.get('rpn')}, Confidence={m.get('confidence')}")

    def _show_routing_summary(self):
        """NEW: Show routing analytics."""
        if not self.resource_router:
            return
        
        analytics = self.resource_router.get_routing_analytics()
        print("\nIntelligent Routing Summary:")
        print(f"  Total Routing Decisions: {analytics.get('total_routes', 0)}")
        print(f"  Escalations: {analytics.get('escalations', 0)}")
        print(f"  Avg Routing Confidence: {analytics.get('avg_confidence', 0):.2f}")
        
        strategies = analytics.get('strategy_distribution', {})
        if strategies:
            print("  Strategy Distribution:")
            for strategy, count in strategies.items():
                print(f"    - {strategy}: {count}")

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

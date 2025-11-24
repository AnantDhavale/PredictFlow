"""
PredictFlow Context Engine
--------------------------

Purpose:
  - Aggregate multi-dimensional context for workflow steps
  - Enable semantic understanding beyond static FMEA scores
  - Provide context-aware routing and risk adjustment

Context Dimensions:
  - Business: customer value, contract status, strategic alignment
  - Temporal: deadlines, time-of-day patterns, seasonal factors
  - Historical: past success/failure patterns, similar cases
  - Stakeholder: workload, expertise, availability
  - Environmental: system health, market conditions, compliance
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ContextSnapshot:
    """Structured context state for a workflow step."""
    
    # Business Context
    customer_value: float = 0.5  # 0-1: tier, revenue, strategic importance
    urgency_level: float = 0.5   # 0-1: deadline proximity, business impact
    compliance_risk: float = 0.0 # 0-1: regulatory sensitivity
    
    # Temporal Context
    time_pressure: float = 0.0   # 0-1: deadline urgency
    seasonal_factor: float = 0.5 # 0-1: seasonal patterns (e.g., end-of-quarter)
    time_of_day_risk: float = 0.5 # 0-1: after-hours risk increase
    
    # Historical Context
    success_rate: float = 0.8    # 0-1: historical success for similar steps
    avg_duration: float = 0.0    # minutes: typical completion time
    failure_patterns: List[str] = None  # Known failure modes
    
    # Stakeholder Context
    assignee_workload: float = 0.5  # 0-1: current load
    assignee_expertise: float = 0.7 # 0-1: skill match
    assignee_available: bool = True
    
    # Environmental Context
    system_health: float = 1.0   # 0-1: infrastructure status
    concurrent_load: float = 0.3 # 0-1: system congestion
    market_volatility: float = 0.5  # 0-1: external stability
    
    # Metadata
    timestamp: str = None
    step_id: str = None
    
    def __post_init__(self):
        if self.failure_patterns is None:
            self.failure_patterns = []
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def aggregate_score(self) -> float:
        """Compute overall context severity (0-1)."""
        weights = {
            'customer_value': 0.15,
            'urgency_level': 0.20,
            'compliance_risk': 0.15,
            'time_pressure': 0.10,
            'success_rate': -0.15,  # Inverse: higher success = lower risk
            'assignee_workload': 0.10,
            'system_health': -0.10,  # Inverse
            'market_volatility': 0.05
        }
        
        score = 0.5  # Baseline
        for attr, weight in weights.items():
            value = getattr(self, attr, 0.5)
            if isinstance(value, (int, float)):
                score += weight * value
        
        return max(0.0, min(1.0, score))


class ContextEngine:
    """
    PredictFlow Context Aggregation Engine
    
    Collects, normalizes, and synthesizes context from multiple sources
    to provide semantic understanding of workflow state.
    """
    
    def __init__(self):
        self.context_cache = {}  # step_id -> ContextSnapshot
        self.historical_data = {}  # step_type -> aggregated stats
    
    # ------------------------------------------------------------------
    # 1. Context Collection
    # ------------------------------------------------------------------
    
    def collect_context(
        self, 
        step: Dict[str, Any], 
        workflow_state: Dict[str, Any]
    ) -> ContextSnapshot:
        """
        Aggregate context from step metadata and workflow state.
        """
        step_id = str(step.get('id', 'unknown'))
        
        context = ContextSnapshot(
            step_id=step_id,
            customer_value=self._extract_customer_value(step, workflow_state),
            urgency_level=self._extract_urgency(step, workflow_state),
            compliance_risk=self._extract_compliance_risk(step),
            time_pressure=self._calculate_time_pressure(step, workflow_state),
            seasonal_factor=self._calculate_seasonal_factor(),
            time_of_day_risk=self._calculate_time_of_day_risk(),
            success_rate=self._get_historical_success_rate(step),
            assignee_workload=self._get_assignee_workload(step, workflow_state),
            assignee_expertise=self._get_assignee_expertise(step),
            assignee_available=self._check_assignee_availability(step),
            system_health=self._get_system_health(workflow_state),
            concurrent_load=self._get_concurrent_load(workflow_state)
        )
        
        self.context_cache[step_id] = context
        return context
    
    # ------------------------------------------------------------------
    # 2. Business Context Extractors
    # ------------------------------------------------------------------
    
    def _extract_customer_value(
        self, 
        step: Dict[str, Any], 
        workflow_state: Dict[str, Any]
    ) -> float:
        """Determine customer/stakeholder value tier."""
        # Check workflow metadata
        customer_tier = workflow_state.get('customer_tier', 'standard')
        revenue = workflow_state.get('customer_annual_revenue', 0)
        
        # Semantic understanding
        tier_map = {
            'enterprise': 0.95,
            'premium': 0.80,
            'standard': 0.50,
            'trial': 0.30
        }
        
        tier_score = tier_map.get(customer_tier.lower(), 0.5)
        
        # Revenue-based adjustment
        if revenue > 1_000_000:
            tier_score = max(tier_score, 0.90)
        elif revenue > 100_000:
            tier_score = max(tier_score, 0.70)
        
        return tier_score
    
    def _extract_urgency(
        self, 
        step: Dict[str, Any], 
        workflow_state: Dict[str, Any]
    ) -> float:
        """Extract urgency from step metadata or semantic analysis."""
        # Explicit urgency flag
        if step.get('priority') == 'critical':
            return 1.0
        if step.get('priority') == 'high':
            return 0.8
        
        # Semantic keyword detection
        description = str(step.get('description', '')).lower()
        urgency_keywords = ['urgent', 'critical', 'emergency', 'asap', 'immediately']
        
        if any(keyword in description for keyword in urgency_keywords):
            return 0.9
        
        # Workflow-level urgency
        if workflow_state.get('sla_breach_imminent'):
            return 0.95
        
        return 0.5
    
    def _extract_compliance_risk(self, step: Dict[str, Any]) -> float:
        """Identify regulatory or compliance sensitivity."""
        tags = step.get('tags', [])
        compliance_tags = ['gdpr', 'sox', 'hipaa', 'pci', 'regulatory', 'audit']
        
        if any(tag.lower() in compliance_tags for tag in tags):
            return 0.9
        
        description = str(step.get('description', '')).lower()
        if any(term in description for term in ['compliance', 'regulatory', 'audit']):
            return 0.8
        
        return 0.0
    
    # ------------------------------------------------------------------
    # 3. Temporal Context
    # ------------------------------------------------------------------
    
    def _calculate_time_pressure(
        self, 
        step: Dict[str, Any], 
        workflow_state: Dict[str, Any]
    ) -> float:
        """Calculate deadline proximity pressure."""
        deadline = workflow_state.get('deadline')
        if not deadline:
            return 0.0
        
        try:
            deadline_dt = datetime.fromisoformat(deadline)
            now = datetime.utcnow()
            time_remaining = (deadline_dt - now).total_seconds() / 3600  # hours
            
            if time_remaining < 0:
                return 1.0  # Past deadline
            elif time_remaining < 2:
                return 0.95
            elif time_remaining < 24:
                return 0.7
            elif time_remaining < 72:
                return 0.4
            else:
                return 0.1
        except Exception:
            return 0.0
    
    def _calculate_seasonal_factor(self) -> float:
        """Detect seasonal patterns (e.g., end-of-quarter rush)."""
        now = datetime.utcnow()
        day_of_month = now.day
        month = now.month
        
        # End of quarter (March, June, Sept, Dec)
        if month in [3, 6, 9, 12] and day_of_month >= 25:
            return 0.9
        
        # End of month
        if day_of_month >= 28:
            return 0.7
        
        # Holiday seasons (Nov-Dec)
        if month in [11, 12]:
            return 0.6
        
        return 0.5
    
    def _calculate_time_of_day_risk(self) -> float:
        """Adjust risk based on time of day (after-hours = higher risk)."""
        hour = datetime.utcnow().hour
        
        # Night hours (10 PM - 6 AM UTC) → higher risk
        if hour >= 22 or hour < 6:
            return 0.8
        
        # Business hours (9 AM - 5 PM UTC)
        if 9 <= hour <= 17:
            return 0.3
        
        return 0.5
    
    # ------------------------------------------------------------------
    # 4. Historical Context
    # ------------------------------------------------------------------
    
    def _get_historical_success_rate(self, step: Dict[str, Any]) -> float:
        """Retrieve success rate for similar steps from history."""
        step_type = step.get('type', 'unknown')
        
        if step_type in self.historical_data:
            return self.historical_data[step_type].get('success_rate', 0.8)
        
        # Deterministic fallback based on step hash
        step_hash = hashlib.sha256(step_type.encode()).hexdigest()
        pseudo_rate = 0.6 + (int(step_hash[:8], 16) % 30) / 100
        return pseudo_rate
    
    def update_historical_data(
        self, 
        step_type: str, 
        success: bool, 
        duration: float
    ):
        """Update historical statistics (called after step completion)."""
        if step_type not in self.historical_data:
            self.historical_data[step_type] = {
                'total_runs': 0,
                'successes': 0,
                'total_duration': 0.0
            }
        
        stats = self.historical_data[step_type]
        stats['total_runs'] += 1
        stats['successes'] += int(success)
        stats['total_duration'] += duration
        stats['success_rate'] = stats['successes'] / stats['total_runs']
        stats['avg_duration'] = stats['total_duration'] / stats['total_runs']
    
    # ------------------------------------------------------------------
    # 5. Stakeholder Context
    # ------------------------------------------------------------------
    
    def _get_assignee_workload(
        self, 
        step: Dict[str, Any], 
        workflow_state: Dict[str, Any]
    ) -> float:
        """Estimate current workload of assigned resource."""
        assignee = step.get('assignee') or workflow_state.get('default_assignee')
        
        # Could integrate with task management system
        # For now, use workflow state hint
        workload = workflow_state.get(f'{assignee}_workload', 0.5)
        return max(0.0, min(1.0, workload))
    
    def _get_assignee_expertise(self, step: Dict[str, Any]) -> float:
        """Match assignee skill to task requirements."""
        # Placeholder: could integrate with HR/skill database
        return 0.7
    
    def _check_assignee_availability(self, step: Dict[str, Any]) -> bool:
        """Check if assignee is available (not on leave, etc.)."""
        # Placeholder: could integrate with calendar system
        return True
    
    # ------------------------------------------------------------------
    # 6. Environmental Context
    # ------------------------------------------------------------------
    
    def _get_system_health(self, workflow_state: Dict[str, Any]) -> float:
        """Check infrastructure health."""
        return workflow_state.get('system_health', 1.0)
    
    def _get_concurrent_load(self, workflow_state: Dict[str, Any]) -> float:
        """Measure system congestion."""
        active_workflows = workflow_state.get('active_workflow_count', 10)
        max_capacity = workflow_state.get('max_workflow_capacity', 100)
        
        load = active_workflows / max_capacity if max_capacity > 0 else 0.5
        return max(0.0, min(1.0, load))
    
    # ------------------------------------------------------------------
    # 7. Context-Aware Risk Adjustment
    # ------------------------------------------------------------------
    
    def adjust_rpn_with_context(
        self, 
        base_rpn: int, 
        context: ContextSnapshot
    ) -> Dict[str, Any]:
        """
        Adjust FMEA RPN based on context.
        Returns enhanced risk score with explanation.
        """
        context_severity = context.aggregate_score()
        
        # Multiplier based on context (1.0 - 2.0 range)
        multiplier = 1.0 + context_severity
        
        adjusted_rpn = int(base_rpn * multiplier)
        adjusted_rpn = min(adjusted_rpn, 1000)  # Cap at 1000
        
        explanation = self._generate_explanation(base_rpn, adjusted_rpn, context)
        
        return {
            'base_rpn': base_rpn,
            'context_adjusted_rpn': adjusted_rpn,
            'context_severity': round(context_severity, 3),
            'multiplier': round(multiplier, 2),
            'explanation': explanation,
            'context_snapshot': context.to_dict()
        }
    
    def _generate_explanation(
        self, 
        base_rpn: int, 
        adjusted_rpn: int, 
        context: ContextSnapshot
    ) -> str:
        """Generate human-readable context explanation."""
        factors = []
        
        if context.customer_value > 0.8:
            factors.append("high-value customer")
        if context.urgency_level > 0.8:
            factors.append("critical urgency")
        if context.compliance_risk > 0.7:
            factors.append("regulatory sensitivity")
        if context.time_pressure > 0.7:
            factors.append("deadline pressure")
        if context.success_rate < 0.6:
            factors.append("low historical success rate")
        if context.assignee_workload > 0.8:
            factors.append("assignee overload")
        if context.system_health < 0.7:
            factors.append("degraded system health")
        
        if not factors:
            return f"RPN {base_rpn} → {adjusted_rpn} (no significant context factors)"
        
        factor_str = ", ".join(factors)
        return f"RPN {base_rpn} → {adjusted_rpn} due to: {factor_str}"


# ----------------------------------------------------------------------
# Manual Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    engine = ContextEngine()
    
    # Simulate a high-risk scenario
    step = {
        'id': 'approve_enterprise_contract',
        'type': 'approval',
        'description': 'Urgent contract approval required',
        'priority': 'critical',
        'tags': ['compliance', 'sox']
    }
    
    workflow_state = {
        'customer_tier': 'enterprise',
        'customer_annual_revenue': 5_000_000,
        'deadline': (datetime.utcnow() + timedelta(hours=3)).isoformat(),
        'sla_breach_imminent': True,
        'system_health': 0.85,
        'active_workflow_count': 75,
        'max_workflow_capacity': 100
    }
    
    context = engine.collect_context(step, workflow_state)
    print("=== Context Snapshot ===")
    print(f"Customer Value: {context.customer_value}")
    print(f"Urgency: {context.urgency_level}")
    print(f"Compliance Risk: {context.compliance_risk}")
    print(f"Time Pressure: {context.time_pressure}")
    print(f"Aggregate Severity: {context.aggregate_score()}")
    
    print("\n=== Context-Adjusted Risk ===")
    base_rpn = 120  # From FMEA
    adjusted = engine.adjust_rpn_with_context(base_rpn, context)
    print(f"Base RPN: {adjusted['base_rpn']}")
    print(f"Adjusted RPN: {adjusted['context_adjusted_rpn']}")
    print(f"Explanation: {adjusted['explanation']}")

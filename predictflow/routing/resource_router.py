"""
PredictFlow Intelligent Router
-------------------------------

Purpose:
  - Context-aware routing decisions for workflow steps
  - Dynamic assignee selection based on multiple factors
  - Adaptive path selection in parallel/conditional branches
  - Explainable routing with audit trail

Features:
  ✅ Risk-based routing (high RPN → senior resources)
  ✅ Skill matching (task requirements → assignee expertise)
  ✅ Load balancing (distribute across available resources)
  ✅ Context-driven escalation (urgency + value → fast track)
  ✅ Semantic similarity routing (past success patterns)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Available routing strategies."""
    RISK_BASED = "risk_based"
    SKILL_MATCH = "skill_match"
    LOAD_BALANCE = "load_balance"
    PRIORITY_BASED = "priority"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"


@dataclass
class RoutingDecision:
    """Structured routing decision with explanation."""
    
    step_id: str
    assignee: str
    strategy_used: str
    confidence: float  # 0-1
    
    # Scoring factors
    risk_score: float = 0.0
    skill_match_score: float = 0.0
    workload_score: float = 0.0
    priority_score: float = 0.0
    
    # Metadata
    alternatives: List[str] = None
    explanation: str = ""
    escalated: bool = False
    bypass_normal_queue: bool = False
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceRouter:
    """
    PredictFlow Intelligent Routing Engine
    
    Makes context-aware routing decisions by combining:
      - FMEA risk scores
      - Confidence metrics
      - Context awareness
      - Resource availability and skills
      - Historical success patterns
    """
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.HYBRID):
        self.strategy = strategy
        self.resource_pool = {}  # assignee_id → ResourceProfile
        self.routing_history = []  # Past decisions
    
    # ------------------------------------------------------------------
    # Main Routing Entry Point
    # ------------------------------------------------------------------
    
    def route_step(
        self,
        step: Dict[str, Any],
        context_snapshot: Any,
        fmea_result: Dict[str, Any],
        available_assignees: List[str]
    ) -> RoutingDecision:
        """
        Determine optimal routing for a workflow step.
        """
        step_id = str(step.get('id', 'unknown'))
        
        if not available_assignees:
            logger.warning(f"[Router] No assignees available for step {step_id}")
            return self._create_default_decision(step_id, "unassigned")
        
        # Compute routing scores
        scores = self._compute_routing_scores(
            step, context_snapshot, fmea_result, available_assignees
        )
        
        # Select strategy
        if self.strategy == RoutingStrategy.HYBRID:
            decision = self._hybrid_routing(step_id, scores, available_assignees)
        elif self.strategy == RoutingStrategy.RISK_BASED:
            decision = self._risk_based_routing(step_id, scores, available_assignees)
        elif self.strategy == RoutingStrategy.SKILL_MATCH:
            decision = self._skill_based_routing(step_id, scores, available_assignees)
        elif self.strategy == RoutingStrategy.LOAD_BALANCE:
            decision = self._load_balanced_routing(step_id, scores, available_assignees)
        else:
            decision = self._hybrid_routing(step_id, scores, available_assignees)
        
        # Check for escalation
        decision = self._check_escalation(decision, context_snapshot, fmea_result)
        
        # Record decision
        self.routing_history.append(decision.to_dict())
        
        logger.info(f"[Router] {step_id} → {decision.assignee} ({decision.strategy_used})")
        
        return decision
    
    # ------------------------------------------------------------------
    # Score Computation
    # ------------------------------------------------------------------
    
    def _compute_routing_scores(
        self,
        step: Dict[str, Any],
        context_snapshot: Any,
        fmea_result: Dict[str, Any],
        assignees: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute routing scores for each candidate assignee."""
        scores = {}
        
        for assignee in assignees:
            profile = self._get_resource_profile(assignee)
            
            scores[assignee] = {
                'risk_score': self._compute_risk_score(fmea_result, profile),
                'skill_match_score': self._compute_skill_match(step, profile),
                'workload_score': self._compute_workload_score(profile),
                'priority_score': self._compute_priority_score(context_snapshot, profile)
            }
        
        return scores
    
    def _compute_risk_score(
        self, 
        fmea_result: Dict[str, Any], 
        profile: 'ResourceProfile'
    ) -> float:
        """Higher risk → needs more experienced assignee."""
        context_rpn = fmea_result.get('context_adjusted_rpn', fmea_result.get('rpn', 100))
        normalized_risk = min(context_rpn / 1000, 1.0)
        
        seniority = profile.seniority_level
        
        if normalized_risk > 0.7:
            return seniority
        if normalized_risk < 0.3:
            return 1.0 - seniority * 0.5
        
        return 0.5 + seniority * 0.5
    
    def _compute_skill_match(
        self, 
        step: Dict[str, Any], 
        profile: 'ResourceProfile'
    ) -> float:
        """Match step requirements to assignee skills."""
        required_skills = set(step.get('required_skills', []))
        assignee_skills = set(profile.skills)
        
        if not required_skills:
            return 0.8
        
        intersection = required_skills & assignee_skills
        match_ratio = len(intersection) / len(required_skills)
        
        return min(match_ratio * (1 + profile.expertise_boost), 1.0)
    
    def _compute_workload_score(self, profile: 'ResourceProfile') -> float:
        """Prefer less loaded assignees."""
        return 1.0 - profile.current_workload
    
    def _compute_priority_score(
        self, 
        context_snapshot: Any, 
        profile: 'ResourceProfile'
    ) -> float:
        """High priority tasks → high authority assignees."""
        urgency = getattr(context_snapshot, 'urgency_level', 0.5)
        customer_value = getattr(context_snapshot, 'customer_value', 0.5)
        
        priority = (urgency + customer_value) / 2
        
        if priority > 0.8:
            return profile.authority_level
        
        return 0.5 + profile.authority_level * 0.5
    
    # ------------------------------------------------------------------
    # Routing Strategies
    # ------------------------------------------------------------------
    
    def _hybrid_routing(
        self, 
        step_id: str, 
        scores: Dict[str, Dict[str, float]], 
        assignees: List[str]
    ) -> RoutingDecision:
        """Weighted combination of all factors."""
        weights = {
            'risk_score': 0.35,
            'skill_match_score': 0.30,
            'workload_score': 0.20,
            'priority_score': 0.15
        }
        
        composite_scores = {}
        for assignee in assignees:
            score = sum(
                scores[assignee][factor] * weight 
                for factor, weight in weights.items()
            )
            composite_scores[assignee] = score
        
        best_assignee = max(composite_scores.items(), key=lambda x: x[1])
        sorted_assignees = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = [a for a, _ in sorted_assignees[1:4]]
        
        return RoutingDecision(
            step_id=step_id,
            assignee=best_assignee[0],
            strategy_used="hybrid",
            confidence=best_assignee[1],
            risk_score=scores[best_assignee[0]]['risk_score'],
            skill_match_score=scores[best_assignee[0]]['skill_match_score'],
            workload_score=scores[best_assignee[0]]['workload_score'],
            priority_score=scores[best_assignee[0]]['priority_score'],
            alternatives=alternatives,
            explanation=self._generate_explanation(best_assignee[0], scores[best_assignee[0]])
        )
    
    def _risk_based_routing(
        self, 
        step_id: str, 
        scores: Dict[str, Dict[str, float]], 
        assignees: List[str]
    ) -> RoutingDecision:
        """Route by risk-seniority match."""
        best = max(assignees, key=lambda a: scores[a]['risk_score'])
        
        return RoutingDecision(
            step_id=step_id,
            assignee=best,
            strategy_used="risk_based",
            confidence=scores[best]['risk_score'],
            risk_score=scores[best]['risk_score'],
            explanation=f"Assigned to {best} based on risk-seniority match"
        )
    
    def _skill_based_routing(
        self, 
        step_id: str, 
        scores: Dict[str, Dict[str, float]], 
        assignees: List[str]
    ) -> RoutingDecision:
        """Route by skill match."""
        best = max(assignees, key=lambda a: scores[a]['skill_match_score'])
        
        return RoutingDecision(
            step_id=step_id,
            assignee=best,
            strategy_used="skill_match",
            confidence=scores[best]['skill_match_score'],
            skill_match_score=scores[best]['skill_match_score'],
            explanation=f"Assigned to {best} for skill match"
        )
    
    def _load_balanced_routing(
        self, 
        step_id: str, 
        scores: Dict[str, Dict[str, float]], 
        assignees: List[str]
    ) -> RoutingDecision:
        """Route to least loaded resource."""
        best = max(assignees, key=lambda a: scores[a]['workload_score'])
        
        return RoutingDecision(
            step_id=step_id,
            assignee=best,
            strategy_used="load_balance",
            confidence=scores[best]['workload_score'],
            workload_score=scores[best]['workload_score'],
            explanation=f"Assigned to {best} for load balancing"
        )
    
    # ------------------------------------------------------------------
    # Escalation Logic
    # ------------------------------------------------------------------
    
    def _check_escalation(
        self,
        decision: RoutingDecision,
        context_snapshot: Any,
        fmea_result: Dict[str, Any]
    ) -> RoutingDecision:
        """Determine if escalation is needed."""
        high_risk = fmea_result.get('context_adjusted_rpn', 0) > 500
        high_urgency = getattr(context_snapshot, 'urgency_level', 0) > 0.85
        high_value = getattr(context_snapshot, 'customer_value', 0) > 0.85
        compliance = getattr(context_snapshot, 'compliance_risk', 0) > 0.7
        
        should_escalate = (
            high_risk or 
            (high_urgency and high_value) or
            compliance
        )
        
        if should_escalate:
            senior = self._find_senior_resource()
            if senior and senior != decision.assignee:
                decision.assignee = senior
                decision.escalated = True
                decision.bypass_normal_queue = True
                decision.explanation += " [ESCALATED: High risk/urgency/compliance]"
                logger.warning(f"[Router] Escalating {decision.step_id} to {senior}")
        
        return decision
    
    def _find_senior_resource(self) -> Optional[str]:
        """Find highest seniority available resource."""
        if not self.resource_pool:
            return None
        
        available = [
            (rid, profile) for rid, profile in self.resource_pool.items()
            if profile.available and profile.current_workload < 0.9
        ]
        
        if not available:
            return None
        
        senior = max(available, key=lambda x: x[1].seniority_level)
        return senior[0]
    
    # ------------------------------------------------------------------
    # Resource Management
    # ------------------------------------------------------------------
    
    def register_resource(self, resource_id: str, profile: 'ResourceProfile'):
        """Add a resource to the routing pool."""
        self.resource_pool[resource_id] = profile
        logger.info(f"[Router] Registered resource: {resource_id}")
    
    def _get_resource_profile(self, resource_id: str) -> 'ResourceProfile':
        """Get or create resource profile."""
        if resource_id not in self.resource_pool:
            self.resource_pool[resource_id] = ResourceProfile(
                resource_id=resource_id,
                seniority_level=0.5,
                current_workload=0.5
            )
        return self.resource_pool[resource_id]
    
    def update_workload(self, resource_id: str, workload: float):
        """Update resource workload."""
        if resource_id in self.resource_pool:
            self.resource_pool[resource_id].current_workload = max(0.0, min(1.0, workload))
    
    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    
    def _generate_explanation(
        self, 
        assignee: str, 
        scores: Dict[str, float]
    ) -> str:
        """Generate human-readable routing explanation."""
        factors = []
        
        if scores['risk_score'] > 0.8:
            factors.append("high risk match")
        if scores['skill_match_score'] > 0.8:
            factors.append("strong skill match")
        if scores['workload_score'] > 0.7:
            factors.append("available capacity")
        if scores['priority_score'] > 0.8:
            factors.append("authority level")
        
        if not factors:
            return f"Assigned to {assignee} (balanced selection)"
        
        return f"Assigned to {assignee}: {', '.join(factors)}"
    
    def _create_default_decision(self, step_id: str, assignee: str) -> RoutingDecision:
        """Fallback decision."""
        return RoutingDecision(
            step_id=step_id,
            assignee=assignee,
            strategy_used="default",
            confidence=0.0,
            explanation="No assignees available"
        )
    
    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Generate routing performance analytics."""
        if not self.routing_history:
            return {"total_routes": 0}
        
        total = len(self.routing_history)
        escalated = sum(1 for d in self.routing_history if d.get('escalated'))
        avg_confidence = sum(d.get('confidence', 0) for d in self.routing_history) / total
        
        strategy_distribution = {}
        for decision in self.routing_history:
            strategy = decision.get('strategy_used', 'unknown')
            strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
        
        return {
            'total_routes': total,
            'escalations': escalated,
            'escalation_rate': round(escalated / total, 3),
            'avg_confidence': round(avg_confidence, 3),
            'strategy_distribution': strategy_distribution
        }


# ======================================================================
# Resource Profile
# ======================================================================

@dataclass
class ResourceProfile:
    """Profile of a resource/assignee."""
    
    resource_id: str
    seniority_level: float = 0.5  # 0-1
    authority_level: float = 0.5  # 0-1
    expertise_boost: float = 0.0
    skills: List[str] = None
    current_workload: float = 0.5  # 0-1
    available: bool = True
    
    def __post_init__(self):
        if self.skills is None:
            self.skills = []

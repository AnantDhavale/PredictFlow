# predictflow/context/semantic_rules.py

class SemanticRuleEngine:
    """
    Interpret natural language business rules and apply context.
    
    Example rules:
      - "If customer is enterprise AND deadline < 24h, escalate to VP"
      - "During end-of-quarter, increase approval thresholds by 20%"
      - "After 3 failures of same type, trigger manual review"
    """
    
    def __init__(self, adapter=None):
        self.adapter = adapter or ConfidenceAdapterFactory.get_adapter()
        self.rules = []
    
    def add_rule(self, rule_text: str, action: callable):
        """Register a semantic rule with an action."""
        self.rules.append({
            'text': rule_text,
            'embedding': self.adapter.embed(rule_text),
            'action': action
        })
    
    def evaluate(self, step: Dict[str, Any], context: ContextSnapshot) -> List[Dict]:
        """
        Match step against rules using semantic similarity.
        Returns triggered actions.
        """
        step_description = f"{step.get('description', '')} priority={step.get('priority')}"
        step_embedding = self.adapter.embed(step_description)
        
        triggered = []
        for rule in self.rules:
            similarity = cosine_similarity(step_embedding, rule['embedding'])
            
            # Also check structured context conditions
            if similarity > 0.7 and self._check_conditions(rule['text'], context):
                triggered.append({
                    'rule': rule['text'],
                    'similarity': similarity,
                    'action': rule['action']
                })
        
        return triggered
    
    def _check_conditions(self, rule_text: str, context: ContextSnapshot) -> bool:
        """Parse and evaluate context conditions from rule text."""
        # Simple keyword matching (could use LLM for better parsing)
        if 'enterprise' in rule_text.lower():
            if context.customer_value < 0.8:
                return False
        
        if 'deadline' in rule_text.lower() and '<' in rule_text:
            if context.time_pressure < 0.7:
                return False
        
        if 'end-of-quarter' in rule_text.lower():
            if context.seasonal_factor < 0.8:
                return False
        
        return True

"""
Real-time Failure Prediction Engine
-----------------------------------

Purpose:
  - Real-time failure prediction during workflow execution
  - Integration with PredictFlow executor
  - Caching and performance optimization
  - Explainability (attention weights)
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import deque

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from predictflow.prediction.lstm_predictor import LSTMFailurePredictor, load_pretrained_model
from predictflow.prediction.data_processor import FailureDataProcessor

logger = logging.getLogger(__name__)


class FailurePredictionEngine:
    """
    Real-time failure prediction engine for workflow execution.
    
    Features:
      - Maintains execution history per step
      - Real-time prediction with caching
      - Attention-based explainability
      - Performance monitoring
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        sequence_length: int = 10,
        cache_size: int = 1000
    ):
        """
        Initialize prediction engine.
        
        Args:
            model_path: Path to trained model checkpoint
            sequence_length: Number of historical executions to use
            cache_size: Maximum cache size per step
        """
        self.sequence_length = sequence_length
        self.cache_size = cache_size
        
        # Load model
        if model_path and TORCH_AVAILABLE:
            try:
                self.model = load_pretrained_model(model_path)
                logger.info(f"Loaded trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}, using untrained model")
                self.model = None
        else:
            self.model = None
            logger.warning("No model loaded - predictions will use fallback heuristics")
        
        # Data processor
        self.processor = FailureDataProcessor(
            sequence_length=sequence_length,
            normalize=True
        )
        
        # Execution history cache per step
        self.execution_history: Dict[str, deque] = {}
        
        # Prediction cache (avoid recomputing)
        self.prediction_cache: Dict[str, float] = {}
        
        # Performance metrics
        self.stats = {
            'predictions_made': 0,
            'cache_hits': 0,
            'avg_inference_time': 0.0
        }
    
    def predict_failure_probability(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Predict failure probability for an upcoming step execution.
        
        Args:
            step: Step metadata (id, description, FMEA scores, etc.)
            context: Current workflow context (system state, workload, etc.)
            use_cache: Whether to use cached predictions
            
        Returns:
            Dictionary with:
              - failure_probability: float (0-1)
              - confidence: prediction confidence
              - explanation: human-readable explanation
              - attention_weights: importance of historical time steps
        """
        step_id = step.get('id', 'unknown')
        
        # Check cache
        cache_key = self._generate_cache_key(step, context)
        if use_cache and cache_key in self.prediction_cache:
            self.stats['cache_hits'] += 1
            return {
                'failure_probability': self.prediction_cache[cache_key],
                'confidence': 0.8,
                'explanation': 'Cached prediction',
                'attention_weights': None,
                'from_cache': True
            }
        
        # Get execution history
        if step_id not in self.execution_history:
            # No history yet - use fallback
            return self._fallback_prediction(step, context)
        
        history = list(self.execution_history[step_id])
        
        if len(history) < self.sequence_length:
            # Insufficient history
            return self._fallback_prediction(step, context)
        
        # Prepare features
        features = self.processor.process_single_sequence(history, step_id)
        
        if features is None:
            return self._fallback_prediction(step, context)
        
        # Model prediction
        if self.model is not None:
            import time
            start_time = time.time()
            
            failure_prob = float(self.model.predict_proba(features)[0])
            attention = self.model.get_attention_weights(features)
            
            inference_time = time.time() - start_time
            self._update_stats(inference_time)
            
            # Generate explanation
            explanation = self._generate_explanation(
                step, failure_prob, attention, history
            )
            
            result = {
                'failure_probability': failure_prob,
                'confidence': self._compute_confidence(history),
                'explanation': explanation,
                'attention_weights': attention.tolist() if attention is not None else None,
                'from_cache': False,
                'inference_time_ms': inference_time * 1000
            }
            
            # Cache result
            self.prediction_cache[cache_key] = failure_prob
            
            return result
        
        else:
            # Fallback if no model
            return self._fallback_prediction(step, context)
    
    def record_execution(
        self,
        step_id: str,
        execution_data: Dict[str, Any]
    ):
        """
        Record a step execution for future predictions.
        
        Args:
            step_id: Step identifier
            execution_data: Execution record with outcome and metrics
        """
        if step_id not in self.execution_history:
            self.execution_history[step_id] = deque(maxlen=self.cache_size)
        
        self.execution_history[step_id].append(execution_data)
        
        # Invalidate prediction cache for this step
        self._invalidate_cache(step_id)
        
        logger.debug(f"Recorded execution for {step_id} ({len(self.execution_history[step_id])} total)")
    
    def _fallback_prediction(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback prediction when LSTM model unavailable or insufficient data.
        Uses FMEA and context heuristics.
        """
        # Extract risk indicators
        rpn = step.get('rpn', 100)
        context_adjusted_rpn = step.get('context_adjusted_rpn', rpn)
        confidence = step.get('confidence', 0.5)
        context_severity = context.get('context_severity', 0.5)
        
        # Simple heuristic formula
        # High risk + low confidence + high context severity = higher failure probability
        normalized_rpn = min(context_adjusted_rpn / 1000.0, 1.0)
        failure_prob = (
            0.4 * normalized_rpn +
            0.3 * (1.0 - confidence) +
            0.3 * context_severity
        )
        
        failure_prob = max(0.0, min(1.0, failure_prob))
        
        return {
            'failure_probability': failure_prob,
            'confidence': 0.5,  # Low confidence for heuristic
            'explanation': f'Heuristic prediction based on RPN={context_adjusted_rpn}, confidence={confidence:.2f}',
            'attention_weights': None,
            'from_cache': False,
            'fallback': True
        }
    
    def _generate_explanation(
        self,
        step: Dict[str, Any],
        failure_prob: float,
        attention: Optional[np.ndarray],
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable explanation for prediction."""
        risk_level = "high" if failure_prob > 0.7 else "medium" if failure_prob > 0.4 else "low"
        
        explanation = f"Predicted {risk_level} failure risk ({failure_prob:.1%})"
        
        # Add attention-based insights
        if attention is not None and len(attention.shape) > 0:
            attention_flat = attention.flatten()
            most_important_idx = int(np.argmax(attention_flat))
            
            if most_important_idx < len(history):
                important_exec = history[most_important_idx]
                if important_exec.get('failed'):
                    explanation += f" - Recent failure pattern detected"
                elif important_exec.get('rpn', 0) > 300:
                    explanation += f" - High-risk execution in recent history"
        
        # Add current context
        if step.get('context_adjusted_rpn', 0) > 400:
            explanation += " - Current context increases risk"
        
        return explanation
    
    def _compute_confidence(self, history: List[Dict[str, Any]]) -> float:
        """
        Compute prediction confidence based on history quality.
        
        More history + diverse patterns = higher confidence
        """
        history_length = len(history)
        
        # Base confidence from history length
        length_confidence = min(history_length / (self.sequence_length * 3), 1.0)
        
        # Diversity: mix of successes and failures improves confidence
        failures = sum(1 for h in history if h.get('failed', False))
        successes = history_length - failures
        
        if failures > 0 and successes > 0:
            diversity_score = 1.0
        elif history_length > 20:
            diversity_score = 0.7  # Lots of data, but all same outcome
        else:
            diversity_score = 0.5
        
        confidence = 0.6 * length_confidence + 0.4 * diversity_score
        
        return round(confidence, 2)
    
    def _generate_cache_key(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate cache key for a prediction."""
        step_id = step.get('id', 'unknown')
        rpn = step.get('context_adjusted_rpn', step.get('rpn', 0))
        workload = context.get('assignee_workload', 0.5)
        
        return f"{step_id}:{rpn}:{workload:.2f}"
    
    def _invalidate_cache(self, step_id: str):
        """Invalidate cache entries for a step after new execution."""
        keys_to_remove = [k for k in self.prediction_cache if k.startswith(f"{step_id}:")]
        for key in keys_to_remove:
            del self.prediction_cache[key]
    
    def _update_stats(self, inference_time: float):
        """Update performance statistics."""
        self.stats['predictions_made'] += 1
        
        # Running average of inference time
        n = self.stats['predictions_made']
        prev_avg = self.stats['avg_inference_time']
        self.stats['avg_inference_time'] = (prev_avg * (n - 1) + inference_time) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        cache_hit_rate = 0.0
        if self.stats['predictions_made'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / self.stats['predictions_made']
        
        return {
            'predictions_made': self.stats['predictions_made'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': round(cache_hit_rate, 3),
            'avg_inference_time_ms': round(self.stats['avg_inference_time'] * 1000, 2),
            'steps_tracked': len(self.execution_history),
            'total_history_records': sum(len(h) for h in self.execution_history.values())
        }
    
    def clear_cache(self):
        """Clear prediction cache (useful after model retraining)."""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")


# Integration helper for PredictFlow Executor
class PredictionIntegration:
    """
    Helper class to integrate prediction engine with PredictFlow Executor.
    """
    
    def __init__(self, engine: FailurePredictionEngine):
        self.engine = engine
    
    def before_step_hook(self, context: Dict[str, Any], step: Dict[str, Any]):
        """
        Hook to run before step execution.
        Makes failure prediction and stores in context.
        """
        prediction = self.engine.predict_failure_probability(step, context)
        
        # Store prediction in context
        step['failure_prediction'] = prediction
        context[f"{step['id']}_predicted_failure_prob"] = prediction['failure_probability']
        
        # Log high-risk predictions
        if prediction['failure_probability'] > 0.7:
            logger.warning(
                f"[Prediction] High failure risk for {step['id']}: "
                f"{prediction['failure_probability']:.1%} - {prediction['explanation']}"
            )
    
    def after_step_hook(self, context: Dict[str, Any], step: Dict[str, Any]):
        """
        Hook to run after step execution.
        Records execution outcome for learning.
        """
        from datetime import datetime
        
        step_id = step['id']
        
        # Extract execution data
        execution_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'rpn': step.get('rpn', 100),
            'confidence': step.get('confidence', 0.5),
            'context_severity': step.get('context_snapshot', {}).get('aggregate_score', 0.5),
            'assignee_workload': step.get('routing_decision', {}).get('workload_score', 0.5),
            'duration': step.get('duration', 60),
            'queue_time': step.get('queue_time', 0),
            'system_health': context.get('workflow_state', {}).get('system_health', 1.0),
            'failed': step.get('failed', context.get('last_error') is not None)
        }
        
        # Record execution
        self.engine.record_execution(step_id, execution_record)


# Example usage
if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    # Create engine
    engine = FailurePredictionEngine(sequence_length=10)
    
    # Simulate historical executions
    print("Simulating execution history...")
    for i in range(15):
        engine.record_execution("validate_order", {
            'timestamp': (datetime.now() - timedelta(hours=15-i)).isoformat(),
            'rpn': np.random.randint(80, 250),
            'confidence': np.random.uniform(0.6, 0.9),
            'context_severity': np.random.uniform(0.3, 0.7),
            'assignee_workload': np.random.uniform(0.3, 0.7),
            'duration': np.random.uniform(40, 180),
            'queue_time': np.random.uniform(0, 30),
            'system_health': np.random.uniform(0.85, 1.0),
            'failed': np.random.random() < 0.1
        })
    
    # Make prediction
    print("\nMaking prediction for next execution...")
    step = {
        'id': 'validate_order',
        'rpn': 180,
        'confidence': 0.75,
        'context_adjusted_rpn': 240
    }
    
    context = {
        'assignee_workload': 0.6,
        'context_severity': 0.55
    }
    
    prediction = engine.predict_failure_probability(step, context)
    
    print(f"\nPrediction Results:")
    print(f"  Failure Probability: {prediction['failure_probability']:.1%}")
    print(f"  Confidence: {prediction['confidence']:.2f}")
    print(f"  Explanation: {prediction['explanation']}")
    print(f"  Fallback Used: {prediction.get('fallback', False)}")
    
    # Show statistics
    print(f"\nEngine Statistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

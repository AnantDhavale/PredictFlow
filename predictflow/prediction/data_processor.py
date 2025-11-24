
"""
Failure Data Processor for LSTM Training
----------------------------------------

Purpose:
  - Process historical workflow execution data
  - Engineer features for LSTM training
  - Create time-series sequences
  - Handle data normalization and augmentation
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FailureDataProcessor:
    """
    Process workflow execution history into LSTM-ready sequences.
    
    Features extracted per execution:
      - RPN (risk priority number)
      - Confidence score
      - Context severity (aggregated)
      - Time-based features (hour, day of week, etc.)
      - Resource workload
      - Historical success rate
      - Step duration
      - Queue time
      - System health
      - Concurrent load
    """
    
    # Feature indices for reference
    FEATURE_NAMES = [
        'rpn_normalized',
        'confidence',
        'context_severity',
        'hour_of_day',
        'day_of_week',
        'workload',
        'historical_success_rate',
        'duration_normalized',
        'queue_time_normalized',
        'system_health'
    ]
    
    def __init__(
        self,
        sequence_length: int = 10,
        normalize: bool = True
    ):
        """
        Initialize data processor.
        
        Args:
            sequence_length: Number of past executions to use as context
            normalize: Whether to normalize features
        """
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Normalization parameters (learned from training data)
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        
        # Statistics
        self.step_statistics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_executions': 0,
            'failures': 0,
            'success_rate': 1.0,
            'avg_duration': 0.0,
            'recent_failures': []
        })
    
    def process_execution_history(
        self,
        executions: List[Dict[str, Any]],
        step_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert execution history into training sequences.
        
        Args:
            executions: List of execution records for a step
            step_id: Step identifier
            
        Returns:
            X: Feature sequences (num_sequences, seq_len, num_features)
            y: Target labels (num_sequences,) - 1 for failure, 0 for success
        """
        if len(executions) < self.sequence_length + 1:
            logger.warning(
                f"Not enough history for {step_id}: "
                f"{len(executions)} < {self.sequence_length + 1}"
            )
            return np.array([]), np.array([])
        
        # Sort by timestamp
        executions = sorted(executions, key=lambda x: x.get('timestamp', ''))
        
        # Extract features for each execution
        features_list = []
        labels_list = []
        
        for i in range(len(executions) - self.sequence_length):
            # Sequence of past executions
            sequence = executions[i:i + self.sequence_length]
            
            # Target: next execution outcome
            target = executions[i + self.sequence_length]
            
            # Extract features
            sequence_features = [
                self._extract_features(exec_record, step_id) 
                for exec_record in sequence
            ]
            
            features_list.append(sequence_features)
            labels_list.append(1 if target.get('failed', False) else 0)
        
        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.float32)
        
        # Normalize features
        if self.normalize:
            X = self._normalize_features(X)
        
        logger.info(
            f"Processed {len(X)} sequences for {step_id} "
            f"(failure rate: {y.mean():.2%})"
        )
        
        return X, y
    
    def _extract_features(
        self,
        execution: Dict[str, Any],
        step_id: str
    ) -> List[float]:
        """
        Extract feature vector from a single execution record.
        
        Expected execution record format:
        {
            'timestamp': ISO string,
            'rpn': int,
            'confidence': float,
            'context_severity': float,
            'assignee_workload': float,
            'duration': float (seconds),
            'queue_time': float (seconds),
            'system_health': float,
            'failed': bool
        }
        """
        # Parse timestamp
        try:
            ts = datetime.fromisoformat(execution.get('timestamp', ''))
            hour_of_day = ts.hour / 24.0  # Normalize to 0-1
            day_of_week = ts.weekday() / 7.0  # Normalize to 0-1
        except Exception:
            hour_of_day = 0.5
            day_of_week = 0.5
        
        # Extract and normalize features
        features = [
            min(execution.get('rpn', 100) / 1000.0, 1.0),  # RPN normalized
            execution.get('confidence', 0.5),  # Already 0-1
            execution.get('context_severity', 0.5),  # Already 0-1
            hour_of_day,
            day_of_week,
            execution.get('assignee_workload', 0.5),  # Already 0-1
            self.step_statistics[step_id].get('success_rate', 1.0),
            min(execution.get('duration', 60) / 3600.0, 1.0),  # Duration in hours
            min(execution.get('queue_time', 0) / 3600.0, 1.0),  # Queue time in hours
            execution.get('system_health', 1.0)  # Already 0-1
        ]
        
        # Update statistics
        self._update_statistics(step_id, execution)
        
        return features
    
    def _update_statistics(self, step_id: str, execution: Dict[str, Any]):
        """Update running statistics for a step."""
        stats = self.step_statistics[step_id]
        stats['total_executions'] += 1
        
        if execution.get('failed', False):
            stats['failures'] += 1
        
        # Calculate success rate (with smoothing)
        total = stats['total_executions']
        failures = stats['failures']
        stats['success_rate'] = (total - failures + 1) / (total + 2)  # Laplace smoothing
        
        # Update average duration
        duration = execution.get('duration', 0)
        prev_avg = stats['avg_duration']
        stats['avg_duration'] = (prev_avg * (total - 1) + duration) / total
    
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize features using Z-score normalization.
        
        Args:
            X: Feature array (num_sequences, seq_len, num_features)
            
        Returns:
            Normalized features
        """
        if self.feature_means is None:
            # Compute normalization parameters from data
            # Reshape to (num_samples * seq_len, num_features)
            X_reshaped = X.reshape(-1, X.shape[-1])
            
            self.feature_means = np.mean(X_reshaped, axis=0)
            self.feature_stds = np.std(X_reshaped, axis=0) + 1e-8  # Avoid division by zero
            
            logger.info("Computed normalization parameters")
        
        # Apply normalization
        X_normalized = (X - self.feature_means) / self.feature_stds
        
        return X_normalized
    
    def process_single_sequence(
        self,
        recent_executions: List[Dict[str, Any]],
        step_id: str
    ) -> Optional[np.ndarray]:
        """
        Process a single sequence for real-time prediction.
        
        Args:
            recent_executions: Recent execution history (must be >= sequence_length)
            step_id: Step identifier
            
        Returns:
            Feature array (1, seq_len, num_features) or None if insufficient data
        """
        if len(recent_executions) < self.sequence_length:
            logger.warning(
                f"Insufficient history for prediction: "
                f"{len(recent_executions)} < {self.sequence_length}"
            )
            return None
        
        # Take last N executions
        sequence = recent_executions[-self.sequence_length:]
        
        # Extract features
        features = [
            self._extract_features(exec_record, step_id)
            for exec_record in sequence
        ]
        
        X = np.array([features], dtype=np.float32)  # Add batch dimension
        
        # Normalize
        if self.normalize and self.feature_means is not None:
            X = (X - self.feature_means) / self.feature_stds
        
        return X
    
    def split_train_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        shuffle: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and test sets.
        
        Args:
            X: Features
            y: Labels
            test_size: Fraction of data for testing
            shuffle: Whether to shuffle before splitting
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        if shuffle:
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]
        
        X_train = X[:-n_test]
        X_test = X[-n_test:]
        y_train = y[:-n_test]
        y_test = y[-n_test:]
        
        logger.info(
            f"Split data: {len(X_train)} train, {len(X_test)} test "
            f"(test failure rate: {y_test.mean():.2%})"
        )
        
        return X_train, X_test, y_train, y_test
    
    def balance_classes(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'oversample'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance imbalanced classes.
        
        Args:
            X: Features
            y: Labels
            method: 'oversample' or 'undersample'
            
        Returns:
            Balanced X, y
        """
        failure_indices = np.where(y == 1)[0]
        success_indices = np.where(y == 0)[0]
        
        n_failures = len(failure_indices)
        n_successes = len(success_indices)
        
        if n_failures == n_successes:
            return X, y
        
        logger.info(
            f"Original class distribution: "
            f"{n_successes} successes, {n_failures} failures"
        )
        
        if method == 'oversample':
            # Oversample minority class
            if n_failures < n_successes:
                minority_indices = failure_indices
                target_size = n_successes
            else:
                minority_indices = success_indices
                target_size = n_failures
            
            # Random oversample with replacement
            oversampled_indices = np.random.choice(
                minority_indices,
                size=target_size - len(minority_indices),
                replace=True
            )
            
            all_indices = np.concatenate([
                np.arange(len(X)),
                oversampled_indices
            ])
            
        else:  # undersample
            # Undersample majority class
            if n_failures < n_successes:
                majority_indices = success_indices
                target_size = n_failures
            else:
                majority_indices = failure_indices
                target_size = n_successes
            
            # Random undersample
            undersampled_majority = np.random.choice(
                majority_indices,
                size=target_size,
                replace=False
            )
            
            minority_indices = failure_indices if n_failures < n_successes else success_indices
            
            all_indices = np.concatenate([
                minority_indices,
                undersampled_majority
            ])
        
        # Shuffle
        np.random.shuffle(all_indices)
        
        X_balanced = X[all_indices]
        y_balanced = y[all_indices]
        
        logger.info(
            f"Balanced distribution: {len(X_balanced)} samples, "
            f"failure rate: {y_balanced.mean():.2%}"
        )
        
        return X_balanced, y_balanced


# Example usage
if __name__ == "__main__":
    # Simulate execution history
    executions = []
    for i in range(50):
        executions.append({
            'timestamp': (datetime.now() - timedelta(hours=50-i)).isoformat(),
            'rpn': np.random.randint(50, 300),
            'confidence': np.random.uniform(0.5, 0.95),
            'context_severity': np.random.uniform(0.3, 0.8),
            'assignee_workload': np.random.uniform(0.2, 0.8),
            'duration': np.random.uniform(30, 300),
            'queue_time': np.random.uniform(0, 60),
            'system_health': np.random.uniform(0.8, 1.0),
            'failed': np.random.random() < 0.15  # 15% failure rate
        })
    
    processor = FailureDataProcessor(sequence_length=10)
    X, y = processor.process_execution_history(executions, "test_step")
    
    print(f"Generated {len(X)} training sequences")
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Failure rate: {y.mean():.2%}")
    
    # Balance classes
    X_balanced, y_balanced = processor.balance_classes(X, y, method='oversample')
    print(f"\nBalanced failure rate: {y_balanced.mean():.2%}")

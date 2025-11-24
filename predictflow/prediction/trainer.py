"""
LSTM Failure Prediction Trainer
-------------------------------

Purpose:
  - Train LSTM models on historical workflow data
  - Handle imbalanced classes
  - Early stopping and learning rate scheduling
  - Model evaluation and metrics
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from predictflow.prediction.lstm_predictor import LSTMFailurePredictor, ModelCheckpoint

logger = logging.getLogger(__name__)


class FailurePredictionTrainer:
    """
    Trainer for LSTM failure prediction models.
    
    Features:
      - Handles imbalanced datasets with weighted loss
      - Early stopping to prevent overfitting
      - Learning rate scheduling
      - Comprehensive metrics tracking
    """
    
    def __init__(
        self,
        model: LSTMFailurePredictor,
        device: str = "cpu",
        checkpoint_dir: str = "predictflow/prediction/models"
    ):
        """
        Initialize trainer.
        
        Args:
            model: LSTM model to train
            device: 'cpu' or 'cuda'
            checkpoint_dir: Directory for saving checkpoints
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for training")
        
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.checkpoint_manager = ModelCheckpoint(checkpoint_dir)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': [],
            'val_auc': []
        }
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 10,
        class_weight: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features (n_samples, seq_len, n_features)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            early_stopping_patience: Epochs to wait before early stopping
            class_weight: Optional weights for imbalanced classes
            
        Returns:
            Training history and final metrics
        """
        logger.info("Starting LSTM training...")
        
        # Create data loaders
        train_loader = self._create_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, batch_size, shuffle=False)
        
        # Loss function with class weights
        if class_weight is None:
            # Auto-compute class weights
            n_samples = len(y_train)
            n_positive = y_train.sum()
            n_negative = n_samples - n_positive
            
            if n_positive > 0 and n_negative > 0:
                pos_weight = n_negative / n_positive
            else:
                pos_weight = 1.0
            
            pos_weight_tensor = torch.tensor([pos_weight], device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCELoss()
        
        # Optimizer and scheduler
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer
            )
            
            # Validate
            val_loss, val_metrics = self._validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            self.checkpoint_manager.save(
                self.model,
                optimizer,
                epoch,
                val_loss,
                val_metrics,
                is_best=is_best
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training completed!")
        
        return {
            'history': self.history,
            'final_metrics': val_metrics,
            'best_val_loss': best_val_loss
        }
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, _ = self.model(batch_X)
            outputs = outputs.squeeze()
            
            # Compute loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(batch_X)
                outputs = outputs.squeeze()
                
                # Compute loss
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # Collect predictions
                probabilities = outputs.cpu().numpy()
                predictions = (outputs > 0.5).float().cpu().numpy()
                labels = batch_y.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                all_labels.extend(labels)
        
        avg_loss = total_loss / len(dataloader)
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1': f1_score(all_labels, all_predictions, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probabilities) if len(np.unique(all_labels)) > 1 else 0.5,
            'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist()
        }
        
        return avg_loss, metrics
    
    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        return dataloader
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Returns:
            Comprehensive evaluation metrics
        """
        test_loader = self._create_dataloader(X_test, y_test, batch_size, shuffle=False)
        criterion = nn.BCELoss()
        
        _, metrics = self._validate_epoch(test_loader, criterion)
        
        logger.info("Test Set Evaluation:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  AUC:       {metrics['auc']:.4f}")
        
        return metrics


# Example usage
if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch required for this example")
        exit(1)
    
    from predictflow.prediction.data_processor import FailureDataProcessor
    from datetime import datetime, timedelta
    
    # Generate synthetic data
    print("Generating synthetic training data...")
    executions = []
    for i in range(200):
        executions.append({
            'timestamp': (datetime.now() - timedelta(hours=200-i)).isoformat(),
            'rpn': np.random.randint(50, 400),
            'confidence': np.random.uniform(0.4, 0.95),
            'context_severity': np.random.uniform(0.2, 0.9),
            'assignee_workload': np.random.uniform(0.1, 0.9),
            'duration': np.random.uniform(20, 400),
            'queue_time': np.random.uniform(0, 100),
            'system_health': np.random.uniform(0.7, 1.0),
            'failed': np.random.random() < 0.2  # 20% failure rate
        })
    
    # Process data
    processor = FailureDataProcessor(sequence_length=10)
    X, y = processor.process_execution_history(executions, "example_step")
    
    # Balance classes
    X, y = processor.balance_classes(X, y, method='oversample')
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_train_test(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = processor.split_train_test(X_train, y_train, test_size=0.2)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create model
    from predictflow.prediction.lstm_predictor import create_default_model
    model = create_default_model()
    
    # Train
    trainer = FailurePredictionTrainer(model)
    results = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=16,
        early_stopping_patience=5
    )
    
    # Evaluate
    test_metrics = trainer.evaluate(X_test, y_test)

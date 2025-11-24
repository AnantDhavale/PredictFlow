"""
LSTM Failure Predictor for PredictFlow
--------------------------------------

Purpose:
  - Time-series prediction of workflow step failures
  - Uses historical execution patterns to predict future failures
  - Integrates with FMEA and context awareness

Features:
  ✅ Bidirectional LSTM for temporal pattern learning
  ✅ Attention mechanism for important time steps
  ✅ Multi-feature input (RPN, confidence, workload, context)
  ✅ Probabilistic failure prediction (0-1)
  ✅ Lightweight for edge deployment
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# PyTorch imports with fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Install with: pip install torch")

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Attention mechanism to focus on important time steps."""
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_size)
        Returns:
            context: (batch_size, hidden_size)
            attention_weights: (batch_size, seq_len)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch, seq_len)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)
        
        # Weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output                       # (batch, seq_len, hidden)
        ).squeeze(1)  # (batch, hidden)
        
        return context, attention_weights


class LSTMFailurePredictor(nn.Module):
    """
    LSTM-based failure predictor for workflow steps.
    
    Architecture:
      - Input: Time series of execution features
      - Bidirectional LSTM layers
      - Attention mechanism
      - Dense layers with dropout
      - Sigmoid output (failure probability)
    """
    
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        """
        Initialize LSTM predictor.
        
        Args:
            input_size: Number of input features per time step
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super(LSTMFailurePredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
        
        # Dense layers
        lstm_output_size = hidden_size * 2  # bidirectional
        
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, 32)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(32, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(32)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            output: Failure probability (batch_size, 1)
            attention_weights: Attention weights if enabled (batch_size, seq_len)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention or use last hidden state
        attention_weights = None
        if self.use_attention:
            context, attention_weights = self.attention(lstm_out)
        else:
            # Use last time step
            context = lstm_out[:, -1, :]  # (batch, hidden*2)
        
        # Dense layers
        out = F.relu(self.fc1(context))
        out = self.bn1(out)
        out = self.dropout1(out)
        
        out = F.relu(self.fc2(out))
        out = self.bn2(out)
        out = self.dropout2(out)
        
        out = torch.sigmoid(self.fc3(out))  # Failure probability
        
        return out, attention_weights
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict failure probability for input sequences.
        
        Args:
            x: Numpy array (batch_size, seq_len, input_size)
            
        Returns:
            Failure probabilities (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            output, _ = self.forward(x_tensor)
            return output.squeeze().numpy()
    
    def get_attention_weights(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input sequence
            
        Returns:
            Attention weights or None if attention not used
        """
        if not self.use_attention:
            return None
        
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            _, attention_weights = self.forward(x_tensor)
            return attention_weights.numpy() if attention_weights is not None else None


class ModelCheckpoint:
    """Simple model checkpoint manager."""
    
    def __init__(self, save_dir: str = "predictflow/prediction/models"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float('inf')
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'model_config': {
                'input_size': model.input_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'use_attention': model.use_attention
            }
        }
        
        # Save latest
        checkpoint_path = self.save_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best
        if is_best or loss < self.best_loss:
            self.best_loss = loss
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    
    @staticmethod
    def load_model(checkpoint_path: str) -> LSTMFailurePredictor:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        config = checkpoint['model_config']
        
        model = LSTMFailurePredictor(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            use_attention=config['use_attention']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Loaded trained model from {checkpoint_path}")
        return model


# Utility functions for model creation
def create_default_model() -> Optional[LSTMFailurePredictor]:
    """Create a default LSTM model with standard configuration."""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available, cannot create model")
        return None
    
    return LSTMFailurePredictor(
        input_size=10,      # RPN, confidence, workload, context features
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        use_attention=True
    )


def load_pretrained_model(model_path: str) -> Optional[LSTMFailurePredictor]:
    """Load a pretrained model from disk."""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available")
        return None
    
    try:
        return ModelCheckpoint.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


# Example usage
if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch required for this example")
        exit(1)
    
    # Create model
    model = create_default_model()
    print(model)
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    input_size = 10
    
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    output, attention = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output.squeeze().detach().numpy()}")
    
    if attention is not None:
        print(f"Attention weights shape: {attention.shape}")

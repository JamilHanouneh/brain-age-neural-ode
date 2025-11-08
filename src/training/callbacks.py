"""
Training callbacks
"""
import torch
from pathlib import Path
import json


class EarlyStoppingCallback:
    """Early stopping based on validation metric"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs with no improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop
        
        Returns:
            True if should stop
        """
        if self.best_value is None:
            self.best_value = value
        elif value < (self.best_value - self.min_delta):
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class ModelCheckpointCallback:
    """Save model checkpoint"""
    
    def __init__(self, checkpoint_dir: str, monitor: str = 'val_loss', mode: str = 'min'):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        
        if mode == 'min':
            self.best_value = float('inf')
        else:
            self.best_value = float('-inf')
    
    def __call__(self, model: torch.nn.Module, value: float, epoch: int) -> bool:
        """
        Check if should save checkpoint
        
        Returns:
            True if saved
        """
        should_save = False
        
        if self.mode == 'min':
            if value < self.best_value:
                self.best_value = value
                should_save = True
        else:
            if value > self.best_value:
                self.best_value = value
                should_save = True
        
        if should_save:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(model.state_dict(), path)
            print(f"✓ Checkpoint saved: {path}")
        
        return should_save


class MetricsLogger:
    """Log metrics to file"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {'train': [], 'val': []}
    
    def log_metrics(self, metrics: dict, split: str = 'train'):
        """Log metrics"""
        self.history[split].append(metrics)
    
    def save(self, filename: str = 'metrics.json'):
        """Save metrics to file"""
        path = self.log_dir / filename
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

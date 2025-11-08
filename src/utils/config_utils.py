"""
Configuration management utilities
"""
import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf


class ConfigManager:
    """Manages configuration loading and device setup"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.device = self.setup_device()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_device(self) -> torch.device:
        """
        Setup compute device based on config and availability
        Supports: CPU, CUDA (NVIDIA GPU), MPS (Apple Silicon)
        """
        device_config = self.config['compute']['device']
        
        if device_config == 'auto':
            # Auto-detect best available device
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
                print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                print("✓ Using Apple Silicon GPU (MPS)")
            else:
                device = torch.device('cpu')
                print("✓ Using CPU (No GPU detected)")
                print("  Note: Training will be slower. Consider using Google Colab for free GPU access.")
        
        elif device_config == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠ CUDA requested but not available. Falling back to CPU.")
                device = torch.device('cpu')
        
        elif device_config == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                print("✓ Using Apple Silicon GPU (MPS)")
            else:
                print("⚠ MPS requested but not available. Falling back to CPU.")
                device = torch.device('cpu')
        
        elif device_config == 'cpu':
            device = torch.device('cpu')
            print("✓ Using CPU (as configured)")
        
        else:
            raise ValueError(f"Unknown device: {device_config}")
        
        return device
    
    def get_num_workers(self) -> int:
        """Get number of workers for data loading"""
        num_workers = self.config['compute']['num_workers']
        
        # Set to 0 on Windows or if using single-threaded
        if os.name == 'nt':  # Windows
            print("⚠ Windows detected: setting num_workers=0 for stability")
            return 0
        
        return num_workers
    
    def setup_reproducibility(self):
        """Setup for reproducible results"""
        seed = self.config['compute']['seed']
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if self.config['compute']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"✓ Reproducibility enabled (seed={seed})")
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("Configuration Summary")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Dataset: {self.config['data']['dataset']}")
        print(f"Model: {self.config['model']['type']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Epochs: {self.config['training']['num_epochs']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        print("="*60 + "\n")


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✓ Configuration saved to {save_path}")

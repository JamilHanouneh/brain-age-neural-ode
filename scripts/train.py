"""
Main training script
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from pathlib import Path

from src.utils.config_utils import ConfigManager
from src.data.dataloader import BrainAgingDataModule
from src.models.neural_ode import NeuralODEFlow
from src.training.trainer import BrainAgingTrainer


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train brain aging model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Brain Aging Neural ODE - Training")
    print("="*60)
    
    # Load configuration
    print("\n✓ Loading configuration...")
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    device = config_manager.device
    
    # Setup reproducibility
    config_manager.setup_reproducibility()
    
    # Print configuration summary
    config_manager.print_config_summary()
    
    # Setup data
    print("\n✓ Setting up data...")
    data_module = BrainAgingDataModule(config)
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Create model
    print("\n✓ Creating model...")
    model_config = config['model']
    
    if model_config['type'] == 'neural_ode_flow':
        model = NeuralODEFlow(
            input_dim=model_config['input_dim'],
            latent_dim=model_config['latent_dim'],
            hidden_dim=model_config['neural_ode']['hidden_dim'],
            num_layers=model_config['neural_ode']['num_layers'],
            activation=model_config['neural_ode']['activation'],
            solver=model_config['neural_ode']['solver'],
            rtol=model_config['neural_ode']['rtol'],
            atol=model_config['neural_ode']['atol'],
            use_adjoint=model_config['neural_ode']['adjoint']
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")
    
    # Create trainer
    print("\n✓ Creating trainer...")
    trainer = BrainAgingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"\n✓ Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    print("\n✓ Training complete!")
    print(f"  Best model saved to: {trainer.checkpoint_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()

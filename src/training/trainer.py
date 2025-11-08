"""
Training loop for brain aging model
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from typing import Dict, Optional
import os
from tqdm import tqdm
import json
from pathlib import Path


class BrainAgingTrainer:
    """Trainer for brain aging model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device
    ):
        """
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Compute device
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training config
        self.train_config = config['training']
        self.num_epochs = self.train_config['num_epochs']
        self.grad_clip = self.train_config['grad_clip']
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        from src.models.losses import BrainAgingLoss
        loss_weights = self.train_config['loss_weights']
        self.criterion = BrainAgingLoss(
            age_weight=loss_weights['age_prediction'],
            recon_weight=loss_weights['reconstruction'],
            prior_weight=loss_weights['latent_prior'],
            diffeomorph_weight=loss_weights['diffeomorphic'],
            uncertainty_weight=loss_weights['uncertainty']
        )
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
        
        # Setup output directories
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.log_dir = Path(config['logging']['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision training
        self.use_amp = config['compute']['use_amp'] and device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✓ Using mixed precision training (AMP)")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        optimizer_name = self.train_config['optimizer'].lower()
        lr = float(self.train_config['learning_rate'])  
        weight_decay = float(self.train_config['weight_decay'])  
        
        if optimizer_name == 'adamw':
            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        print(f"✓ Optimizer: {optimizer_name.upper()}")
        return optimizer
    
    def _setup_scheduler(self) -> Optional[object]:
        """Setup learning rate scheduler"""
        scheduler_name = self.train_config['scheduler'].lower()
        if scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=1e-6 
            )

        if scheduler_name == 'none':
            return None
        elif scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=1e-6
            )
        elif scheduler_name == 'step':
            scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_name == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        print(f"✓ Scheduler: {scheduler_name}")
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_age_loss = 0.0
        total_recon_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs}")
        
        for batch_idx, (x, age_true) in enumerate(pbar):
            # Move to device
            x = x.to(self.device)
            age_true = age_true.to(self.device).unsqueeze(1)
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    age_pred, z, h = self.model(x)
                    x_recon = self.model.inverse(age_pred, z)
                    
                    losses = self.criterion(age_pred, age_true, x_recon, x, z)
                    loss = losses['loss']
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                age_pred, z, h = self.model(x)
                x_recon = self.model.inverse(age_pred, z)
                
                losses = self.criterion(age_pred, age_true, x_recon, x, z)
                loss = losses['loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_age_loss += losses['loss_age'].item()
            total_recon_loss += losses['loss_recon'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'age_loss': f"{losses['loss_age'].item():.4f}"
            })
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_age_loss = total_age_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        
        return {
            'loss': avg_loss,
            'age_loss': avg_age_loss,
            'recon_loss': avg_recon_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        total_age_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, age_true in self.val_loader:
                # Move to device
                x = x.to(self.device)
                age_true = age_true.to(self.device).unsqueeze(1)
                
                # Forward pass
                age_pred, z, h = self.model(x)
                x_recon = self.model.inverse(age_pred, z)
                
                losses = self.criterion(age_pred, age_true, x_recon, x, z)
                
                # Compute MAE
                mae = torch.abs(age_pred - age_true).mean()
                
                # Update metrics
                total_loss += losses['loss'].item()
                total_age_loss += losses['loss_age'].item()
                total_mae += mae.item()
                num_batches += 1
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_age_loss = total_age_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return {
            'loss': avg_loss,
            'age_loss': avg_age_loss,
            'mae': avg_mae
        }
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        early_stopping_patience = self.train_config['early_stopping']['patience']
        no_improve_count = 0
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Age Loss: {train_metrics['age_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f} years")
            
            # Save checkpoint
            if (epoch + 1) % self.train_config['checkpoint']['save_freq'] == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            
            # Save best model
            if val_metrics['mae'] < self.best_val_loss:
                self.best_val_loss = val_metrics['mae']
                self.save_checkpoint("best_model.pt")
                print(f"  ✓ New best model! MAE: {self.best_val_loss:.4f} years")
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early stopping
            if self.train_config['early_stopping']['enabled']:
                if no_improve_count >= early_stopping_patience:
                    print(f"\n✓ Early stopping triggered after {epoch+1} epochs")
                    break
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best validation MAE: {self.best_val_loss:.4f} years")
        print("="*60)
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
    
    def save_history(self):
        """Save training history"""
        history_path = self.log_dir / 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history
            }, f, indent=2)
        
        print(f"✓ Training history saved to {history_path}")

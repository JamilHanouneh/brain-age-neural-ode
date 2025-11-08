"""
Unit tests for training
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from src.models.neural_ode import NeuralODEFlow
from src.models.losses import BrainAgingLoss
from src.training.trainer import BrainAgingTrainer


class TestBrainAgingLoss:
    """Test loss computation"""
    
    @pytest.fixture
    def loss_fn(self):
        return BrainAgingLoss(
            age_weight=1.0,
            recon_weight=0.5,
            prior_weight=0.1
        )
    
    def test_loss_computation(self, loss_fn):
        """Test loss can be computed"""
        batch_size = 4
        input_dim = 100
        
        age_pred = torch.randn(batch_size, 1)
        age_true = torch.randn(batch_size, 1)
        x_recon = torch.randn(batch_size, input_dim)
        x_true = torch.randn(batch_size, input_dim)
        z = torch.randn(batch_size, input_dim - 1)
        
        losses = loss_fn(age_pred, age_true, x_recon, x_true, z)
        
        assert 'loss' in losses
        assert 'loss_age' in losses
        assert 'loss_recon' in losses
        assert not torch.isnan(losses['loss'])
    
    def test_loss_with_uncertainty(self, loss_fn):
        """Test loss with uncertainty"""
        batch_size = 4
        input_dim = 100
        
        age_pred = torch.randn(batch_size, 1)
        age_true = torch.randn(batch_size, 1)
        x_recon = torch.randn(batch_size, input_dim)
        x_true = torch.randn(batch_size, input_dim)
        z = torch.randn(batch_size, input_dim - 1)
        uncertainty = torch.abs(torch.randn(batch_size, 1)) + 0.1
        
        losses = loss_fn(age_pred, age_true, x_recon, x_true, z, uncertainty)
        
        assert 'loss_uncertainty' in losses
        assert losses['loss_uncertainty'] > 0


class TestTrainer:
    """Test training functionality"""
    
    @pytest.fixture
    def config(self):
        return {
            'compute': {
                'device': 'cpu',
                'num_workers': 0,
                'use_amp': False,
                'deterministic': True,
                'seed': 42
            },
            'training': {
                'batch_size': 4,
                'num_epochs': 2,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'optimizer': 'adam',
                'scheduler': 'none',
                'grad_clip': 1.0,
                'loss_weights': {
                    'age_prediction': 1.0,
                    'reconstruction': 0.5,
                    'latent_prior': 0.1,
                    'diffeomorphic': 0.01,
                    'uncertainty': 0.05
                },
                'early_stopping': {'enabled': False},
                'checkpoint': {'save_freq': 1, 'save_best_only': False, 'monitor': 'val_mae'}
            },
            'logging': {
                'checkpoint_dir': 'outputs/models',
                'log_dir': 'outputs/logs'
            },
            'model': {
                'input_dim': 100,
                'latent_dim': 100
            }
        }
    
    @pytest.fixture
    def model(self):
        return NeuralODEFlow(
            input_dim=100,
            latent_dim=100,
            hidden_dim=64,
            num_layers=2
        )
    
    @pytest.fixture
    def dummy_dataloader(self):
        """Create dummy dataloader"""
        from torch.utils.data import DataLoader, TensorDataset
        
        X = torch.randn(16, 100)  # 16 samples, 100 features
        ages = torch.randn(16, 1)
        
        dataset = TensorDataset(X, ages)
        return DataLoader(dataset, batch_size=4, shuffle=True)
    
    def test_trainer_creation(self, config, model, dummy_dataloader):
        """Test trainer can be created"""
        device = torch.device('cpu')
        
        trainer = BrainAgingTrainer(
            model=model,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=config,
            device=device
        )
        
        assert trainer is not None
        assert trainer.num_epochs == 2
    
    def test_training_epoch(self, config, model, dummy_dataloader):
        """Test single training epoch"""
        device = torch.device('cpu')
        
        trainer = BrainAgingTrainer(
            model=model,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=config,
            device=device
        )
        
        metrics = trainer.train_epoch()
        
        assert 'loss' in metrics
        assert metrics['loss'] > 0
        assert not np.isnan(metrics['loss'])
    
    def test_validation(self, config, model, dummy_dataloader):
        """Test validation"""
        device = torch.device('cpu')
        
        trainer = BrainAgingTrainer(
            model=model,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=config,
            device=device
        )
        
        metrics = trainer.validate()
        
        assert 'loss' in metrics
        assert 'mae' in metrics
        assert metrics['loss'] > 0
        assert metrics['mae'] > 0
    
    def test_checkpoint_save_load(self, config, model, dummy_dataloader):
        """Test checkpoint saving and loading"""
        device = torch.device('cpu')
        
        trainer = BrainAgingTrainer(
            model=model,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=config,
            device=device
        )
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'test_checkpoint.pt'
            trainer.checkpoint_dir = Path(tmpdir)
            
            # Save checkpoint
            trainer.save_checkpoint('test_checkpoint.pt')
            assert checkpoint_path.exists()
            
            # Load checkpoint
            trainer.load_checkpoint(str(checkpoint_path))
            assert trainer.current_epoch >= 0


class TestDataLoading:
    """Test data loading functionality"""
    
    def test_brain_aging_dataset(self):
        """Test BrainAgingDataset"""
        from src.data.dataloader import BrainAgingDataset
        import pandas as pd
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data
            data_dir = Path(tmpdir)
            (data_dir / 'processed' / 'train').mkdir(parents=True)
            
            # Create dummy numpy file
            np.save(data_dir / 'processed' / 'train' / 'subject_001.npy',
                   np.random.rand(100).astype(np.float32))
            
            # Create metadata
            metadata_df = pd.DataFrame({
                'subject_id': ['subject_001'],
                'age': [45.0],
                'split': ['train']
            })
            metadata_file = data_dir / 'metadata.csv'
            metadata_df.to_csv(metadata_file, index=False)
            
            # Create dataset
            dataset = BrainAgingDataset(
                data_dir=str(data_dir),
                metadata_file=str(metadata_file),
                split='train'
            )
            
            assert len(dataset) == 1
            
            # Get item
            data, age = dataset[0]
            assert data.shape == (100,)
            assert age.item() == 45.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Age prediction inference
"""
import torch
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class AgePredictor:
    """Predict brain age from preprocessed data"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        uncertainty: bool = False
    ):
        """
        Args:
            model: Trained model
            device: Compute device
            uncertainty: Whether to compute uncertainty
        """
        self.model = model.to(device)
        self.device = device
        self.uncertainty = uncertainty
        self.model.eval()
    
    def predict(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> dict:
        """
        Predict brain age
        
        Args:
            x: Input features [batch_size, input_dim]
            return_latent: Return latent representations
        
        Returns:
            Dictionary with predictions and optional uncertainty
        """
        with torch.no_grad():
            x = x.to(self.device)
            
            if self.uncertainty:
                # MC Dropout: multiple forward passes
                predictions = []
                for _ in range(20):
                    age_pred, z, h = self.model(x)
                    predictions.append(age_pred.cpu().numpy())
                
                predictions = np.array(predictions)  # [num_samples, batch, 1]
                mean_age = predictions.mean(axis=0)
                std_age = predictions.std(axis=0)
                
                result = {
                    'age': mean_age,
                    'uncertainty': std_age
                }
            else:
                age_pred, z, h = self.model(x)
                result = {
                    'age': age_pred.cpu().numpy()
                }
            
            if return_latent:
                result['latent'] = h.cpu().numpy()
        
        return result
    
    def predict_batch(
        self,
        data_loader,
        return_all: bool = True
    ) -> dict:
        """
        Predict on entire dataloader
        
        Args:
            data_loader: PyTorch DataLoader
            return_all: Return all predictions and uncertainties
        
        Returns:
            Dictionary with predictions and optionally uncertainties
        """
        all_ages = []
        all_uncertainties = [] if self.uncertainty else None
        
        for x, _ in data_loader:
            results = self.predict(x, return_latent=False)
            
            all_ages.append(results['age'])
            if self.uncertainty:
                all_uncertainties.append(results['uncertainty'])
        
        all_ages = np.concatenate(all_ages, axis=0).squeeze()
        
        result = {'ages': all_ages}
        if self.uncertainty:
            result['uncertainties'] = np.concatenate(all_uncertainties, axis=0).squeeze()
        
        return result
    
    def save_predictions(
        self,
        predictions: dict,
        output_path: str
    ):
        """Save predictions to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            str(output_path),
            ages=predictions['ages'],
            uncertainties=predictions.get('uncertainties', None)
        )
        
        print(f"✓ Predictions saved to {output_path}")

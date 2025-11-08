"""
Uncertainty quantification
"""
import numpy as np
import torch
from typing import Tuple


class UncertaintyEstimator:
    """Estimate prediction uncertainty"""
    
    @staticmethod
    def mc_dropout_uncertainty(
        model: torch.nn.Module,
        x: torch.Tensor,
        num_samples: int = 20,
        device: torch.device = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate uncertainty using MC Dropout
        
        Args:
            model: Model with dropout
            x: Input tensor
            num_samples: Number of MC samples
            device: Compute device
        
        Returns:
            mean predictions and uncertainties
        """
        predictions = []
        
        model.train()  # Keep dropout active
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred, _, _ = model(x)
                predictions.append(pred.cpu().numpy())
        
        model.eval()
        
        predictions = np.array(predictions)  # [num_samples, batch, 1]
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std
    
    @staticmethod
    def ensemble_uncertainty(
        models: list,
        x: torch.Tensor,
        device: torch.device = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate uncertainty using ensemble
        
        Args:
            models: List of models
            x: Input tensor
            device: Compute device
        
        Returns:
            mean predictions and uncertainties
        """
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                pred, _, _ = model(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)  # [num_models, batch, 1]
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std
    
    @staticmethod
    def calibration_error(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        true_values: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Compute calibration error
        
        Args:
            predictions: Predicted values
            uncertainties: Estimated uncertainties
            true_values: Ground truth values
            num_bins: Number of bins for calibration
        
        Returns:
            Calibration error
        """
        # Sort by uncertainty
        sorted_idx = np.argsort(uncertainties.flatten())
        
        # Divide into bins
        bin_size = len(sorted_idx) // num_bins
        
        calibration_errors = []
        
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else len(sorted_idx)
            
            bin_idx = sorted_idx[start_idx:end_idx]
            
            bin_pred = predictions[bin_idx]
            bin_true = true_values[bin_idx]
            bin_unc = uncertainties[bin_idx]
            
            # Empirical error in this bin
            empirical_error = np.mean((bin_pred - bin_true) ** 2)
            
            # Expected error based on uncertainty
            expected_error = np.mean(bin_unc ** 2)
            
            calibration_errors.append((empirical_error - expected_error) ** 2)
        
        return np.mean(calibration_errors)
    
    @staticmethod
    def confidence_intervals(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence intervals
        
        Args:
            predictions: Predicted values
            uncertainties: Estimated uncertainties
            confidence: Confidence level (0-1)
        
        Returns:
            lower and upper bounds
        """
        # Assume Gaussian distribution
        z_score = 1.96 if confidence == 0.95 else 2.576  # for 99%
        
        margin = z_score * uncertainties
        
        lower = predictions - margin
        upper = predictions + margin
        
        return lower, upper

"""
Custom loss functions for brain aging model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional  


class BrainAgingLoss(nn.Module):
    """Combined loss for brain aging model"""
    
    def __init__(
        self,
        age_weight: float = 1.0,
        recon_weight: float = 0.5,
        prior_weight: float = 0.1,
        diffeomorph_weight: float = 0.01,
        uncertainty_weight: float = 0.05
    ):
        """
        Args:
            age_weight: Weight for age prediction loss
            recon_weight: Weight for reconstruction loss
            prior_weight: Weight for latent prior loss
            diffeomorph_weight: Weight for diffeomorphic regularization
            uncertainty_weight: Weight for uncertainty loss
        """
        super().__init__()
        
        self.age_weight = age_weight
        self.recon_weight = recon_weight
        self.prior_weight = prior_weight
        self.diffeomorph_weight = diffeomorph_weight
        self.uncertainty_weight = uncertainty_weight
    
    def forward(
        self,
        age_pred: torch.Tensor,
        age_true: torch.Tensor,
        x_recon: torch.Tensor,
        x_true: torch.Tensor,
        z: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute combined loss
        
        Args:
            age_pred: Predicted ages [batch_size, 1]
            age_true: True ages [batch_size, 1]
            x_recon: Reconstructed input [batch_size, input_dim]
            x_true: True input [batch_size, input_dim]
            z: Latent variability [batch_size, latent_dim-1]
            uncertainty: Predicted uncertainty [batch_size, 1]
        
        Returns:
            Dictionary of losses
        """
        # Age prediction loss (MSE)
        loss_age = F.mse_loss(age_pred, age_true)
        
        # Reconstruction loss (MSE)
        loss_recon = F.mse_loss(x_recon, x_true)
        
        # Latent prior loss (encourage Gaussian distribution)
        loss_prior = torch.mean(z ** 2)
        
        # Diffeomorphic regularization (smoothness)
        # This is a placeholder - full implementation would use velocity field gradients
        loss_diffeomorph = torch.tensor(0.0, device=z.device)
        
        # Uncertainty loss (if using uncertainty quantification)
        loss_uncertainty = torch.tensor(0.0, device=z.device)
        if uncertainty is not None:
            # Negative log likelihood with learned uncertainty
            # loss = 0.5 * (error^2 / sigma^2 + log(sigma^2))
            error = (age_pred - age_true) ** 2
            loss_uncertainty = 0.5 * (error / (uncertainty ** 2) + torch.log(uncertainty ** 2))
            loss_uncertainty = torch.mean(loss_uncertainty)
        
        # Combined loss
        total_loss = (
            self.age_weight * loss_age +
            self.recon_weight * loss_recon +
            self.prior_weight * loss_prior +
            self.diffeomorph_weight * loss_diffeomorph +
            self.uncertainty_weight * loss_uncertainty
        )
        
        return {
            'loss': total_loss,
            'loss_age': loss_age,
            'loss_recon': loss_recon,
            'loss_prior': loss_prior,
            'loss_diffeomorph': loss_diffeomorph,
            'loss_uncertainty': loss_uncertainty
        }


class AgePredictionLoss(nn.Module):
    """Simple age prediction loss"""
    
    def __init__(self, loss_type: str = 'mse'):
        """
        Args:
            loss_type: Type of loss ('mse', 'mae', 'huber')
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        age_pred: torch.Tensor,
        age_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute age prediction loss"""
        if self.loss_type == 'mse':
            return F.mse_loss(age_pred, age_true)
        elif self.loss_type == 'mae':
            return F.l1_loss(age_pred, age_true)
        elif self.loss_type == 'huber':
            return F.huber_loss(age_pred, age_true)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

"""
Base normalizing flow implementation
"""
import torch
import torch.nn as nn
from typing import Tuple


class CouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flows"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = 'tanh'
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension of transform networks
            activation: Activation function
        """
        super().__init__()
        
        half_dim = input_dim // 2
        
        # Scale and translation networks
        if activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.scale_net = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, half_dim)
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, half_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Jacobian determinant
        
        Returns:
            x_transformed: Transformed input
            log_det_J: Log determinant of Jacobian
        """
        x1, x2 = x.chunk(2, dim=1)
        
        # Transform first half
        scale = self.scale_net(x2)
        translate = self.translate_net(x2)
        
        x1_transformed = x1 * torch.exp(scale) + translate
        x_transformed = torch.cat([x1_transformed, x2], dim=1)
        
        # Jacobian determinant
        log_det_J = scale.sum(dim=1)
        
        return x_transformed, log_det_J
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse transformation"""
        x1, x2 = x.chunk(2, dim=1)
        
        scale = self.scale_net(x2)
        translate = self.translate_net(x2)
        
        x1_original = (x1 - translate) * torch.exp(-scale)
        x_original = torch.cat([x1_original, x2], dim=1)
        
        # Jacobian determinant (negative for inverse)
        log_det_J = -scale.sum(dim=1)
        
        return x_original, log_det_J


class NormalizingFlow(nn.Module):
    """Normalizing flow with coupling layers"""
    
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 16,
        hidden_dim: int = 512
    ):
        """
        Args:
            input_dim: Input dimension
            num_layers: Number of coupling layers
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            CouplingLayer(input_dim, hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all layers"""
        log_det_J = torch.zeros(x.size(0), device=x.device)
        
        for layer in self.layers:
            x, ldj = layer(x)
            log_det_J += ldj
        
        return x, log_det_J
    
    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse through layers"""
        log_det_J = torch.zeros(z.size(0), device=z.device)
        
        for layer in reversed(self.layers):
            z, ldj = layer.inverse(z)
            log_det_J += ldj
        
        return z, log_det_J

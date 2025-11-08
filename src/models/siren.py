"""
Sinusoidal Representation Networks (SIREN)
"""
import torch
import torch.nn as nn
import numpy as np


class SineLayer(nn.Module):
    """Layer with sine activation"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            omega_0: Frequency of sine function
            is_first: Whether this is the first layer
        """
        super().__init__()
        
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights according to SIREN paper"""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/n, 1/n]
                self.linear.weight.uniform_(
                    -1 / self.linear.in_features,
                    1 / self.linear.in_features
                )
            else:
                # Hidden layers: uniform in [-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0]
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """Sinusoidal Representation Network"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        omega_0: float = 30.0,
        final_activation: bool = True
    ):
        """
        Args:
            in_features: Input dimension (e.g., spatial coords + age)
            hidden_features: Hidden layer dimension
            hidden_layers: Number of hidden layers
            out_features: Output dimension (velocity field components)
            omega_0: Sine frequency
            final_activation: Use sine activation on output
        """
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(SineLayer(
            in_features, hidden_features,
            omega_0=omega_0, is_first=True
        ))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(
                hidden_features, hidden_features,
                omega_0=omega_0, is_first=False
            ))
        
        # Output layer
        final_layer = nn.Linear(hidden_features, out_features)
        
        # Initialize output layer
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / omega_0
            final_layer.weight.uniform_(-bound, bound)
        
        if final_activation:
            layers.append(SineLayer(
                hidden_features, out_features,
                omega_0=omega_0, is_first=False
            ))
        else:
            layers.append(final_layer)
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Input coordinates [batch_size, in_features]
        
        Returns:
            Output [batch_size, out_features]
        """
        return self.net(coords)

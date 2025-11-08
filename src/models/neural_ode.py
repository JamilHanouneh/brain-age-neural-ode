"""
Neural Ordinary Differential Equation implementation
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from typing import Optional, Callable


class ODEFunc(nn.Module):
    """ODE function: dh/dt = f(h, t)"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 3,
        activation: str = 'tanh'
    ):
        """
        Args:
            hidden_dim: Dimension of hidden state
            num_layers: Number of layers in the network
            activation: Activation function ('tanh', 'relu', 'swish')
        """
        super().__init__()
        
        # Select activation function
        if activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'swish':
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    act_fn
                ])
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    act_fn
                ])
        
        self.net = nn.Sequential(*layers)
        
        # Initialize with small weights for stability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt
        
        Args:
            t: Time (not used in autonomous ODE, but required by torchdiffeq)
            h: Hidden state [batch_size, hidden_dim]
        
        Returns:
            dh/dt: Time derivative [batch_size, hidden_dim]
        """
        return self.net(h)


class NeuralODEFlow(nn.Module):
    """Neural ODE-based normalizing flow for brain aging"""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        activation: str = 'tanh',
        solver: str = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4,
        use_adjoint: bool = True
    ):
        """
        Args:
            input_dim: Dimension of input (PCA-reduced brain features)
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for ODE function
            num_layers: Number of layers in ODE function
            activation: Activation function
            solver: ODE solver ('dopri5', 'adams', 'euler')
            rtol: Relative tolerance for solver
            atol: Absolute tolerance for solver
            use_adjoint: Use adjoint method (memory efficient)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint
        
        # Encoder: maps input to initial hidden state
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # ODE function
        self.ode_func = ODEFunc(latent_dim, num_layers, activation)
        
        # Decoder: maps final hidden state back to input space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Integration time span
        self.register_buffer('t_span', torch.tensor([0.0, 1.0]))
    
    def forward(
        self,
        x: torch.Tensor,
        return_trajectory: bool = False
    ) -> tuple:
        """
        Forward pass: x -> [age, z]
        
        Args:
            x: Input brain features [batch_size, input_dim]
            return_trajectory: Return full ODE trajectory
        
        Returns:
            age: Predicted age [batch_size, 1]
            z: Latent variability [batch_size, latent_dim-1]
            h: Full latent representation [batch_size, latent_dim]
        """
        batch_size = x.size(0)
        
        # Encode to initial hidden state
        h0 = self.encoder(x)
        
        # Solve ODE
        if self.use_adjoint:
            ode_int = odeint_adjoint
        else:
            ode_int = odeint
        
        h_trajectory = ode_int(
            self.ode_func,
            h0,
            self.t_span,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver
        )
        
        # Get final state
        h1 = h_trajectory[-1]  # [batch_size, latent_dim]
        
        # Extract age (first dimension) and variability (rest)
        age = h1[:, :1]  # [batch_size, 1]
        z = h1[:, 1:]    # [batch_size, latent_dim-1]
        
        if return_trajectory:
            return age, z, h1, h_trajectory
        else:
            return age, z, h1
    
    def inverse(
        self,
        age: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Inverse pass: [age, z] -> x
        
        Args:
            age: Age values [batch_size, 1]
            z: Latent variability [batch_size, latent_dim-1]
        
        Returns:
            x_recon: Reconstructed input [batch_size, input_dim]
        """
        # Concatenate age and variability
        h1 = torch.cat([age, z], dim=1)
        
        # Solve ODE backwards
        t_span_inv = torch.flip(self.t_span, dims=[0])
        
        if self.use_adjoint:
            ode_int = odeint_adjoint
        else:
            ode_int = odeint
        
        h_trajectory = ode_int(
            self.ode_func,
            h1,
            t_span_inv,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver
        )
        
        # Get initial state
        h0 = h_trajectory[-1]
        
        # Decode to input space
        x_recon = self.decoder(h0)
        
        return x_recon
    
    def compute_jacobian_trace(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute trace of Jacobian for likelihood calculation
        
        Args:
            h: Hidden state [batch_size, latent_dim]
        
        Returns:
            trace: Trace of Jacobian [batch_size]
        """
        # Use Hutchinson's trace estimator
        batch_size = h.size(0)
        
        # Sample random vector
        epsilon = torch.randn_like(h)
        
        # Compute Jacobian-vector product
        h_copy = h.detach().requires_grad_(True)
        dh_dt = self.ode_func(torch.zeros(1), h_copy)
        
        # Compute trace
        jvp = torch.autograd.grad(
            dh_dt, h_copy,
            grad_outputs=epsilon,
            create_graph=True
        )[0]
        
        trace = (jvp * epsilon).sum(dim=1)
        
        return trace

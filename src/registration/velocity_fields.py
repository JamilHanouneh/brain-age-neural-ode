"""
Velocity field operations for diffeomorphic modeling
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class VelocityFieldProcessor:
    """Process and manipulate velocity fields"""
    
    @staticmethod
    def compute_jacobian(v: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian of velocity field
        
        Args:
            v: Velocity field [batch, 3, depth, height, width]
        
        Returns:
            Jacobian [batch, 3, 3, depth, height, width]
        """
        batch_size = v.size(0)
        spatial_size = v.shape[2:]
        
        # Compute spatial gradients
        # dv_i/dx_j for each component
        jacobians = []
        
        for i in range(3):  # For each velocity component
            component = v[:, i:i+1, :, :, :]  # [batch, 1, d, h, w]
            
            # Compute gradients using finite differences
            # Gradient in x direction
            grad_x = component[:, :, :, :, 1:] - component[:, :, :, :, :-1]
            
            jacobians.append(grad_x)
        
        return torch.stack(jacobians, dim=1)
    
    @staticmethod
    def compute_divergence(v: torch.Tensor) -> torch.Tensor:
        """
        Compute divergence of velocity field
        
        Args:
            v: Velocity field [batch, 3, depth, height, width]
        
        Returns:
            Divergence [batch, 1, depth, height, width]
        """
        # Compute spatial gradients for each component
        # div(v) = dv_x/dx + dv_y/dy + dv_z/dz
        
        div = torch.zeros(v.size(0), 1, *v.shape[2:], device=v.device)
        
        for i in range(3):
            component = v[:, i:i+1]
            
            # Gradient in i-th direction
            if i == 0:  # x
                grad = component[:, :, :, :, 1:] - component[:, :, :, :, :-1]
                div[:, :, :, :, :-1] += grad
            elif i == 1:  # y
                grad = component[:, :, :, 1:, :] - component[:, :, :, :-1, :]
                div[:, :, :, :-1, :] += grad
            else:  # z
                grad = component[:, :, 1:, :, :] - component[:, :, :-1, :, :]
                div[:, :, :-1, :, :] += grad
        
        return div
    
    @staticmethod
    def regularize_velocity_field(
        v: torch.Tensor,
        lambda_reg: float = 0.01
    ) -> torch.Tensor:
        """
        Regularize velocity field using smoothness constraint
        
        Args:
            v: Velocity field
            lambda_reg: Regularization strength
        
        Returns:
            Regularization loss
        """
        # Compute spatial gradients (smoothness)
        grad_loss = 0.0
        
        for i in range(3):  # For each dimension
            component = v[:, i:i+1]
            
            # Laplacian (smooth through Gaussian blur)
            blurred = F.avg_pool3d(
                F.pad(component, (1, 1, 1, 1, 1, 1), mode='reflect'),
                kernel_size=3,
                stride=1,
                padding=0
            )
            
            grad_loss += torch.sum((component - blurred) ** 2)
        
        return lambda_reg * grad_loss
    
    @staticmethod
    def project_to_pca_space(
        v: np.ndarray,
        pca_model
    ) -> np.ndarray:
        """
        Project velocity field to PCA space
        
        Args:
            v: Velocity field (flattened)
            pca_model: Fitted PCA model
        
        Returns:
            PCA-projected coefficients
        """
        v_flat = v.flatten() if len(v.shape) > 1 else v
        return pca_model.transform(v_flat.reshape(1, -1))[0]
    
    @staticmethod
    def reconstruct_from_pca(
        coeffs: np.ndarray,
        pca_model,
        shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Reconstruct velocity field from PCA coefficients
        
        Args:
            coeffs: PCA coefficients
            pca_model: Fitted PCA model
            shape: Original shape
        
        Returns:
            Reconstructed velocity field
        """
        v_reconstructed = pca_model.inverse_transform(
            coeffs.reshape(1, -1)
        )[0]
        
        return v_reconstructed.reshape(shape)

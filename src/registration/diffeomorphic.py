"""
Diffeomorphic registration utilities
"""
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from typing import Tuple


class VelocityFieldExponential(nn.Module):
    """Exponential map for velocity fields"""
    
    def __init__(self, num_steps: int = 7):
        """
        Args:
            num_steps: Number of scaling and squaring steps
        """
        super().__init__()
        self.num_steps = num_steps
    
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute exponential of velocity field using scaling and squaring
        
        Args:
            v: Velocity field [batch, *spatial, 3]
        
        Returns:
            phi: Deformation field (displacement)
        """
        # Simplified implementation - full version would use numerical integration
        phi = v / (2 ** self.num_steps)
        
        for _ in range(self.num_steps):
            # Compose twice
            phi = phi + phi  # Simplified - actual implementation uses interpolation
        
        return phi


class DiffeomorphicRegistration:
    """Diffeomorphic image registration"""
    
    @staticmethod
    def estimate_displacement_field(
        moving: np.ndarray,
        fixed: np.ndarray,
        regularization: float = 1.0
    ) -> np.ndarray:
        """
        Estimate displacement field between images
        
        Args:
            moving: Moving image
            fixed: Fixed image
            regularization: Regularization strength
        
        Returns:
            Displacement field
        """
        # Convert to SimpleITK
        moving_itk = sitk.GetImageFromArray(moving.astype(np.float32))
        fixed_itk = sitk.GetImageFromArray(fixed.astype(np.float32))
        
        # Setup registration
        registration_method = sitk.ImageRegistrationMethod()
        
        # Similarity metric
        registration_method.SetMetricAsCorrelation()
        
        # Optimizer
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=2.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        
        # Transform: B-spline
        transform = sitk.BSplineTransformInitializer(
            fixed_itk, [4, 4, 4]
        )
        registration_method.SetInitialTransform(transform, inPlace=True)
        
        # Execute
        final_transform = registration_method.Execute(fixed_itk, moving_itk)
        
        # Get displacement field
        displacement_field = sitk.GetArrayFromImage(
            final_transform.GetDisplacementField()
        )
        
        return displacement_field
    
    @staticmethod
    def compose_displacements(
        d1: np.ndarray,
        d2: np.ndarray
    ) -> np.ndarray:
        """Compose two displacement fields"""
        # Simplified composition - would use proper interpolation
        return d1 + d2

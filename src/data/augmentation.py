"""
Data augmentation for brain MRI
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class BrainAugmentation:
    """Data augmentation for 3D brain images"""
    
    def __init__(
        self,
        rotation_range: float = 10.0,
        scaling_range: float = 0.1,
        noise_std: float = 0.05,
        probability: float = 0.5
    ):
        """
        Args:
            rotation_range: Maximum rotation in degrees
            scaling_range: Maximum scaling factor deviation
            noise_std: Standard deviation of Gaussian noise
            probability: Probability of applying augmentation
        """
        self.rotation_range = rotation_range
        self.scaling_range = scaling_range
        self.noise_std = noise_std
        self.probability = probability
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to image"""
        if np.random.rand() > self.probability:
            return image
        
        # Random rotation
        if np.random.rand() < 0.5:
            image = self._rotate(image)
        
        # Random scaling
        if np.random.rand() < 0.5:
            image = self._scale(image)
        
        # Random noise
        if np.random.rand() < 0.5:
            image = self._add_noise(image)
        
        return image
    
    def _rotate(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random rotation"""
        # Generate random rotation angles (in degrees)
        angle_x = np.random.uniform(-self.rotation_range, self.rotation_range)
        angle_y = np.random.uniform(-self.rotation_range, self.rotation_range)
        angle_z = np.random.uniform(-self.rotation_range, self.rotation_range)
        
        # Convert to radians
        angle_x = np.deg2rad(angle_x)
        angle_y = np.deg2rad(angle_y)
        angle_z = np.deg2rad(angle_z)
        
        # Create rotation matrices
        # For simplicity, we'll use z-axis rotation only
        # Full 3D rotation can be added if needed
        cos_z = np.cos(angle_z)
        sin_z = np.sin(angle_z)
        
        # Create affine grid
        theta = torch.tensor([
            [cos_z, -sin_z, 0, 0],
            [sin_z, cos_z, 0, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        # Add batch and channel dimensions if needed
        if image.dim() == 3:
            image = image.unsqueeze(0).unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Apply affine transformation
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        rotated = F.grid_sample(image, grid, align_corners=False)
        
        if squeeze_output:
            rotated = rotated.squeeze(0).squeeze(0)
        
        return rotated
    
    def _scale(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random scaling"""
        scale_factor = 1.0 + np.random.uniform(-self.scaling_range, self.scaling_range)
        
        # Add batch and channel dimensions if needed
        if image.dim() == 3:
            image = image.unsqueeze(0).unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Apply scaling via interpolation
        scaled = F.interpolate(
            image,
            scale_factor=scale_factor,
            mode='trilinear',
            align_corners=False
        )
        
        # Crop or pad to original size
        original_size = image.size()[2:]
        scaled_size = scaled.size()[2:]
        
        # Calculate padding or cropping
        if scale_factor > 1.0:
            # Crop center
            start = [(s - o) // 2 for s, o in zip(scaled_size, original_size)]
            scaled = scaled[
                :, :,
                start[0]:start[0]+original_size[0],
                start[1]:start[1]+original_size[1],
                start[2]:start[2]+original_size[2]
            ]
        else:
            # Pad
            pad = [(o - s) // 2 for o, s in zip(original_size, scaled_size)]
            pad = [p for p_pair in pad for p in (p_pair, p_pair)][::-1]
            scaled = F.pad(scaled, pad)
        
        if squeeze_output:
            scaled = scaled.squeeze(0).squeeze(0)
        
        return scaled
    
    def _add_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise"""
        noise = torch.randn_like(image) * self.noise_std
        noisy = image + noise
        return noisy

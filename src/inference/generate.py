"""
Generate age-conditioned templates
"""
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Dict, Optional


class TemplateGenerator:
    """Generate age-conditioned brain templates"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        pca_model: Optional = None
    ):
        """
        Args:
            model: Trained model
            device: Compute device
            pca_model: PCA model for reconstruction (optional)
        """
        self.model = model.to(device)
        self.device = device
        self.pca_model = pca_model
        self.model.eval()
    
    def generate_templates(
        self,
        ages: List[int],
        num_samples: int = 10000,
        return_samples: bool = False
    ) -> Dict:
        """
        Generate templates for given ages
        
        Args:
            ages: List of ages
            num_samples: Number of samples for Monte Carlo estimation
            return_samples: Return individual samples or just means
        
        Returns:
            Dictionary with templates and statistics
        """
        templates = {}
        
        print(f"✓ Generating templates for ages: {ages}")
        
        for age in ages:
            print(f"  Generating template for age {age}...")
            
            # Create age tensor
            age_tensor = torch.ones(num_samples, 1).to(self.device) * age
            
            # Sample variability from prior
            latent_dim = self.model.latent_dim
            z = torch.randn(num_samples, latent_dim - 1).to(self.device)
            
            # Generate samples
            with torch.no_grad():
                velocity_fields = self.model.inverse(age_tensor, z)
            
            velocity_fields = velocity_fields.cpu().numpy()
            
            # Compute statistics
            template_mean = velocity_fields.mean(axis=0)
            template_std = velocity_fields.std(axis=0)
            
            templates[age] = {
                'mean': template_mean,
                'std': template_std,
                'samples': velocity_fields if return_samples else None
            }
        
        return templates
    
    def save_templates(
        self,
        templates: Dict,
        output_dir: str,
        nifti_format: bool = True
    ):
        """
        Save templates to files
        
        Args:
            templates: Dictionary of templates
            output_dir: Output directory
            nifti_format: Save as NIFTI files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n✓ Saving templates to {output_dir}")
        
        for age, template_data in templates.items():
            template = template_data['mean']
            
            if nifti_format:
                # Save as NIFTI
                img = nib.Nifti1Image(template.astype(np.float32), 
                                      affine=np.eye(4))
                filepath = output_path / f'template_age_{age}.nii.gz'
                nib.save(img, filepath)
                print(f"  Saved: {filepath}")
            else:
                # Save as numpy
                filepath = output_path / f'template_age_{age}.npy'
                np.save(filepath, template)
                print(f"  Saved: {filepath}")
    
    def generate_disease_trajectory(
        self,
        healthy_age: float,
        disease_severity: List[float],
        num_samples: int = 1000
    ) -> Dict:
        """
        Generate disease progression trajectory
        
        Args:
            healthy_age: Age of healthy template
            disease_severity: List of disease severity levels [0, 1]
            num_samples: Samples per severity level
        
        Returns:
            Dictionary with disease trajectories
        """
        trajectories = {}
        
        for severity in disease_severity:
            # Generate healthy template
            healthy_age_t = torch.ones(num_samples, 1).to(self.device) * healthy_age
            z_healthy = torch.randn(num_samples, self.model.latent_dim - 1).to(self.device)
            
            with torch.no_grad():
                healthy_brain = self.model.inverse(healthy_age_t, z_healthy)
            
            # Add disease component (simplified)
            # In full implementation, would have separate pathology model
            disease_brain = healthy_brain * (1.0 - severity * 0.1)  # Simple atrophy model
            
            trajectories[severity] = {
                'mean': disease_brain.mean(axis=0).cpu().numpy(),
                'std': disease_brain.std(axis=0).cpu().numpy()
            }
        
        return trajectories

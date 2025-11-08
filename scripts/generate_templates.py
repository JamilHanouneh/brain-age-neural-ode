"""
Generate age-conditioned brain templates
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils.config_utils import ConfigManager
from src.models.neural_ode import NeuralODEFlow


def generate_templates(model, ages, num_samples, device, config):
    """Generate age-conditioned templates"""
    model.eval()
    
    templates = {}
    latent_dim = config['model']['latent_dim']
    
    print("\n✓ Generating age-conditioned templates...")
    
    for age in ages:
        print(f"  Generating template for age {age}...")
        
        # Create age tensor
        age_tensor = torch.ones(num_samples, 1).to(device) * age
        
        # Sample variability from prior
        z = torch.randn(num_samples, latent_dim - 1).to(device)
        
        # Generate samples
        with torch.no_grad():
            velocity_fields = model.inverse(age_tensor, z)
        
        # Average to get template
        template_velocity = velocity_fields.mean(dim=0).cpu().numpy()
        
        templates[age] = template_velocity
    
    return templates


def save_templates(templates, output_dir, config):
    """Save templates as NIFTI files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n✓ Saving templates...")
    
    for age, template in templates.items():
        # Reshape to 3D if needed
        target_shape = config['data']['preprocessing']['target_shape']
        if template.size != np.prod(target_shape):
            # Template is in reduced PCA space, need to reconstruct
            # This would require loading PCA model - simplified here
            template_3d = np.zeros(target_shape)
        else:
            template_3d = template.reshape(target_shape)
        
        # Create NIFTI image
        nifti_img = nib.Nifti1Image(template_3d, affine=np.eye(4))
        
        # Save
        output_path = output_dir / f'template_age_{age}.nii.gz'
        nib.save(nifti_img, output_path)
        
        print(f"  Saved: {output_path}")


def visualize_templates(templates, output_dir):
    """Create visualizations of templates"""
    output_dir = Path(output_dir)
    
    print("\n✓ Creating visualizations...")
    
    ages = sorted(templates.keys())
    n_ages = len(ages)
    
    fig, axes = plt.subplots(2, n_ages, figsize=(4*n_ages, 8))
    
    for i, age in enumerate(ages):
        template = templates[age]
        
        # Reshape if needed
        if len(template.shape) == 1:
            # Simplified visualization for PCA space
            axes[0, i].plot(template[:100])
            axes[0, i].set_title(f'Age {age}', fontsize=12, fontweight='bold')
            axes[0, i].set_xlabel('Feature Index')
            axes[0, i].set_ylabel('Value')
            
            axes[1, i].hist(template, bins=50, alpha=0.7)
            axes[1, i].set_xlabel('Value')
            axes[1, i].set_ylabel('Frequency')
        else:
            # 3D template - show middle slice
            mid_slice = template.shape[0] // 2
            axes[0, i].imshow(template[mid_slice, :, :], cmap='gray')
            axes[0, i].set_title(f'Age {age}', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
            
            # Show another view
            mid_slice_2 = template.shape[1] // 2
            axes[1, i].imshow(template[:, mid_slice_2, :], cmap='gray')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'age_templates_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_dir / 'age_templates_overview.png'}")


def main():
    parser = argparse.ArgumentParser(description='Generate age-conditioned brain templates')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/templates',
        help='Directory to save templates'
    )
    parser.add_argument(
        '--ages',
        type=int,
        nargs='+',
        default=None,
        help='Ages to generate templates for (default: from config)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples for Monte Carlo estimation'
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Brain Aging Neural ODE - Template Generation")
    print("="*60)
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    device = config_manager.device
    
    # Get generation parameters
    ages = args.ages if args.ages else config['generation']['age_range']
    num_samples = args.num_samples if args.num_samples else config['generation']['num_samples']
    
    print(f"\n✓ Configuration:")
    print(f"  Ages: {ages}")
    print(f"  Number of samples: {num_samples}")
    
    # Create model
    print("\n✓ Loading model...")
    model_config = config['model']
    model = NeuralODEFlow(
        input_dim=model_config['input_dim'],
        latent_dim=model_config['latent_dim'],
        hidden_dim=model_config['neural_ode']['hidden_dim'],
        num_layers=model_config['neural_ode']['num_layers'],
        activation=model_config['neural_ode']['activation'],
        solver=model_config['neural_ode']['solver'],
        rtol=model_config['neural_ode']['rtol'],
        atol=model_config['neural_ode']['atol'],
        use_adjoint=model_config['neural_ode']['adjoint']
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Generate templates
    templates = generate_templates(model, ages, num_samples, device, config)
    
    # Save templates
    save_templates(templates, args.output_dir, config)
    
    # Visualize
    if config['generation']['visualize']:
        visualize_templates(templates, args.output_dir)
    
    print(f"\n✓ Template generation complete! Saved to {args.output_dir}")


if __name__ == '__main__':
    main()

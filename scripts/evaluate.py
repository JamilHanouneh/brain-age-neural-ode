"""
Model evaluation script
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config_utils import ConfigManager
from src.data.dataloader import BrainAgingDataModule
from src.models.neural_ode import NeuralODEFlow


def evaluate_model(model, test_loader, device, config):
    """Evaluate model on test set"""
    model.eval()
    
    all_predictions = []
    all_true_ages = []
    all_uncertainties = []
    
    print("\n✓ Evaluating model...")
    
    with torch.no_grad():
        for x, age_true in test_loader:
            x = x.to(device)
            age_true = age_true.to(device)
            
            # Predict
            age_pred, z, h = model(x)
            
            all_predictions.extend(age_pred.cpu().numpy().flatten())
            all_true_ages.extend(age_true.cpu().numpy().flatten())
    
    all_predictions = np.array(all_predictions)
    all_true_ages = np.array(all_true_ages)
    
    # Compute metrics
    mae = mean_absolute_error(all_true_ages, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_true_ages, all_predictions))
    r2 = r2_score(all_true_ages, all_predictions)
    pearson_r, pearson_p = pearsonr(all_true_ages, all_predictions)
    spearman_r, spearman_p = spearmanr(all_true_ages, all_predictions)
    
    # Brain age gap (BAG)
    brain_age_gap = all_predictions - all_true_ages
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mean_bag': np.mean(brain_age_gap),
        'std_bag': np.std(brain_age_gap)
    }
    
    return metrics, all_predictions, all_true_ages, brain_age_gap


def evaluate_by_age_group(all_true_ages, all_predictions, age_bins):
    """Evaluate performance by age group"""
    print("\n✓ Age-stratified analysis:")
    
    age_group_metrics = []
    
    for i in range(len(age_bins) - 1):
        age_min = age_bins[i]
        age_max = age_bins[i + 1]
        
        # Filter by age group
        mask = (all_true_ages >= age_min) & (all_true_ages < age_max)
        true_ages_group = all_true_ages[mask]
        pred_ages_group = all_predictions[mask]
        
        if len(true_ages_group) == 0:
            continue
        
        # Compute metrics for this group
        mae = mean_absolute_error(true_ages_group, pred_ages_group)
        rmse = np.sqrt(mean_squared_error(true_ages_group, pred_ages_group))
        
        age_group_metrics.append({
            'age_range': f'{age_min}-{age_max}',
            'n_subjects': len(true_ages_group),
            'mae': mae,
            'rmse': rmse
        })
        
        print(f"  Age {age_min}-{age_max}: N={len(true_ages_group)}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    
    return age_group_metrics


def plot_results(all_true_ages, all_predictions, brain_age_gap, output_dir):
    """Create evaluation plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Scatter plot: True vs Predicted Age
    plt.figure(figsize=(10, 8))
    plt.scatter(all_true_ages, all_predictions, alpha=0.5, s=30)
    plt.plot([all_true_ages.min(), all_true_ages.max()], 
             [all_true_ages.min(), all_true_ages.max()], 
             'r--', lw=2, label='Identity line')
    plt.xlabel('Chronological Age (years)', fontsize=12)
    plt.ylabel('Predicted Brain Age (years)', fontsize=12)
    plt.title('Brain Age Prediction Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'age_prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Brain Age Gap Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(brain_age_gap, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(0, color='r', linestyle='--', lw=2, label='Zero BAG')
    plt.axvline(np.mean(brain_age_gap), color='g', linestyle='--', lw=2, 
                label=f'Mean BAG: {np.mean(brain_age_gap):.2f}')
    plt.xlabel('Brain Age Gap (years)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Brain Age Gap Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'brain_age_gap_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Residual Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(all_true_ages, brain_age_gap, alpha=0.5, s=30)
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.xlabel('Chronological Age (years)', fontsize=12)
    plt.ylabel('Brain Age Gap (years)', fontsize=12)
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'residual_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Bland-Altman Plot
    mean_age = (all_true_ages + all_predictions) / 2
    diff_age = all_predictions - all_true_ages
    mean_diff = np.mean(diff_age)
    std_diff = np.std(diff_age)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_age, diff_age, alpha=0.5, s=30)
    plt.axhline(mean_diff, color='r', linestyle='-', lw=2, label=f'Mean: {mean_diff:.2f}')
    plt.axhline(mean_diff + 1.96*std_diff, color='g', linestyle='--', lw=2, 
                label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}')
    plt.axhline(mean_diff - 1.96*std_diff, color='g', linestyle='--', lw=2,
                label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}')
    plt.xlabel('Mean of True and Predicted Age (years)', fontsize=12)
    plt.ylabel('Difference (Predicted - True)', fontsize=12)
    plt.title('Bland-Altman Plot', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'bland_altman_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate brain aging model')
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
        default='outputs/evaluation',
        help='Directory to save evaluation results'
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Brain Aging Neural ODE - Evaluation")
    print("="*60)
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    device = config_manager.device
    
    # Setup data
    print("\n✓ Loading test data...")
    data_module = BrainAgingDataModule(config)
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
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
    
    # Evaluate
    metrics, all_predictions, all_true_ages, brain_age_gap = evaluate_model(
        model, test_loader, device, config
    )
    
    # Print overall metrics
    print("\n" + "="*60)
    print("Overall Performance")
    print("="*60)
    print(f"MAE: {metrics['mae']:.4f} years")
    print(f"RMSE: {metrics['rmse']:.4f} years")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
    print(f"Spearman r: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.2e})")
    print(f"Mean Brain Age Gap: {metrics['mean_bag']:.4f} ± {metrics['std_bag']:.4f} years")
    
    # Age-stratified analysis
    age_bins = config['evaluation']['age_bins']
    age_group_metrics = evaluate_by_age_group(all_true_ages, all_predictions, age_bins)
    
    # Create plots
    plot_results(all_true_ages, all_predictions, brain_age_gap, args.output_dir)
    
    # Save metrics
    import json
    output_path = Path(args.output_dir) / 'metrics.json'
    with open(output_path, 'w') as f:
# Convert numpy types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        json.dump({
            'overall': convert_to_native(metrics),
            'age_groups': convert_to_native(age_group_metrics)
        }, f, indent=2)

    
    print(f"\n✓ Evaluation complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()

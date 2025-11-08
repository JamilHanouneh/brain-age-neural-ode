"""
Visualization utilities for brain imaging
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import nibabel as nib
from typing import Optional, Tuple


class BrainVisualizer:
    """Visualize brain MRI data and results"""
    
    @staticmethod
    def plot_brain_slices(
        brain: np.ndarray,
        slice_indices: Optional[Tuple[int, int, int]] = None,
        title: str = "Brain MRI",
        cmap: str = 'gray',
        figsize: Tuple[int, int] = (12, 4)
    ):
        """
        Plot axial, coronal, and sagittal slices
        
        Args:
            brain: 3D brain array [depth, height, width]
            slice_indices: (axial, coronal, sagittal) slice indices
            title: Figure title
            cmap: Colormap
            figsize: Figure size
        """
        if slice_indices is None:
            # Default: middle slices
            slice_indices = (
                brain.shape[0] // 2,
                brain.shape[1] // 2,
                brain.shape[2] // 2
            )
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Axial slice
        axes[0].imshow(brain[slice_indices[0], :, :], cmap=cmap)
        axes[0].set_title('Axial')
        axes[0].axis('off')
        
        # Coronal slice
        axes[1].imshow(brain[:, slice_indices[1], :], cmap=cmap)
        axes[1].set_title('Coronal')
        axes[1].axis('off')
        
        # Sagittal slice
        axes[2].imshow(brain[:, :, slice_indices[2]], cmap=cmap)
        axes[2].set_title('Sagittal')
        axes[2].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_age_prediction_scatter(
        ages_true: np.ndarray,
        ages_pred: np.ndarray,
        figsize: Tuple[int, int] = (8, 8),
        save_path: Optional[str] = None
    ):
        """Plot true vs predicted ages"""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(ages_true, ages_pred, alpha=0.6, s=50, edgecolors='k')
        
        # Identity line
        min_age = min(ages_true.min(), ages_pred.min())
        max_age = max(ages_true.max(), ages_pred.max())
        ax.plot([min_age, max_age], [min_age, max_age], 'r--', lw=2, label='Identity')
        
        ax.set_xlabel('Chronological Age (years)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Brain Age (years)', fontsize=12, fontweight='bold')
        ax.set_title('Brain Age Prediction', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_brain_age_gap_histogram(
        brain_age_gap: np.ndarray,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """Plot brain age gap distribution"""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(brain_age_gap, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        
        # Add statistics
        mean_bag = np.mean(brain_age_gap)
        std_bag = np.std(brain_age_gap)
        
        ax.axvline(mean_bag, color='r', linestyle='--', lw=2, 
                   label=f'Mean: {mean_bag:.2f} years')
        ax.axvline(0, color='g', linestyle='--', lw=2, label='Zero BAG')
        
        ax.set_xlabel('Brain Age Gap (years)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Brain Age Gap Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_age_trajectories(
        ages: np.ndarray,
        figures: list,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """Plot multiple brain templates showing aging trajectory"""
        n_ages = len(ages)
        n_slices = 3  # Axial, coronal, sagittal
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_slices, n_ages, figure=fig)
        
        for i, age in enumerate(ages):
            brain = figures[i]
            
            slice_indices = (
                brain.shape[0] // 2,
                brain.shape[1] // 2,
                brain.shape[2] // 2
            )
            
            # Axial
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(brain[slice_indices[0], :, :], cmap='gray')
            ax.set_title(f'Age {age}', fontweight='bold')
            ax.axis('off')
            
            # Coronal
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(brain[:, slice_indices[1], :], cmap='gray')
            ax.axis('off')
            
            # Sagittal
            ax = fig.add_subplot(gs[2, i])
            ax.imshow(brain[:, :, slice_indices[2]], cmap='gray')
            ax.axis('off')
        
        fig.suptitle('Brain Aging Trajectory', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_uncertainty_maps(
        mean_pred: np.ndarray,
        std_pred: np.ndarray,
        figsize: Tuple[int, int] = (14, 5),
        save_path: Optional[str] = None
    ):
        """Plot mean and uncertainty predictions"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Mean predictions
        im1 = axes[0].scatter(range(len(mean_pred)), mean_pred, 
                              c=mean_pred, cmap='viridis', s=50, alpha=0.7)
        axes[0].set_xlabel('Subject', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Predicted Age (years)', fontsize=12, fontweight='bold')
        axes[0].set_title('Mean Predictions', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[0])
        
        # Uncertainty
        im2 = axes[1].scatter(range(len(std_pred)), std_pred,
                              c=std_pred, cmap='plasma', s=50, alpha=0.7)
        axes[1].set_xlabel('Subject', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Uncertainty (std)', fontsize=12, fontweight='bold')
        axes[1].set_title('Prediction Uncertainty', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[1])
        
        fig.suptitle('Uncertainty Quantification', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

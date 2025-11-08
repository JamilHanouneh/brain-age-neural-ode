"""
Evaluation metrics for brain aging
"""
import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error
)
from scipy.stats import pearsonr, spearmanr, kendalltau


class BrainAgingMetrics:
    """Compute brain aging-specific metrics"""
    
    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared"""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def compute_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute various correlations"""
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        kendall_r, kendall_p = kendalltau(y_true, y_pred)
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'kendall_r': kendall_r,
            'kendall_p': kendall_p
        }
    
    @staticmethod
    def compute_brain_age_gap(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute brain age gap statistics"""
        bag = y_pred - y_true
        
        return {
            'mean_bag': np.mean(bag),
            'std_bag': np.std(bag),
            'median_bag': np.median(bag),
            'min_bag': np.min(bag),
            'max_bag': np.max(bag)
        }
    
    @staticmethod
    def compute_median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Median Absolute Error"""
        return median_absolute_error(y_true, y_pred)
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute all metrics"""
        metrics = {
            'mae': BrainAgingMetrics.compute_mae(y_true, y_pred),
            'rmse': BrainAgingMetrics.compute_rmse(y_true, y_pred),
            'r2': BrainAgingMetrics.compute_r2(y_true, y_pred),
            'median_ae': BrainAgingMetrics.compute_median_absolute_error(y_true, y_pred)
        }
        
        metrics.update(BrainAgingMetrics.compute_correlation(y_true, y_pred))
        metrics.update(BrainAgingMetrics.compute_brain_age_gap(y_true, y_pred))
        
        return metrics

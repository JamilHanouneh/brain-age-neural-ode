"""
Unit tests for preprocessing
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path

import torch
from src.data.preprocessing import BrainPreprocessor


class TestBrainPreprocessor:
    """Test brain preprocessing"""
    
    @pytest.fixture
    def config(self):
        return {
            'data': {
                'preprocessing': {
                    'target_shape': [64, 64, 64],
                    'normalize': True,
                    'normalization_method': 'z-score',
                    'pca_components': 50
                },
                'processed_data_dir': 'data/processed',
                'metadata_dir': 'data/metadata'
            }
        }
    
    @pytest.fixture
    def preprocessor(self, config):
        return BrainPreprocessor(config)
    
    def test_preprocessor_creation(self, preprocessor):
        """Test preprocessor can be created"""
        assert preprocessor is not None
        assert preprocessor.target_shape == (64, 64, 64)
    
    def test_skull_strip(self, preprocessor):
        """Test skull stripping"""
        # Create dummy brain image
        brain = np.random.rand(100, 100, 100)
        brain[:30, :, :] = 0  # Add background
        
        stripped = preprocessor._simple_skull_strip(brain)
        
        # Check output shape
        assert stripped.shape == brain.shape
        
        # Check that background is removed
        assert np.mean(stripped) < np.mean(brain)
    
    def test_intensity_normalization_zscore(self, preprocessor):
        """Test Z-score normalization"""
        # Create dummy brain image
        brain = np.random.rand(64, 64, 64) * 100 + 50
        
        normalized = preprocessor._normalize_intensity(brain)
        
        # Check normalized values
        brain_voxels = normalized[normalized > 0]
        assert abs(np.mean(brain_voxels)) < 0.1  # Should be close to 0
        assert abs(np.std(brain_voxels) - 1.0) < 0.1  # Should be close to 1
    
    def test_resize_image(self, preprocessor):
        """Test image resizing"""
        original = np.random.rand(100, 100, 100)
        target_shape = (64, 64, 64)
        
        resized = preprocessor._resize_image(original, target_shape)
        
        # Check output shape
        assert resized.shape == target_shape
    
    def test_process_single_brain(self, preprocessor):
        """Test processing single brain"""
        # Create temporary NIFTI file
        import nibabel as nib
        
        brain_data = np.random.rand(100, 100, 100)
        nifti_img = nib.Nifti1Image(brain_data, affine=np.eye(4))
        
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
            nib.save(nifti_img, f.name)
            temp_path = f.name
        
        try:
            processed = preprocessor._process_single_brain(temp_path)
            
            assert processed is not None
            assert processed.shape == preprocessor.target_shape
            assert processed.dtype == np.float32
        finally:
            Path(temp_path).unlink()  # Clean up


class TestDataAugmentation:
    """Test data augmentation"""
    
    def test_augmentation_applied(self):
        """Test augmentation is applied"""
        from src.data.augmentation import BrainAugmentation
        
        augmenter = BrainAugmentation(probability=1.0)  # Always apply
        
        x = torch.randn(64, 64, 64)
        x_aug = augmenter(x)
        
        # Should be different (with high probability)
        assert not torch.allclose(x, x_aug)
    
    def test_augmentation_probability(self):
        """Test augmentation probability"""
        from src.data.augmentation import BrainAugmentation
        import torch
        
        augmenter = BrainAugmentation(probability=0.0)  # Never apply
        
        x = torch.randn(64, 64, 64)
        x_aug = augmenter(x)
        
        # Should be identical
        assert torch.allclose(x, x_aug)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Pytest configuration and fixtures
"""
import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Get appropriate device for testing"""
    return torch.device('cpu')


@pytest.fixture
def dummy_brain():
    """Create dummy brain image"""
    return np.random.rand(64, 64, 64).astype(np.float32)


@pytest.fixture
def dummy_tensor():
    """Create dummy tensor"""
    return torch.randn(4, 100)


@pytest.fixture(autouse=True)
def seed_everything():
    """Seed for reproducibility"""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def pytest_configure(config):
    """Add custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )

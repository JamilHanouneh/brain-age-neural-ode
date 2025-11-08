"""
Expanded unit tests for models
"""
import pytest
import torch
import numpy as np
from src.models.neural_ode import NeuralODEFlow, ODEFunc
from src.models.normalizing_flow import NormalizingFlow, CouplingLayer
from src.models.siren import SIREN, SineLayer


class TestODEFunc:
    """Test ODE function"""
    
    def test_ode_func_creation(self):
        """Test ODE function creation"""
        ode_func = ODEFunc(
            hidden_dim=128,
            num_layers=3,
            activation='tanh'
        )
        assert ode_func is not None
    
    def test_ode_func_forward(self):
        """Test ODE function forward pass"""
        ode_func = ODEFunc(hidden_dim=128, num_layers=2)
        
        t = torch.tensor([0.0])
        h = torch.randn(4, 128)
        
        dh_dt = ode_func(t, h)
        
        assert dh_dt.shape == h.shape
        assert not torch.isnan(dh_dt).any()


class TestNeuralODEFlow:
    """Test Neural ODE Flow"""
    
    @pytest.fixture
    def model(self):
        return NeuralODEFlow(
            input_dim=100,
            latent_dim=100,
            hidden_dim=128,
            num_layers=2,
            solver='dopri5',
            use_adjoint=False
        )
    
    def test_model_forward(self, model):
        """Test forward pass"""
        x = torch.randn(4, 100)
        age, z, h = model(x)
        
        assert age.shape == (4, 1)
        assert z.shape == (4, 99)
        assert h.shape == (4, 100)
    
    def test_model_inverse(self, model):
        """Test inverse pass"""
        age = torch.randn(4, 1)
        z = torch.randn(4, 99)
        
        x = model.inverse(age, z)
        
        assert x.shape == (4, 100)
    
    def test_full_cycle(self, model):
        """Test encoding and decoding"""
        x_original = torch.randn(4, 100)
        
        # Encode
        age, z, _ = model(x_original)
        
        # Decode
        x_reconstructed = model.inverse(age, z)
        
        # Should have reasonable error
        error = torch.mean((x_original - x_reconstructed) ** 2)
        assert error < 1.0


class TestCouplingLayer:
    """Test coupling layers"""
    
    def test_coupling_layer_creation(self):
        """Test coupling layer creation"""
        layer = CouplingLayer(input_dim=100, hidden_dim=64)
        assert layer is not None
    
    def test_coupling_layer_forward(self):
        """Test forward pass"""
        layer = CouplingLayer(input_dim=100, hidden_dim=64)
        
        x = torch.randn(4, 100)
        x_transformed, log_det_J = layer(x)
        
        assert x_transformed.shape == x.shape
        assert log_det_J.shape == (4,)
        assert not torch.isnan(x_transformed).any()
    
    def test_coupling_layer_inverse(self):
        """Test inverse pass"""
        layer = CouplingLayer(input_dim=100, hidden_dim=64)
        
        x = torch.randn(4, 100)
        x_transformed, _ = layer(x)
        x_recovered, _ = layer.inverse(x_transformed)
        
        # Should recover approximate original
        error = torch.mean((x - x_recovered) ** 2)
        assert error < 0.1


class TestNormalizingFlow:
    """Test normalizing flow"""
    
    def test_flow_creation(self):
        """Test flow creation"""
        flow = NormalizingFlow(
            input_dim=100,
            num_layers=8,
            hidden_dim=64
        )
        assert flow is not None
    
    def test_flow_forward(self):
        """Test forward pass"""
        flow = NormalizingFlow(input_dim=100, num_layers=4)
        
        x = torch.randn(4, 100)
        z, log_det_J = flow(x)
        
        assert z.shape == x.shape
        assert log_det_J.shape == (4,)
    
    def test_flow_inverse(self):
        """Test inverse pass"""
        flow = NormalizingFlow(input_dim=100, num_layers=4)
        
        z = torch.randn(4, 100)
        x, log_det_J = flow.inverse(z)
        
        assert x.shape == z.shape
        assert log_det_J.shape == (4,)


class TestSineLayer:
    """Test SIREN sine layer"""
    
    def test_sine_layer_creation(self):
        """Test sine layer creation"""
        layer = SineLayer(in_features=64, out_features=64, omega_0=30.0)
        assert layer is not None
    
    def test_sine_layer_forward(self):
        """Test forward pass"""
        layer = SineLayer(in_features=64, out_features=64)
        
        x = torch.randn(4, 64)
        y = layer(x)
        
        assert y.shape == (4, 64)
        assert torch.all(y >= -1.1) and torch.all(y <= 1.1)  # Sine bounds


class TestSIREN:
    """Test full SIREN model"""
    
    def test_siren_creation(self):
        """Test SIREN creation"""
        model = SIREN(
            in_features=5,
            hidden_features=64,
            hidden_layers=4,
            out_features=3
        )
        assert model is not None
    
    def test_siren_forward(self):
        """Test forward pass"""
        model = SIREN(
            in_features=5,
            hidden_features=64,
            hidden_layers=3,
            out_features=3
        )
        
        # Input: (x, y, z, age, pathology)
        coords = torch.randn(4, 5)
        output = model(coords)
        
        assert output.shape == (4, 3)
        assert not torch.isnan(output).any()
    
    def test_siren_coordinate_encoding(self):
        """Test SIREN learns coordinate encoding"""
        model = SIREN(
            in_features=3,
            hidden_features=32,
            hidden_layers=3,
            out_features=1
        )
        
        # Test that it can learn simple patterns
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # Create simple training data
        coords = torch.randn(100, 3)
        targets = torch.sin(coords[:, 0:1]) * torch.cos(coords[:, 1:2])
        
        # Train briefly
        for _ in range(10):
            pred = model(coords)
            loss = torch.mean((pred - targets) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Should have reduced loss
        with torch.no_grad():
            final_loss = torch.mean((model(coords) - targets) ** 2)
        assert final_loss < 1.0  # Should learn something


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

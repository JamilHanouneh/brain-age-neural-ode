"""
Multimodal attention fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalAttention(nn.Module):
    """Attention-based multimodal fusion"""
    
    def __init__(
        self,
        modality_dims: list,
        fusion_dim: int = 256,
        num_heads: int = 4
    ):
        """
        Args:
            modality_dims: List of input dimensions for each modality
            fusion_dim: Dimension for attention computation
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        
        # Project each modality to fusion dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, fusion_dim)
            for dim in modality_dims
        ])
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(fusion_dim * len(modality_dims), fusion_dim)
    
    def forward(self, modalities: list) -> tuple:
        """
        Args:
            modalities: List of modality tensors
        
        Returns:
            fused: Fused representation
            attention_weights: Attention weights for each modality
        """
        # Project modalities
        projected = [
            proj(mod) for proj, mod in zip(self.projections, modalities)
        ]
        
        # Stack for attention
        stacked = torch.stack(projected, dim=1)  # [batch, num_modalities, fusion_dim]
        
        # Self-attention
        attended, attention_weights = self.attention(stacked, stacked, stacked)
        
        # Flatten and project
        batch_size = stacked.size(0)
        flattened = attended.reshape(batch_size, -1)
        fused = self.output_proj(flattened)
        
        return fused, attention_weights


class EarlyFusion(nn.Module):
    """Early fusion: concatenate modalities"""
    
    def __init__(self, modality_dims: list, output_dim: int):
        """
        Args:
            modality_dims: List of input dimensions
            output_dim: Output dimension
        """
        super().__init__()
        
        total_dim = sum(modality_dims)
        self.linear = nn.Linear(total_dim, output_dim)
    
    def forward(self, modalities: list) -> torch.Tensor:
        """Concatenate and project"""
        concatenated = torch.cat(modalities, dim=1)
        return self.linear(concatenated)


class LateFusion(nn.Module):
    """Late fusion: average predictions from each modality"""
    
    def __init__(self, modality_dims: list, hidden_dim: int):
        """
        Args:
            modality_dims: List of input dimensions
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            for dim in modality_dims
        ])
    
    def forward(self, modalities: list) -> torch.Tensor:
        """Average outputs from each modality"""
        outputs = [head(mod) for head, mod in zip(self.heads, modalities)]
        return torch.stack(outputs).mean(dim=0)

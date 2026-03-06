"""
MLP architecture used in Appendix A.1 of:
"Understanding Hallucinations through Mode Interpolation".
"""

import torch
import torch.nn as nn
from utils import sinusoidal_embedding



DEFAULT_NONLINEARITY = nn.LeakyReLU(negative_slope=0.02)

class DEFAULT_NORMALIZER(nn.LayerNorm):
    """Default normalization layer (LayerNorm), matching original toy implementation."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        # Keep signature compatible with previous class usage.
        del momentum
        super().__init__(normalized_shape=num_features, eps=eps)




class Block(nn.Module):
    """
    Residual MLP block with timestep conditioning.

    Structure:
        Norm → LeakyReLU → Linear
        + projected timestep embedding
        → Norm → LeakyReLU → Linear
        + skip connection
    """

    nonlinearity = DEFAULT_NONLINEARITY
    normalizer = DEFAULT_NORMALIZER

    def __init__(self, in_features, out_features, t_features):
        super().__init__()

        self.norm0 = self.normalizer(num_features=in_features)
        self.norm1 = self.normalizer(num_features=out_features)
        self.t_proj = nn.Linear(in_features=t_features, out_features=out_features)
        self.linear0 = nn.Linear(in_features=in_features, out_features=out_features)
        self.linear1 = nn.Linear(in_features=out_features, out_features=out_features)
        self.skip = (
            nn.Identity()
            if in_features == out_features
            else nn.Linear(in_features=in_features, out_features=out_features)
        )

    def forward(self, x, t_embedding):
        """
        Forward pass.

        Args:
            x: (B, in_features)
            t_embedding: (B, t_features)

        Returns:
            (B, out_features)
        """
        h = self.linear0(self.nonlinearity(self.norm0(x)))
        t_projected = self.t_proj(t_embedding)
        h_t = h + t_projected
        out = self.linear1(self.nonlinearity(self.norm1(h_t)))

        return out + self.skip(x)


class NN(nn.Module):
    """
    Timestep-conditioned residual MLP.

    Architecture:
        Linear → 3×Block → Norm → Linear
    """

    nonlinearity = DEFAULT_NONLINEARITY
    normalizer = DEFAULT_NORMALIZER

    def __init__(self, in_features, hidden_features, t_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.t_features = t_features

        self.linear0 = nn.Linear(in_features=in_features, out_features=hidden_features)

        self.block0 = Block(hidden_features, hidden_features, t_features)
        self.block1 = Block(hidden_features, hidden_features, t_features)
        self.block2 = Block(hidden_features, hidden_features, t_features)

        self.norm = self.normalizer(hidden_features)
        self.linear1 = nn.Linear(hidden_features, in_features)

        self.mlp_t = nn.Sequential(
            nn.Linear(hidden_features, t_features),
            self.nonlinearity,
            nn.Linear(t_features, t_features),
        )

    def forward(self, x, t):
        """
        Forward pass.

        Args:
            x: (B, in_features)
            t: (B,) diffusion timesteps

        Returns:
            (B, in_features) model output
        """
        t_embedding = sinusoidal_embedding(t, self.hidden_features)
        t_embedding = self.mlp_t(t_embedding)

        x = self.linear0(x)
        x = self.block0(x, t_embedding)
        x = self.block1(x, t_embedding)
        x = self.block2(x, t_embedding)

        return self.linear1(self.norm(x))
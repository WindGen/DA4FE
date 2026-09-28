"""Map a DA4FE feature vector to Stable-Diffusion cross-attention tokens."""

import torch
import torch.nn as nn


class EEGConditionAdapter(nn.Module):
    """Small trainable adapter used as the EEG conditioning stage.

    Stable Diffusion's U-Net consumes a sequence of ``context_dim`` vectors.
    Keeping this adapter separate makes it possible to freeze both DA4FE and
    the diffusion backbone during the initial generation training stage.
    """

    def __init__(self, input_dim, context_dim=768, num_tokens=4, hidden_dim=1024, dropout=0.0):
        super().__init__()
        if input_dim <= 0 or context_dim <= 0 or num_tokens <= 0:
            raise ValueError("input_dim, context_dim and num_tokens must be positive")
        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.num_tokens = int(num_tokens)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_tokens * self.context_dim),
        )
        self.output_norm = nn.LayerNorm(self.context_dim)

    def forward(self, features):
        if features.ndim != 2:
            raise ValueError(f"Expected DA4FE features [B, D], got {tuple(features.shape)}")
        tokens = self.net(features).view(features.shape[0], self.num_tokens, self.context_dim)
        return self.output_norm(tokens)

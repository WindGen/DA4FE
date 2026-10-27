"""Projection head used to align DA4FE features with frozen CLIP features."""

import torch.nn as nn


class EEGImageProjection(nn.Module):
    """Map a DA4FE feature vector into the frozen image encoder's space."""

    def __init__(self, input_dim, output_dim=1024, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, features):
        return self.net(features)

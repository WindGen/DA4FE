import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Augmentation import get_augmentation


class Aug_Channel_Embedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        aug_idxs = configs.augmentations.split(",")
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in aug_idxs]
        )
        self.Channel_Embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.pos_emb = PositionalEmbedding(d_model=configs.seq_len)

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.transpose(1, 2)  # (batch_size, enc_in, seq_len)
        aug_idx = random.randint(0, len(self.augmentation) - 1)
        x_aug = self.augmentation[aug_idx](x)
        x_aug = x_aug + self.pos_emb(x_aug)
        return self.Channel_Embedding(x_aug)


class Aug_Temporal_Embedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        aug_idxs = configs.augmentations.split(",")
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in aug_idxs]
        )
        self.Temporal_Embedding = (
            CrossChannelPatching(configs)
            if self.patch_len > 1
            else nn.Linear(configs.enc_in, configs.d_model)
        )
        self.pos_emb = PositionalEmbedding(d_model=configs.seq_len)

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.transpose(1, 2)  # (batch_size, enc_in, seq_len)
        aug_idx = random.randint(0, len(self.augmentation) - 1)
        x_aug = self.augmentation[aug_idx](x)
        x_aug = x_aug + self.pos_emb(x_aug)
        if self.patch_len == 1:
            x_aug = x_aug.transpose(1, 2)
        return self.Temporal_Embedding(x_aug)


class Aug_Frequency_Embedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        self.freq_seq_len = configs.seq_len // 2 + 1
        aug_idxs = configs.augmentations.split(",")
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in aug_idxs]
        )
        self.Frequency_Embedding = (
            CrossChannelPatching(configs)
            if self.patch_len > 1
            else nn.Linear(configs.enc_in, configs.d_model)
        )
        self.pos_emb = PositionalEmbedding(d_model=self.freq_seq_len)

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.transpose(1, 2)  # (batch_size, enc_in, seq_len)
        aug_idx = random.randint(0, len(self.augmentation) - 1)
        x_aug = self.augmentation[aug_idx](x)
        
        x_fft = torch.fft.rfft(x_aug, dim=-1)
        x_freq = torch.log1p(torch.abs(x_fft).pow(2))

        x_freq = x_freq + self.pos_emb(x_freq)
        if self.patch_len == 1:
            x_freq = x_freq.transpose(1, 2)
        return self.Frequency_Embedding(x_freq)


class BranchFusion(nn.Module):
    def __init__(
        self,
        num_branches,
        branch_dims,
        mode="add",
        hidden_dim=None,
        output_dim=None,
        dropout=0.0,
        branch_weights=None,
        normalize_add_weights=True,
    ):
        super().__init__()
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if len(branch_dims) != num_branches:
            raise ValueError(
                f"Expected {num_branches} branch dims, got {len(branch_dims)}"
            )

        self.num_branches = num_branches
        self.branch_dims = list(branch_dims)
        self.mode = mode
        if output_dim is None:
            output_dim = self.branch_dims[0]
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else self.output_dim

        if mode == "add":
            self.fusion = None
            self.projections = nn.ModuleList(
                [
                    nn.Identity()
                    if branch_dim == self.output_dim
                    else nn.Linear(branch_dim, self.output_dim)
                    for branch_dim in self.branch_dims
                ]
            )
            if branch_weights is None:
                branch_weights = [1.0] * num_branches
            if len(branch_weights) != num_branches:
                raise ValueError(
                    f"Expected {num_branches} branch weights, got {len(branch_weights)}"
                )
            weights = torch.tensor(branch_weights, dtype=torch.float32)
            if torch.any(weights < 0):
                raise ValueError("branch_weights must be non-negative")
            if normalize_add_weights:
                weight_sum = torch.sum(weights)
                if weight_sum <= 0:
                    raise ValueError("branch_weights must sum to a positive value")
                weights = weights / weight_sum
            self.register_buffer("branch_weights", weights)
        elif mode == "concat_mlp":
            self.projections = None
            self.branch_norms = nn.ModuleList(
                [nn.LayerNorm(branch_dim) for branch_dim in self.branch_dims]
            )
            self.fusion = nn.Sequential(
                nn.Linear(sum(self.branch_dims), self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.Dropout(dropout),
            )
            self.branch_weights = None
        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

    def forward(self, features):
        if not features:
            raise ValueError("features must contain at least one tensor")
        if len(features) == 1:
            return features[0]
        if len(features) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} features, got {len(features)}"
            )

        if self.mode == "add":
            projected = [
                projection(feature)
                for projection, feature in zip(self.projections, features)
            ]
            stacked = torch.stack(projected, dim=0)
            weight_shape = [self.num_branches] + [1] * (stacked.dim() - 1)
            weights = self.branch_weights.view(*weight_shape)
            return torch.sum(stacked * weights, dim=0)

        normalized = [
            norm(feature) for norm, feature in zip(self.branch_norms, features)
        ]
        return self.fusion(torch.cat(normalized, dim=-1))


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = True

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        sin_term = torch.sin(position * div_term)
        cos_term = torch.cos(position * div_term)
        pe[:, 0::2] = sin_term
        pe[:, 1::2] = cos_term[:, : pe[:, 1::2].shape[1]]

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class CrossChannelPatching(nn.Module):
    def __init__(self, configs):
        super().__init__()
        patch_len = configs.patch_len
        stride = configs.patch_len
        self.tokenConv = nn.Conv2d(
            in_channels=1,
            out_channels=configs.d_model,
            kernel_size=(configs.enc_in, patch_len),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )
        self.padding = nn.ReplicationPad1d((0, stride))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.padding(x).unsqueeze(1)
        x = self.tokenConv(x).squeeze(2).transpose(1, 2)
        return x

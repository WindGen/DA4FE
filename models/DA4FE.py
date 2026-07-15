from copy import deepcopy

import torch.nn as nn

from layers.Components import (
    Aug_Channel_Embedding,
    Aug_Frequency_Embedding,
    Aug_Temporal_Embedding,
    BranchFusion,
)
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.v_layer = configs.v_layer
        self.t_layer = configs.t_layer
        self.f_layer = (
            configs.f_layer if getattr(configs, "f_layer", None) is not None else configs.t_layer
        )
        self.channel_dim = (
            getattr(configs, "da4fe_channel_dim", None) or configs.d_model
        )
        self.temporal_dim = (
            getattr(configs, "da4fe_temporal_dim", None) or configs.d_model
        )
        self.frequency_dim = (
            getattr(configs, "da4fe_frequency_dim", None) or configs.d_model
        )
        self.fusion_output_dim = (
            getattr(configs, "da4fe_fusion_out_dim", None) or configs.d_model
        )
        self.active_branches = (
            int(self.v_layer > 0) + int(self.t_layer > 0) + int(self.f_layer > 0)
        )
        self.active_branch_weights = []
        self.active_branch_dims = []

        def _branch_configs(branch_dim):
            branch_configs = deepcopy(configs)
            branch_configs.d_model = branch_dim
            return branch_configs

        self.channel_encoder = (
            nn.Sequential(
                Aug_Channel_Embedding(_branch_configs(self.channel_dim)),
                Encoder(
                    [
                        EncoderLayer(_branch_configs(self.channel_dim))
                        for _ in range(self.v_layer)
                    ]
                ),
            )
            if self.v_layer > 0
            else nn.Identity()
        )
        if self.v_layer > 0:
            self.active_branch_weights.append(
                getattr(configs, "da4fe_channel_weight", 0.3)
            )
            self.active_branch_dims.append(self.channel_dim)
        self.temporal_encoder = (
            nn.Sequential(
                Aug_Temporal_Embedding(_branch_configs(self.temporal_dim)),
                Encoder(
                    [
                        EncoderLayer(_branch_configs(self.temporal_dim))
                        for _ in range(self.t_layer)
                    ]
                ),
            )
            if self.t_layer > 0
            else nn.Identity()
        )
        if self.t_layer > 0:
            self.active_branch_weights.append(
                getattr(configs, "da4fe_temporal_weight", 0.6)
            )
            self.active_branch_dims.append(self.temporal_dim)
        self.frequency_encoder = (
            nn.Sequential(
                Aug_Frequency_Embedding(_branch_configs(self.frequency_dim)),
                Encoder(
                    [
                        EncoderLayer(_branch_configs(self.frequency_dim))
                        for _ in range(self.f_layer)
                    ]
                ),
            )
            if self.f_layer > 0
            else nn.Identity()
        )
        if self.f_layer > 0:
            self.active_branch_weights.append(
                getattr(configs, "da4fe_frequency_weight", 0.1)
            )
            self.active_branch_dims.append(self.frequency_dim)
        self.fusion = BranchFusion(
            num_branches=self.active_branches,
            branch_dims=self.active_branch_dims,
            mode=getattr(configs, "da4fe_fusion_mode", "add"),
            hidden_dim=getattr(configs, "da4fe_fusion_hidden_dim", None),
            output_dim=self.fusion_output_dim,
            dropout=configs.dropout,
            branch_weights=self.active_branch_weights,
        )
        self.projector = nn.Linear(self.fusion_output_dim, configs.num_class)

    def forward(self, x_enc, return_features=False):
        features = []
        if self.v_layer > 0:
            features.append(self.channel_encoder(x_enc).mean(1))
        if self.t_layer > 0:
            features.append(self.temporal_encoder(x_enc).mean(1))
        if self.f_layer > 0:
            features.append(self.frequency_encoder(x_enc).mean(1))

        fused = self.fusion(features)
        logits = self.projector(fused)
        if return_features:
            return logits, fused
        return logits

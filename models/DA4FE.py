import torch.nn as nn

from layers.Components import (
    Aug_Channel_Embedding,
    Aug_Frequency_Embedding,
    Aug_Temporal_Embedding,
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

        self.channel_encoder = (
            nn.Sequential(
                Aug_Channel_Embedding(configs),
                Encoder([EncoderLayer(configs) for _ in range(self.v_layer)]),
            )
            if self.v_layer > 0
            else nn.Identity()
        )
        self.temporal_encoder = (
            nn.Sequential(
                Aug_Temporal_Embedding(configs),
                Encoder([EncoderLayer(configs) for _ in range(self.t_layer)]),
            )
            if self.t_layer > 0
            else nn.Identity()
        )
        self.frequency_encoder = (
            nn.Sequential(
                Aug_Frequency_Embedding(configs),
                Encoder([EncoderLayer(configs) for _ in range(self.f_layer)]),
            )
            if self.f_layer > 0
            else nn.Identity()
        )
        self.projector = nn.Linear(configs.d_model, configs.num_class)

    def forward(self, x_enc):
        channel = self.channel_encoder(x_enc).mean(1) if self.v_layer > 0 else 0
        temporal = self.temporal_encoder(x_enc).mean(1) if self.t_layer > 0 else 0
        frequency = self.frequency_encoder(x_enc).mean(1) if self.f_layer > 0 else 0
        logits = self.projector(channel + temporal + frequency)
        return logits

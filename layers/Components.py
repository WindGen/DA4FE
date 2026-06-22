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
        x_freq = torch.abs(torch.fft.rfft(x_aug, dim=-1))
        x_freq = x_freq + self.pos_emb(x_freq)
        if self.patch_len == 1:
            x_freq = x_freq.transpose(1, 2)
        return self.Frequency_Embedding(x_freq)


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

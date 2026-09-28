"""Datasets and image preprocessing for DreamDiffusion-style Stage-3 training."""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from data_provider.data_loader import EEGImageNetHuggingFaceLoader


def _resize_crop(image, size, train):
    """DreamDiffusion-compatible RGB preprocessing with values in ``[-1, 1]``."""
    size = int(size)
    if size <= 0:
        raise ValueError(f"generation_image_size must be positive, got {size}")
    width, height = image.size
    scale = size / float(min(width, height))
    new_size = (max(size, int(round(width * scale))), max(size, int(round(height * scale))))
    image = image.resize(new_size, resample=2)  # PIL.Image.Resampling.BICUBIC
    max_left = new_size[0] - size
    max_top = new_size[1] - size
    if train:
        left = random.randint(0, max_left) if max_left else 0
        top = random.randint(0, max_top) if max_top else 0
    else:
        left = max_left // 2
        top = max_top // 2
    image = image.crop((left, top, left + size, top + size))
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(np.moveaxis(array, -1, 0)).contiguous()


class EEGImageNetHuggingFaceGenerationDataset(Dataset):
    """HF EEG-image pairs used by Stage-3 without a local image directory."""

    def __init__(self, args, flag):
        self.flag = str(flag).upper()
        self.training = self.flag == "TRAIN"
        self.image_size = int(getattr(args, "generation_image_size", 512))
        self.base = EEGImageNetHuggingFaceLoader(args=args, root_path="", flag=self.flag)
        # Fail at construction, rather than halfway through an epoch, with an
        # actionable message when a dataset revision has no paired images.
        self.base._resolve_image_column(self.base.dataset, required=True)
        if len(self.base):
            self.base.get_image(0)
        self.y = self.base.y
        self.num_class = self.base.num_class
        self.feature_dim = self.base.feature_dim
        self.max_seq_len = self.base.max_seq_len

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        eeg, label = self.base[index]
        image = _resize_crop(self.base.get_image(index), self.image_size, self.training)
        return {
            "eeg": eeg.float(),
            "image": image.float(),
            "label": label.long().reshape(()),
            "subject": self.base.get_subject(index),
            "label_name": self.base.get_label_name(index),
        }


def generation_collate(batch, max_len=None):
    """Pad variable-length EEG and stack image tensors for the generation model."""
    if not batch:
        raise ValueError("Cannot collate an empty generation batch")
    eeg_values = [item["eeg"] for item in batch]
    lengths = [int(value.shape[0]) for value in eeg_values]
    target_len = int(max_len or max(lengths))
    feature_dim = int(eeg_values[0].shape[-1])
    eeg = torch.zeros(len(batch), target_len, feature_dim, dtype=torch.float32)
    padding_mask = torch.zeros(len(batch), target_len, dtype=torch.bool)
    for index, value in enumerate(eeg_values):
        end = min(target_len, int(value.shape[0]))
        eeg[index, :end] = value[:end]
        padding_mask[index, :end] = True
    return {
        "eeg": eeg,
        "padding_mask": padding_mask,
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "subject": [item["subject"] for item in batch],
        "label_name": [item["label_name"] for item in batch],
    }

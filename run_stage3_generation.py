"""Train DreamDiffusion-style image generation from an existing DA4FE run."""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from exp.exp_stage3_generation import Exp_Stage3_Generation


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def str_or_none(value):
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return str(value)


parser = argparse.ArgumentParser(description="DreamDiffusion-style Stage-3 EEG image generation")

# DA4FE checkpoint and Hugging Face data configuration.
parser.add_argument("--stage1_run_dir", type=str_or_none, default=None, help="DA4FE run directory containing run_params.json, split_summary.json and checkpoint.pth")  # DA4FE参数目录
parser.add_argument("--stage1_checkpoint", type=str_or_none, default=None, help="optional DA4FE checkpoint path; otherwise checkpoint.pth is read from --stage1_run_dir")  # DA4FE权重
parser.add_argument("--stage1_run_params", type=str_or_none, default=None, help="optional run_params.json path; otherwise read from --stage1_run_dir")  # DA4FE参数文件
parser.add_argument("--stage1_split_summary", type=str_or_none, default=None, help="optional split_summary.json path; otherwise read from --stage1_run_dir")  # 数据划分参数
parser.add_argument("--eeg_hf_dataset_id", type=str_or_none, default=None, help="override Hugging Face dataset id from DA4FE run_params.json")  # HF数据集
parser.add_argument("--eeg_hf_cache_dir", type=str_or_none, default=None, help="override Hugging Face cache directory")  # HF缓存目录
parser.add_argument("--eeg_hf_split_mode", choices=["original", "self"], default=None, help="override original/self split mode from DA4FE run_params.json")  # HF划分方式
parser.add_argument("--eeg_hf_split_seed", type=int, default=None, help="random seed for HF self split")  # HF划分随机种子
parser.add_argument("--eeg_hf_train_ratio", type=float, default=None, help="train ratio for HF self split")  # HF训练比例
parser.add_argument("--eeg_hf_val_ratio", type=float, default=None, help="validation ratio for HF self split")  # HF验证比例
parser.add_argument("--eeg_hf_test_ratio", type=float, default=None, help="test ratio for HF self split")  # HF测试比例

# Local pretrained models; no large model is downloaded automatically.
parser.add_argument("--pretrained_root", type=str, default="../pretrained", help="directory containing stable-diffusion-v1-5 and clip-vit-large-patch14")  # 预训练权重根目录
parser.add_argument("--sd_model_path", type=str_or_none, default=None, help="local Diffusers Stable Diffusion directory; defaults to pretrained_root/stable-diffusion-v1-5")  # Stable Diffusion权重
parser.add_argument("--clip_model_path", type=str_or_none, default=None, help="local CLIP vision directory; defaults to pretrained_root/clip-vit-large-patch14")  # CLIP权重

# EEG-to-diffusion condition adapter.
parser.add_argument("--generation_context_tokens", type=int, default=4, help="number of EEG tokens passed to Stable Diffusion cross-attention")  # EEG条件token数
parser.add_argument("--generation_condition_dim", type=int, default=768, help="Stable Diffusion cross-attention dimension")  # 条件维度
parser.add_argument("--generation_adapter_hidden_dim", type=int, default=1024, help="hidden dimension of EEG condition adapter")  # 条件适配器隐藏维度
parser.add_argument("--generation_adapter_dropout", type=float, default=0.0, help="dropout in EEG condition adapter")  # 条件适配器dropout

# Generation data and optimization parameters.
parser.add_argument("--generation_image_size", type=int, default=512, help="square image resolution used by VAE and sampling")  # 图片尺寸
parser.add_argument("--generation_epochs", type=int, default=50, help="number of Stage-3 training epochs")  # 训练轮数
parser.add_argument("--generation_batch_size", type=int, default=2, help="Stage-3 batch size")  # 训练批大小
parser.add_argument("--generation_lr", type=float, default=1e-4, help="learning rate for EEG condition adapter")  # 条件适配器学习率
parser.add_argument("--generation_weight_decay", type=float, default=1e-2, help="AdamW weight decay")  # 权重衰减
parser.add_argument("--generation_clip_weight", type=float, default=0.1, help="weight of optional CLIP image alignment loss; zero disables CLIP loading")  # CLIP损失权重
parser.add_argument("--generation_stage1_lr", type=float, default=0.0, help="learning rate for DA4FE; zero freezes the Stage-1 feature extractor")  # DA4FE微调学习率
parser.add_argument("--generation_train_unet", type=str2bool, default=False, help="train Stable Diffusion cross-attention parameters in addition to the EEG adapter")  # 是否训练U-Net交叉注意力
parser.add_argument("--generation_grad_clip", type=float, default=1.0, help="gradient clipping norm")  # 梯度裁剪

# Evaluation and output parameters.
parser.add_argument("--generation_num_inference_steps", type=int, default=30, help="diffusion sampling steps")  # 采样步数
parser.add_argument("--generation_num_samples", type=int, default=4, help="number of generated images per EEG")  # 每条EEG生成图片数
parser.add_argument("--generation_eval_items", type=int, default=32, help="maximum EEG items sampled per split; zero means all")  # 采样样本上限
parser.add_argument("--generation_skip_sampling", type=str2bool, default=False, help="skip saving generated PNGs after training")  # 是否跳过采样
parser.add_argument("--output_dir", type=str, default="../result/generation", help="Stage-3 output directory")  # 输出目录

# Runtime parameters.
parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")  # 数据加载进程数
parser.add_argument("--seed", type=int, default=42, help="random seed")  # 随机种子
parser.add_argument("--use_gpu", type=str2bool, default=True, help="use CUDA when available")  # 是否使用GPU


def main():
    args = parser.parse_args()
    if not args.stage1_run_dir and not args.stage1_run_params:
        parser.error("--stage1_run_dir or --stage1_run_params is required")
    if args.generation_image_size % 8 != 0:
        parser.error("--generation_image_size must be divisible by 8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = str(Path(args.output_dir).expanduser() / timestamp)
    experiment = Exp_Stage3_Generation(args)
    with (Path(args.output_dir) / "run_params.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    experiment.train()
    experiment._evaluate(experiment.test_data, "test")
    if not args.generation_skip_sampling:
        experiment.sample_split(experiment.train_data, "train")
        experiment.sample_split(experiment.val_data, "val")
        experiment.sample_split(experiment.test_data, "test")


if __name__ == "__main__":
    main()

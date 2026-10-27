"""Align DA4FE EEG features with CLIP features from paired original images."""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from exp.exp_stage3a_image_alignment import Exp_Stage3A_ImageAlignment


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


parser = argparse.ArgumentParser(description="Stage-3A: align DA4FE EEG features with paired CLIP image features")

# DA4FE checkpoint and paired Hugging Face EEG-image dataset.
parser.add_argument("--stage1_run_dir", type=str_or_none, default="../result/DA4FE/stage1/20260918_144539", help="DA4FE run directory containing run_params.json, split_summary.json and checkpoint.pth")  # Stage-1参数目录
parser.add_argument("--stage1_checkpoint", type=str_or_none, default=None, help="optional DA4FE checkpoint path; defaults to checkpoint.pth in the run directory")  # Stage-1权重
parser.add_argument("--stage1_run_params", type=str_or_none, default=None, help="optional Stage-1 run_params.json path")  # Stage-1参数文件
parser.add_argument("--stage1_split_summary", type=str_or_none, default=None, help="optional Stage-1 split_summary.json path")  # 数据划分信息
parser.add_argument("--stage3a_resume_checkpoint", type=str_or_none, default=None, help="optional prior 3A checkpoint; use it to warm up the projection while DA4FE is frozen, then resume with DA4FE unfrozen")  # 继续微调3A
parser.add_argument("--eeg_hf_dataset_id", type=str_or_none, default=None, help="override the Hugging Face dataset id saved in Stage-1 parameters")  # HF数据集
parser.add_argument("--eeg_hf_cache_dir", type=str_or_none, default=None, help="override the Hugging Face dataset cache directory")  # 数据缓存
parser.add_argument("--eeg_hf_split_mode", choices=["original", "self"], default=None, help="override original/self split mode")  # 数据划分方式
parser.add_argument("--eeg_hf_split_seed", type=int, default=None, help="override the self-split seed")  # 划分随机种子
parser.add_argument("--eeg_hf_train_ratio", type=float, default=None, help="override the self-split train ratio")  # 训练集比例
parser.add_argument("--eeg_hf_val_ratio", type=float, default=None, help="override the self-split validation ratio")  # 验证集比例
parser.add_argument("--eeg_hf_test_ratio", type=float, default=None, help="override the self-split test ratio")  # 测试集比例

# Frozen CLIP image teacher.
parser.add_argument("--pretrained_root", type=str, default="../pretrained", help="root directory containing local pretrained models")  # 预训练权重目录
parser.add_argument("--clip_model_path", type=str_or_none, default=None, help="local CLIP directory or Hugging Face model id")  # CLIP权重路径
parser.add_argument("--hf_cache_dir", type=str_or_none, default=None, help="Hugging Face cache; None uses the system default")  # HF缓存目录
parser.add_argument("--allow_hf_download", type=str2bool, default=True, help="load from cache and download missing CLIP files when enabled")  # 是否允许下载

# Alignment model and optimization.
parser.add_argument("--stage3a_train_da4fe", type=str2bool, default=False, help="false freezes DA4FE and trains only the projection head; true fine-tunes DA4FE")  # 是否解冻DA4FE
parser.add_argument("--stage3a_epochs", type=int, default=30, help="number of EEG-image alignment epochs")  # 训练轮数
parser.add_argument("--stage3a_batch_size", type=int, default=16, help="alignment batch size")  # 批次大小
parser.add_argument("--stage3a_image_size", type=int, default=224, help="image size used by the CLIP teacher")  # CLIP输入尺寸
parser.add_argument("--stage3a_projection_hidden_dim", type=int, default=1024, help="hidden size of the EEG-to-CLIP projection head")  # 投影头隐藏维度
parser.add_argument("--stage3a_projection_dropout", type=float, default=0.1, help="dropout in the EEG-to-CLIP projection head")  # 投影头Dropout
parser.add_argument("--stage3a_projection_lr", type=float, default=1e-4, help="learning rate for the EEG-to-CLIP projection head")  # 投影头学习率
parser.add_argument("--stage3a_da4fe_lr", type=float, default=1e-6, help="learning rate for DA4FE when stage3a_train_da4fe is true")  # DA4FE微调学习率
parser.add_argument("--stage3a_temperature", type=float, default=0.07, help="temperature for multi-positive EEG-image contrastive loss")  # 对比损失温度
parser.add_argument("--stage3a_classification_weight", type=float, default=0.1, help="auxiliary DA4FE class-loss weight when DA4FE is unfrozen")  # 分类辅助损失权重
parser.add_argument("--stage3a_weight_decay", type=float, default=1e-2, help="AdamW weight decay")  # 权重衰减
parser.add_argument("--stage3a_grad_clip", type=float, default=1.0, help="gradient clipping norm")  # 梯度裁剪

# Runtime and output.
parser.add_argument("--output_dir", type=str, default="../result/generation_alignment", help="directory for the 3A checkpoint and metrics")  # 输出目录
parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")  # 数据加载进程数
parser.add_argument("--seed", type=int, default=42, help="random seed")  # 随机种子
parser.add_argument("--use_gpu", type=str2bool, default=True, help="use CUDA when available")  # 是否使用GPU


def main():
    args = parser.parse_args()
    if not args.stage1_run_dir and not args.stage1_run_params:
        parser.error("--stage1_run_dir or --stage1_run_params is required")
    if args.stage3a_epochs <= 0 or args.stage3a_batch_size <= 0:
        parser.error("--stage3a_epochs and --stage3a_batch_size must be positive")
    if args.stage3a_temperature <= 0:
        parser.error("--stage3a_temperature must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = str(Path(args.output_dir).expanduser() / timestamp)
    experiment = Exp_Stage3A_ImageAlignment(args)
    run_params = vars(args).copy()
    run_params["stage1_checkpoint_resolved"] = str(experiment.stage1_checkpoint_path.resolve())
    run_params["stage1_run_params_resolved"] = str(experiment.stage1_run_params_path.resolve())
    run_params["stage1_args"] = vars(experiment.stage1_args)
    with (Path(args.output_dir) / "run_params.json").open("w", encoding="utf-8") as handle:
        json.dump(run_params, handle, ensure_ascii=False, indent=2)
    experiment.train()


if __name__ == "__main__":
    main()

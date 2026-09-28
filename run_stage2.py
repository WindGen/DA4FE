import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import psutil
import torch

from exp.exp_stage2_classifier import Exp_Stage2_Classifier


STAGE1_PARAM_KEYS = (
    "model",
    "data",
    "root_path",
    "data_path",
    "eeg_hf_dataset_id",
    "eeg_hf_cache_dir",
    "eeg_hf_split_mode",
    "eeg_hf_split_seed",
    "eeg_hf_train_ratio",
    "eeg_hf_val_ratio",
    "eeg_hf_test_ratio",
    "seq_len",
    "requested_seq_len",
    "sampling_rate",
    "frequency_window",
    "frequency_normalization",
    "patch_len",
    "enc_in",
    "d_model",
    "n_heads",
    "t_layer",
    "v_layer",
    "f_layer",
    "da4fe_channel_dim",
    "da4fe_temporal_dim",
    "da4fe_frequency_dim",
    "da4fe_fusion_mode",
    "da4fe_fusion_hidden_dim",
    "da4fe_fusion_out_dim",
    "da4fe_channel_weight",
    "da4fe_temporal_weight",
    "da4fe_frequency_weight",
    "dropout",
    "augmentations",
    "eeg_normalize",
    "eeg_num_classes",
    "eeg_adaptive_seq_len",
    "stage1_l2_normalize",
    "seed",
)


def load_stage1_run_params(path):
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "run_params.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage-1 run_params.json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    saved_args = payload.get("args", payload)
    if not isinstance(saved_args, dict):
        raise ValueError(f"Expected an args object in {path}")
    return path, saved_args


def apply_stage1_run_params(args, path):
    path, saved_args = load_stage1_run_params(path)
    applied = []
    for key in STAGE1_PARAM_KEYS:
        if key in saved_args and saved_args[key] is not None:
            setattr(args, key, saved_args[key])
            applied.append(key)
    args.run_params_path = str(path)
    print(f"Loaded Stage-1 model/data parameters: {path}")
    print(f"Applied parameters: {', '.join(applied)}")
    return args


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def str_or_none(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"none", "null", ""}:
        return None
    return text


def build_stage2_full_setting(args):
    setting = (
        f"{args.model}_{args.data}_seed_{args.seed}_stage2_heads_{args.stage2_classifier_heads}"
        f"_std_{int(args.stage2_standardize_features)}"
        f"_l2norm_{int(args.stage1_l2_normalize)}_bs_{args.batch_size}"
    )
    if args.model == "DA4FE":
        setting += (
            f"_fs_{args.sampling_rate}"
            f"_fwin_{args.frequency_window}"
            f"_fnorm_{args.frequency_normalization}"
        )
    return setting


def build_run_directory_name(args, stage_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = timestamp
    result_root = getattr(args, "result_dir", None)
    if result_root is None:
        result_root = Path(__file__).resolve().parent.parent / "result"
    else:
        result_root = Path(result_root).expanduser()
    roots = [result_root / args.model / stage_name]
    if args.log_dir is not None:
        roots.append(Path(args.log_dir) / args.data / stage_name)

    suffix = 1
    while any((root / candidate).exists() for root in roots):
        candidate = f"{timestamp}_{suffix:02d}"
        suffix += 1
    return candidate


def use_cpus(gpus: list, cpus_per_gpu: int):
    available_cpus = list(range(psutil.cpu_count() or 1))
    requested_cpus = []
    for gpu in gpus:
        requested_cpus.extend(list(range(gpu * cpus_per_gpu, (gpu + 1) * cpus_per_gpu)))

    cpus = [cpu for cpu in requested_cpus if cpu in available_cpus]
    if not cpus:
        cpus = available_cpus

    cpus = sorted(set(cpus))
    process = psutil.Process()
    process.cpu_affinity(cpus)
    print(f"Using {len(cpus)} CPU(s) with affinity {cpus}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DA4FE/TeCh Stage-2 classifier training")
    parser.add_argument("--model", type=str, default="DA4FE", help="model name: [TeCh, DA4FE]")
    parser.add_argument("--data", type=str, default="EEG-ImageNet-HF", help="dataset type")
    parser.add_argument("--root_path", type=str, default=None, help="root path of local data files")
    parser.add_argument("--data_path", type=str, default="EEG-ImageNet", help="data file")
    parser.add_argument("--eeg_hf_dataset_id", type=str, default="luigi-s/EEG_Image_CVPR_ALL_subj")
    parser.add_argument("--eeg_hf_cache_dir", type=str_or_none, default=None)
    parser.add_argument("--run_params", "--run-params", type=str_or_none, default=None, help="Stage-1 run_params.json or its run directory; when provided, model/data parameters are loaded automatically")
    parser.add_argument("--eeg_hf_split_mode", "--eeg-hf-split-mode", type=str, choices=["original", "self"], default="original", help="Hugging Face split mode when run_params is not provided")
    parser.add_argument("--eeg_hf_split_seed", "--eeg-hf-split-seed", type=int, default=None, help="random seed for the Hugging Face self split")
    parser.add_argument("--eeg_hf_train_ratio", "--eeg-hf-train-ratio", type=float, default=0.6, help="self split train ratio")
    parser.add_argument("--eeg_hf_val_ratio", "--eeg-hf-val-ratio", type=float, default=0.2, help="self split validation ratio")
    parser.add_argument("--eeg_hf_test_ratio", "--eeg-hf-test-ratio", type=float, default=0.2, help="self split test ratio")

    parser.add_argument("--seq_len", type=int, default=512, help="input sequence length")
    parser.add_argument(
        "--sampling_rate",
        type=float,
        default=None,
        help=(
            "EEG sampling rate in Hz; required when the DA4FE "
            "frequency branch is enabled"
        ),
    )
    parser.add_argument(
        "--frequency_window",
        type=str,
        default="hann",
        choices=["hann", "rectangular"],
        help="window used before rFFT in the DA4FE frequency branch",
    )
    parser.add_argument(
        "--frequency_normalization",
        type=str,
        default="relative",
        choices=["relative", "physical"],
        help=(
            "frequency normalization: relative power (recommended) "
            "or physical PSD"
        ),
    )
    parser.add_argument("--patch_len", type=int, default=4, help="cross-channel patch length")
    parser.add_argument("--enc_in", type=int, default=128, help="encoder input size")
    parser.add_argument("--d_model", type=int, default=256, help="model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="number of heads")
    parser.add_argument("--t_layer", type=int, default=2, help="temporal encoder layers")
    parser.add_argument("--v_layer", type=int, default=2, help="channel encoder layers")
    parser.add_argument("--f_layer", type=int, default=2, help="frequency encoder layers")
    parser.add_argument("--da4fe_channel_dim", type=int, default=128)
    parser.add_argument("--da4fe_temporal_dim", type=int, default=128)
    parser.add_argument("--da4fe_frequency_dim", type=int, default=128)
    parser.add_argument("--da4fe_fusion_mode", type=str, default="concat_mlp", choices=["add", "concat_mlp"])
    parser.add_argument("--da4fe_fusion_hidden_dim", type=int, default=256)
    parser.add_argument("--da4fe_fusion_out_dim", type=int, default=256)
    parser.add_argument("--da4fe_channel_weight", type=float, default=1.0)
    parser.add_argument("--da4fe_temporal_weight", type=float, default=1.0)
    parser.add_argument("--da4fe_frequency_weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--augmentations", type=str, default="none")
    parser.add_argument("--eeg_normalize", type=str2bool, default=False)
    parser.add_argument("--eeg_num_classes", type=int, default=0)
    parser.add_argument("--eeg_adaptive_seq_len", type=str2bool, default=True)

    # Feature preprocessing before Stage-2 classifiers.
    parser.add_argument("--stage1_l2_normalize", "--stage1-l2-normalize", type=str2bool, default=True, help="L2-normalize Stage-1 features before Stage-2 classifiers")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_dir", type=str_or_none, default=None)
    parser.add_argument(
        "--result_dir",
        type=str_or_none,
        default=None,
        help="root directory for model outputs; default is ../result next to the source tree",
    )
    parser.add_argument("--resume_ckpt", type=str_or_none, required=True, help="stage1 checkpoint path")

    parser.add_argument(
        "--stage2_classifier_heads",
        type=str,
        default="reference",
        help=(
            "classifier heads for stage2. Options: linear_svm,rbf_svm,poly_svm,"
            "logreg,mlp,reference,all"
        ),
    )
    parser.add_argument("--stage2_standardize_features", type=str2bool, default=True)
    parser.add_argument("--stage2_svm_c", type=float, default=1.0)
    parser.add_argument("--stage2_svm_gamma", type=float, default=0.5)
    parser.add_argument("--stage2_svm_degree", type=int, default=3)
    parser.add_argument("--stage2_logreg_c", type=float, default=1.0)
    parser.add_argument("--stage2_mlp_hidden_dims", type=str, default="256")
    parser.add_argument("--stage2_mlp_alpha", type=float, default=1e-4)
    parser.add_argument("--stage2_mlp_lr", type=float, default=1e-3)
    parser.add_argument("--stage2_max_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="random seed; run_params value is used when run_params is provided")

    parser.add_argument("--gpu_idx", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    args = parser.parse_args()
    if args.run_params is not None:
        args = apply_stage1_run_params(args, args.run_params)
    effective_f_layer = (
        args.f_layer if args.f_layer is not None else args.t_layer
    )
    if (
        args.model == "DA4FE"
        and effective_f_layer > 0
        and (args.sampling_rate is None or args.sampling_rate <= 0)
    ):
        parser.error(
            "--sampling_rate must be a positive value when "
            "DA4FE frequency branch is enabled"
        )
    split_ratios = np.asarray(
        [args.eeg_hf_train_ratio, args.eeg_hf_val_ratio, args.eeg_hf_test_ratio],
        dtype=np.float64,
    )
    if np.any(split_ratios <= 0) or not np.isclose(split_ratios.sum(), 1.0, atol=1e-6):
        parser.error("--eeg_hf_train_ratio, --eeg_hf_val_ratio and --eeg_hf_test_ratio must be positive and sum to 1.0")
    if args.result_dir is None:
        args.result_dir = str(Path(__file__).resolve().parent.parent / "result")
    if getattr(args, "requested_seq_len", None) in (None, 0):
        args.requested_seq_len = args.seq_len
    if args.eeg_hf_split_seed is None:
        args.eeg_hf_split_seed = args.seed
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    active_gpus = args.gpu_idx if args.use_multi_gpu else [args.gpu]
    use_cpus(gpus=active_gpus, cpus_per_gpu=12)

    seed = args.seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if args.model != "TCN":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    args.seed = seed
    args.full_setting_name = build_stage2_full_setting(args)
    args.run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    setting = build_run_directory_name(args, stage_name="stage2")
    args.run_directory_name = setting

    exp = Exp_Stage2_Classifier(args)
    print(f">>>>>>>stage2 classifier training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Full setting:", args.full_setting_name)
    exp.train(setting)

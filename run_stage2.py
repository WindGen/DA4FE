import argparse
import os
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import psutil
import torch

from exp.exp_stage2_classifier import Exp_Stage2_Classifier


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

    parser.add_argument("--gpu_idx", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    args = parser.parse_args()
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
    if args.result_dir is None:
        args.result_dir = str(Path(__file__).resolve().parent.parent / "result")
    args.requested_seq_len = args.seq_len
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    active_gpus = args.gpu_idx if args.use_multi_gpu else [args.gpu]
    use_cpus(gpus=active_gpus, cpus_per_gpu=12)

    seed = 42
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

import argparse
import os
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import psutil
import torch

from exp.exp_stage1_feature import Exp_Stage1_Feature


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


def build_stage1_full_setting(args):
    setting = (
        f"{args.model}_{args.data}_seed_{args.seed}_dm_{args.d_model}_dp_{args.dropout}"
        f"_tl_{args.t_layer}_vl_{args.v_layer}_fl_{args.f_layer}_bs_{args.batch_size}"
        f"_lr{args.learning_rate}_aug_{args.augmentations}_pl_{args.patch_len}"
        f"_loss_{args.stage1_loss}_trip_{args.stage1_triplet_type}_tm_{args.stage1_triplet_margin}"
        f"_cew_{args.stage1_ce_weight}_tw_{args.stage1_triplet_weight}"
        f"_useval_{int(getattr(args, 'use_validation', True))}"
        f"_ls_{args.stage1_label_smoothing}"
        f"_afs_{args.stage1_arcface_s}_afm_{args.stage1_arcface_m}"
        f"_cfs_{args.stage1_cosface_s}_cfm_{args.stage1_cosface_m}"
        f"_eegnorm_{int(args.eeg_normalize)}_eegcls_{args.eeg_num_classes}"
    )
    if args.model == "DA4FE":
        setting += (
            f"_fus_{args.da4fe_fusion_mode}_cd_{args.da4fe_channel_dim}"
            f"_td_{args.da4fe_temporal_dim}_fd_{args.da4fe_frequency_dim}"
            f"_fod_{args.da4fe_fusion_out_dim}"
        )
    if args.eeg_adaptive_seq_len and args.requested_seq_len > 0:
        setting += f"_sl_{args.requested_seq_len}"
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
    parser = argparse.ArgumentParser(description="DA4FE/TeCh Stage-1 feature training")
    parser.add_argument("--model", type=str, default="DA4FE", help="model name: [TeCh, DA4FE]")
    parser.add_argument("--data", type=str, default="EEG-ImageNet-HF", help="dataset type")
    parser.add_argument("--root_path", type=str, default=None, help="root path of local data files")
    parser.add_argument("--data_path", type=str, default="EEG-ImageNet", help="data file")
    parser.add_argument("--eeg_hf_dataset_id", type=str, default="luigi-s/EEG_Image_CVPR_ALL_subj")
    parser.add_argument("--eeg_hf_cache_dir", type=str_or_none, default=None)

    parser.add_argument("--seq_len", type=int, default=512, help="input sequence length")
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

    parser.add_argument(
        "--augmentations",
        type=str,
        default="flip0.8,frequency0.,jitter0.,mask0.0,channel0.4,drop0.0",
    )
    parser.add_argument("--eeg_normalize", type=str2bool, default=False)
    parser.add_argument("--eeg_num_classes", type=int, default=0)
    parser.add_argument("--eeg_adaptive_seq_len", type=str2bool, default=True)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_dir", type=str_or_none, default=None)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument(
        "--stage1_loss",
        type=str,
        default="triplet",
        choices=[
            "triplet",
            "ce",
            "ce_triplet",
            "arcface",
            "arcface_triplet",
            "cosface",
            "cosface_triplet",
        ],
        help="stage1 loss type",
    )
    parser.add_argument(
        "--stage1_triplet_type",
        type=str,
        default="semihard",
        choices=["semihard", "batch_hard"],
        help="triplet miner type when stage1 loss includes triplet",
    )
    parser.add_argument(
        "--stage1_triplet_margin",
        type=float,
        default=0.2,
        help="margin used by stage1 triplet loss",
    )
    parser.add_argument(
        "--stage1_ce_weight",
        type=float,
        default=1.0,
        help="weight of cross-entropy style loss when stage1 loss includes classification supervision",
    )
    parser.add_argument(
        "--stage1_triplet_weight",
        type=float,
        default=1.0,
        help="weight of triplet loss when stage1 loss includes triplet supervision",
    )
    parser.add_argument(
        "--stage1_label_smoothing",
        type=float,
        default=0.0,
        help="label smoothing used by CE, ArcFace, and CosFace losses",
    )
    parser.add_argument(
        "--stage1_arcface_s",
        type=float,
        default=30.0,
        help="ArcFace scale parameter",
    )
    parser.add_argument(
        "--stage1_arcface_m",
        type=float,
        default=0.5,
        help="ArcFace angular margin",
    )
    parser.add_argument(
        "--stage1_cosface_s",
        type=float,
        default=30.0,
        help="CosFace scale parameter",
    )
    parser.add_argument(
        "--stage1_cosface_m",
        type=float,
        default=0.35,
        help="CosFace cosine margin",
    )
    parser.add_argument(
        "--stage1_kmeans_clusters",
        type=int,
        default=0,
        help="number of KMeans clusters; 0 uses dataset class count",
    )
    parser.add_argument(
        "--stage1_samples_per_class",
        type=int,
        default=2,
        help="number of samples per class inside each stage1 batch when triplet loss is used",
    )
    parser.add_argument("--resume_ckpt", type=str_or_none, default=None)
    parser.add_argument("--keep_checkpoint", type=str2bool, default=True)
    parser.add_argument(
        "--use_validation",
        "--use_val",
        type=str2bool,
        default=True,
        help=(
            "whether to keep a validation split for checkpoint selection; "
            "when false, validation samples are merged into training and test "
            "KMeans selects the best checkpoint"
        ),
    )
    parser.add_argument(
        "--result_dir",
        type=str_or_none,
        default=None,
        help="root directory for outputs; default is ../result next to the source tree",
    )

    parser.add_argument("--lradj", type=str, default="cosine")
    parser.add_argument("--use_amp", action="store_true", default=False)

    parser.add_argument("--gpu_idx", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    args = parser.parse_args()
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

    Exp = Exp_Stage1_Feature
    test_kmeans_scores = []

    for ii in range(args.itr):
        seed = 42 + ii
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
        args.full_setting_name = build_stage1_full_setting(args)
        args.run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        setting = build_run_directory_name(args, stage_name="stage1")
        args.run_directory_name = setting

        exp = Exp(args)
        print(f">>>>>>>stage1 feature training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Full setting:", args.full_setting_name)
        exp.train(setting)

        print(f">>>>>>>stage1 feature evaluation : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        stage1_metrics = exp.test(setting)
        test_kmeans_scores.append(stage1_metrics["TestKMeans"])
        torch.cuda.empty_cache()

    mean_test_kmeans = float(np.mean(test_kmeans_scores))
    std_test_kmeans = float(np.std(test_kmeans_scores))
    print(f"Stage1 mean test KMeans: {mean_test_kmeans:.4f}")
    print(f"Stage1 std test KMeans: {std_test_kmeans:.4f}")

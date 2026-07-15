import argparse
import os
from datetime import datetime
from pathlib import Path
import torch
from exp.exp_classification import Exp_Classification
import random
import numpy as np
import psutil


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


def build_full_setting(args):
    setting = "{}_{}_seed_{}_dm_{}_dp_{}_tl_{}_vl_{}_bs_{}_lr{}_aug_{}_pl_{}".format(
        args.model,
        args.data,
        args.seed,
        args.d_model,
        args.dropout,
        args.t_layer,
        args.v_layer,
        args.batch_size,
        args.learning_rate,
        args.augmentations,
        args.patch_len,
    )
    if args.model == "DA4FE":
        effective_f_layer = args.f_layer if args.f_layer is not None else args.t_layer
        effective_channel_dim = (
            args.da4fe_channel_dim if args.da4fe_channel_dim is not None else args.d_model
        )
        effective_temporal_dim = (
            args.da4fe_temporal_dim if args.da4fe_temporal_dim is not None else args.d_model
        )
        effective_frequency_dim = (
            args.da4fe_frequency_dim if args.da4fe_frequency_dim is not None else args.d_model
        )
        effective_fusion_out_dim = (
            args.da4fe_fusion_out_dim if args.da4fe_fusion_out_dim is not None else args.d_model
        )
        setting += f"_fl_{effective_f_layer}"
        setting += f"_fus_{args.da4fe_fusion_mode}"
        setting += (
            f"_cd_{effective_channel_dim}"
            f"_td_{effective_temporal_dim}"
            f"_fd_{effective_frequency_dim}"
            f"_fod_{effective_fusion_out_dim}"
        )
        if args.da4fe_fusion_mode == "add":
            setting += (
                f"_cw_{args.da4fe_channel_weight}"
                f"_tw_{args.da4fe_temporal_weight}"
                f"_fw_{args.da4fe_frequency_weight}"
            )
        if (
            args.da4fe_fusion_mode == "concat_mlp"
            and args.da4fe_fusion_hidden_dim is not None
        ):
            setting += f"_fhd_{args.da4fe_fusion_hidden_dim}"

    setting += f"_loss_{args.loss}"
    if args.loss == "ce_triplet":
        setting += f"_tm_{args.triplet_margin}"
        setting += f"_tw_{args.triplet_weight}"

    setting += f"_eegnorm_{int(args.eeg_normalize)}"
    setting += f"_eegcls_{args.eeg_num_classes}"
    effective_eeg_adaptive = args.eeg_adaptive_seq_len and args.requested_seq_len > 0
    setting += f"_eegadapt_{int(effective_eeg_adaptive)}"
    if effective_eeg_adaptive:
        setting += f"_sl_{args.requested_seq_len}"

    return setting


def build_run_directory_name(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = timestamp
    roots = [Path("./checkpoints") / args.model]
    if args.log_dir is not None:
        roots.append(Path(args.log_dir) / args.data)

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
        p = psutil.Process()
        p.cpu_affinity(cpus)
        print(
            "Using {} CPU(s) with affinity {}.".format(
                len(cpus), cpus
            )
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TeCh")
    parser.add_argument(
        "--model",
        type=str,
        # required=True,
        default="DA4FE",
        help="model name, options: [TeCh, DA4FE]",
    )

    # data loader
    parser.add_argument(
        "--data", type=str, #required=True, 
        default="EEG-ImageNet-HF", help="dataset type, e.g. EEG-ImageNet or EEG-ImageNet-HF"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default=None,
        help="root path of the local data files; ignored by EEG-ImageNet-HF unless used manually as a cache location",
    )
    parser.add_argument("--data_path", type=str, default="EEG-ImageNet", help="data file")
    parser.add_argument(
        "--eeg_hf_dataset_id",
        type=str,
        default="luigi-s/EEG_Image_CVPR_ALL_subj",
        help="Hugging Face dataset id used when --data EEG-ImageNet-HF",
    )
    parser.add_argument(
        "--eeg_hf_cache_dir",
        type=str_or_none,
        default=None,
        help="optional cache directory for Hugging Face EEG-ImageNet downloads",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=512, help="input sequence length") # 根据数据自动设置覆盖
    # model define for baselines
    parser.add_argument("--patch_len", type=int, default=4, help="for cross_channel pacthing")
    parser.add_argument("--enc_in", type=int, default=128, help="encoder input size") # 根据数据自动设置覆盖
    parser.add_argument("--d_model", type=int, default=256, help="dimension of model")  #512
    parser.add_argument("--n_heads", type=int, default=4, help="num of heads")  # 8
    parser.add_argument("--t_layer", type=int, default=2, help="num of encoder layers")  #6
    parser.add_argument("--v_layer", type=int, default=2, help="num of encoder layers")  #6
    parser.add_argument("--f_layer", type=int, default=2,help="num of frequency encoder layers for DA4FE; None follows t_layer",)
    parser.add_argument("--da4fe_channel_dim",type=int,default=128,help="channel-branch embedding/encoder dim for DA4FE; None follows d_model",)
    parser.add_argument("--da4fe_temporal_dim",type=int,default=128,help="temporal-branch embedding/encoder dim for DA4FE; None follows d_model",)
    parser.add_argument("--da4fe_frequency_dim",type=int,default=128,help="frequency-branch embedding/encoder dim for DA4FE; None follows d_model",)
    parser.add_argument("--da4fe_fusion_mode",type=str,default="concat_mlp",choices=["add", "concat_mlp"],help="feature fusion mode for DA4FE branches",)
    parser.add_argument("--da4fe_fusion_hidden_dim",type=int,default=256,help="hidden dim of DA4FE fusion MLP; None follows d_model",)
    parser.add_argument("--da4fe_fusion_out_dim",type=int,default=256,help="output dim of DA4FE branch fusion; None follows d_model",)
    
    parser.add_argument( "--da4fe_channel_weight",type=float,default=1,help="channel-branch weight used by DA4FE when fusion mode is add",)
    parser.add_argument("--da4fe_temporal_weight",type=float,default=1,help="temporal-branch weight used by DA4FE when fusion mode is add",)
    parser.add_argument("--da4fe_frequency_weight",type=float,default=1,help="frequency-branch weight used by DA4FE when fusion mode is add",)
    
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
    
    # Augmentation
    parser.add_argument(
        "--augmentations",
        type=str,
        default="flip0.8,frequency0.,jitter0.,mask0.0,channel0.4,drop0.0",
        help="A comma-seperated list of augmentation types (none, jitter or scale). "
             "Randomly applied to each granularity. "
             "Append numbers to specify the strength of the augmentation, e.g., jitter0.1",
    )
    parser.add_argument(
        "--eeg_normalize",
        type=str2bool,
        default=False,
        help="whether to normalize each EEG-ImageNet sample channel-wise over time",
    )
    parser.add_argument(
        "--eeg_num_classes",
        type=int,
        default=0,
        help="number of EEG-ImageNet classes to use; 0 means all classes",
    )
    parser.add_argument(
        "--eeg_adaptive_seq_len",
        type=str2bool,
        default=True,
        help=(
            "whether to resample each EEG-ImageNet sample to --seq_len; "
            "False keeps the original variable-length behavior"
        ),
    )
    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=4, help="data loader num workers"
    )
    parser.add_argument(
        "--log_dir",
        type=str_or_none,
        default=None,
        help=(
            "root directory for training metric logs; use None to save logs "
            "inside the corresponding checkpoints/<model>/<setting> folder"
        ),
    )

    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=500, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=16, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="optimizer learning rate"
    )


    parser.add_argument(
        "--loss",
        type=str,
        default="ce_triplet",
        choices=["ce", "ce_triplet"],
        help="training objective: ce or ce_triplet",
    )
    parser.add_argument(
        "--triplet_margin",
        type=float,
        default=0.2,
        help="margin used by batch-hard triplet loss when --loss ce_triplet",
    )
    parser.add_argument(
        "--triplet_weight",
        type=float,
        default=0.2,
        help="weight of triplet loss in the total objective when --loss ce_triplet",
    )


    parser.add_argument(
        "--lradj", type=str, default="cosine", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--gpu_idx", nargs="+", type=int, default=[0,1,2,3,4,5,6,7], help="List of GPU indices to use")
    parser.add_argument("--use_gpu", type=str2bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multiple gpus"
    )
    # parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')

    args = parser.parse_args()
    args.requested_seq_len = args.seq_len
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # For server training this can reserve CPU groups per GPU, while local runs
    # fall back to the CPUs that actually exist on the machine.
    active_gpus = args.gpu_idx if args.use_multi_gpu else [args.gpu]
    use_cpus(gpus=active_gpus, cpus_per_gpu=12)
    
    # print("Args in experiment:")
    # print(args)

    Exp = Exp_Classification
    avg_metrics=[]
    means=[]
    stds=[]
    # 42
    for ii in range(args.itr):
            seed = 42 + ii
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            if args.model != "TCN":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True


            # setting record of experiments
            args.seed = seed
            args.full_setting_name = build_full_setting(args)
            args.run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            setting = build_run_directory_name(args)
            args.run_directory_name = setting

            exp = Exp(args)  # set experiments
            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            print("Full setting:", args.full_setting_name)
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            avg_metrics.append(exp.test(setting))
            torch.cuda.empty_cache()
            
    means=[np.mean([avg_metrics[i][j] for i in range(args.itr)]) for j in ('Accuracy', 'Precision', 'Recall', 'F1', 'AUROC','AUPRC')]
    stds=[np.std([avg_metrics[i][j] for i in range(args.itr)]) for j in ('Accuracy', 'Precision', 'Recall', 'F1', 'AUROC','AUPRC')]
    print(f'Mean accuracy: {means[0]:.4f}, precision: {means[1]:.4f},recall: {means[2]:.4f}, f1: {means[3]:.4f}, AUROC: {means[4]:.4f}, AUPRC: {means[5]:.4f}')
    print(f'Std accuracy: {stds[0]:.4f}, precision: {stds[1]:.4f},recall: {stds[2]:.4f}, f1: {stds[3]:.4f}, AUROC: {stds[4]:.4f}, AUPRC: {stds[5]:.4f}')

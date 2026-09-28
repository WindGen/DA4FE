import copy
import io
import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import (normalize_batch_ts,bandpass_filter_func)
import warnings
import random
import urllib.request
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from natsort import natsorted

try:
    from PIL import Image
except ImportError:  # pragma: no cover - only required by Stage-3 generation
    Image = None

try:
    from datasets import DatasetDict, concatenate_datasets, load_dataset
except ImportError:  # pragma: no cover - optional runtime dependency
    DatasetDict = None
    concatenate_datasets = None
    load_dataset = None

warnings.filterwarnings("ignore")

class APAVALoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        data_list = np.load(self.label_path)

        all_ids = list(data_list[:, 1])  # id of all samples
        val_ids = [15, 16, 19, 20]  # 15, 19 are AD; 16, 20 are HC
        test_ids = [1, 2, 17, 18]  # 1, 17 are AD; 2, 18 are HC
        train_ids = [int(i) for i in all_ids if i not in val_ids + test_ids]
        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

        self.X, self.y = self.load_apava(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=256, lowcut=0.5, highcut=45)

        self.max_seq_len = self.X.shape[1]

    def load_apava(self, data_path, label_path, flag=None):
        """
        Loads APAVA data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            # print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            # print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            # print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            # print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class TDBRAINLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        train_ids = list(range(1, 18)) + list(
            range(29, 46)
        )  # specify patient ID for training, validation, and test set
        val_ids = [18, 19, 20, 21] + [46, 47, 48, 49]  # 8 patients, 4 positive 4 healthy
        test_ids = [22, 23, 24, 25] + [50, 51, 52, 53]  # 8 patients, 4 positive 4 healthy

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

        self.X, self.y = self.load_tdbrain(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=256, lowcut=0.5, highcut=45)

        self.max_seq_len = self.X.shape[1]

    def load_tdbrain(self, data_path, label_path, flag=None):
        """
        Loads tdbrain data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            # print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            # print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            # print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            # print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class ADFTDLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )
        self.X, self.y = self.load_adfd(self.data_path, self.label_path, flag=flag)

        # pre_process
        # self.X = bandpass_filter_func(self.X, fs=256, lowcut=0.5, highcut=45)
        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ftd_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Frontotemporal Dementia IDs
        ad_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # Alzheimer's disease IDs

        train_ids = (
            cn_list[: int(a * len(cn_list))]
            + ftd_list[: int(a * len(ftd_list))]
            + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
            cn_list[int(a * len(cn_list)) : int(b * len(cn_list))]
            + ftd_list[int(a * len(ftd_list)) : int(b * len(ftd_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
        )
        test_ids = (
            cn_list[int(b * len(cn_list)) :]
            + ftd_list[int(b * len(ftd_list)) :]
            + ad_list[int(b * len(ad_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_adfd(self, data_path, label_path, flag=None):
        """
        Loads adfd or cnbpm data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            # print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            # print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            # print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            # print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class PTBLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.55, 0.7

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_ptb(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=250, lowcut=0.5, highcut=45)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(label_path)
        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        my_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Myocardial infarction IDs

        train_ids = hc_list[: int(a * len(hc_list))] + my_list[: int(a * len(my_list))]
        val_ids = (
            hc_list[int(a * len(hc_list)) : int(b * len(hc_list))]
            + my_list[int(a * len(my_list)) : int(b * len(my_list))]
        )
        test_ids = hc_list[int(b * len(hc_list)) :] + my_list[int(b * len(my_list)) :]

        return train_ids, val_ids, test_ids

    def load_ptb(self, data_path, label_path, flag=None):
        """
        Loads ptb data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            # print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            # print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            # print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            # print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class PTBXLLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_ptbxl(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=250, lowcut=0.5, highcut=45)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(label_path)
        no_list = list(
            data_list[np.where(data_list[:, 0] == 0)][:, 1]
        )  # Normal ECG IDs
        mi_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Myocardial Infarction IDs
        sttc_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # ST/T Change IDs
        cd_list = list(
            data_list[np.where(data_list[:, 0] == 3)][:, 1]
        )  # Conduction Disturbance IDs
        hyp_list = list(
            data_list[np.where(data_list[:, 0] == 4)][:, 1]
        )  # Hypertrophy IDs

        train_ids = (
            no_list[: int(a * len(no_list))]
            + mi_list[: int(a * len(mi_list))]
            + sttc_list[: int(a * len(sttc_list))]
            + cd_list[: int(a * len(cd_list))]
            + hyp_list[: int(a * len(hyp_list))]
        )
        val_ids = (
            no_list[int(a * len(no_list)) : int(b * len(no_list))]
            + mi_list[int(a * len(mi_list)) : int(b * len(mi_list))]
            + sttc_list[int(a * len(sttc_list)) : int(b * len(sttc_list))]
            + cd_list[int(a * len(cd_list)) : int(b * len(cd_list))]
            + hyp_list[int(a * len(hyp_list)) : int(b * len(hyp_list))]
        )
        test_ids = (
            no_list[int(b * len(no_list)) :]
            + mi_list[int(b * len(mi_list)) :]
            + sttc_list[int(b * len(sttc_list)) :]
            + cd_list[int(b * len(cd_list)) :]
            + hyp_list[int(b * len(hyp_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_ptbxl(self, data_path, label_path, flag=None):
        """
        Loads ptb-xl data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            # print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            # print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            # print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            # print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)



class FLAAPLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/feature.npy')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        self.X, self.y = self.load_flaap_dependent(self.data_path, self.label_path, flag=flag)

        # pre_process
        # self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_flaap_dependent(self, data_path, label_path, flag=None):
        '''
        Loads fl-aap data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        X_train = np.load(data_path)
        y_train = np.load(label_path)
        # print(X_train.shape, y_train.shape)

        # 60 : 20 : 20
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        if flag == 'TRAIN':
            return X_train, y_train
        elif flag == 'VAL':
            return X_val, y_val
        elif flag == 'TEST':
            return X_test, y_test
        else:
            raise Exception('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
            torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class UCIHARLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/feature.npy')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        self.X, self.y = self.load_har_dependent(self.data_path, self.label_path, flag=flag)

        # pre_process
        # self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_har_dependent(self, data_path, label_path, flag=None):
        '''
        Loads fl-aap data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        X_train = np.load(data_path)
        y_train = np.load(label_path)
        # print(X_train.shape, y_train.shape)

        X_test = X_train[-2947:]
        y_test = y_train[-2947:]

        X_train, X_val, y_train, y_val = train_test_split(X_train[:-2947], y_train[:-2947], test_size=0.2, random_state=42)
        # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        if flag == 'TRAIN':
            return X_train, y_train
        elif flag == 'VAL':
            return X_val, y_val
        elif flag == 'TEST':
            return X_test, y_test
        else:
            raise Exception('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
            torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class EEGImageNetHuggingFaceLoader(Dataset):
    _split_cache = {}
    _class_name_cache = {}
    _metadata_cache = {}
    _all_dataset_cache = {}
    _custom_split_cache = {}

    def __init__(self, args, root_path, flag=None):
        if load_dataset is None:
            raise ImportError(
                "The Hugging Face EEG-ImageNet loader requires the `datasets` package. "
                "Install it with `pip install datasets` or reinstall from requirements.txt."
            )

        self.args = args
        self.root_path = root_path
        self.flag = str(flag).upper() if flag else "TRAIN"
        self.split_mode = str(
            getattr(args, "eeg_hf_split_mode", "original")
        ).strip().lower()
        if self.split_mode in {"hf", "huggingface", "native"}:
            self.split_mode = "original"
        elif self.split_mode in {"custom", "self_split", "self-split"}:
            self.split_mode = "self"
        if self.split_mode not in {"original", "self"}:
            raise ValueError(
                "eeg_hf_split_mode must be 'original' or 'self', "
                f"got {self.split_mode!r}"
            )
        self.normalize = getattr(args, "eeg_normalize", True)
        self.max_classes = getattr(args, "eeg_num_classes", 0)
        self.split_seed = getattr(args, "eeg_hf_split_seed", None)
        if self.split_seed is None:
            self.split_seed = getattr(args, "seed", 42)
        self.split_seed = int(self.split_seed)
        self.split_ratios = np.asarray(
            [
                float(getattr(args, "eeg_hf_train_ratio", 0.6)),
                float(getattr(args, "eeg_hf_val_ratio", 0.2)),
                float(getattr(args, "eeg_hf_test_ratio", 0.2)),
            ],
            dtype=np.float64,
        )
        if np.any(self.split_ratios <= 0) or not np.isclose(
            self.split_ratios.sum(), 1.0, atol=1e-6
        ):
            raise ValueError(
                "eeg_hf_train_ratio, eeg_hf_val_ratio and "
                "eeg_hf_test_ratio must be positive and sum to 1.0; "
                f"got {self.split_ratios.tolist()}"
            )
        self.target_seq_len = getattr(
            args, "requested_seq_len", getattr(args, "seq_len", None)
        )
        self.adaptive_seq_len = (
            getattr(args, "eeg_adaptive_seq_len", False)
            and self.target_seq_len is not None
            and self.target_seq_len > 0
        )
        self.dataset_id = getattr(
            args, "eeg_hf_dataset_id", "luigi-s/EEG_Image_CVPR_ALL_subj"
        )
        self.cache_dir = getattr(args, "eeg_hf_cache_dir", None)
        self.split_name = self._resolve_split_name(self.flag)

        self.class_names = self._resolve_class_names()
        self.label_to_index = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }
        self.num_class = len(self.class_names)

        if self.split_mode == "self":
            raw_dataset = self._load_custom_split(self.flag)
        else:
            raw_dataset = self._load_split(self.split_name)
        self.dataset = self._prepare_dataset(raw_dataset)
        self.image_column = self._resolve_image_column(self.dataset)
        self.max_seq_len, self.feature_dim = self._resolve_dataset_metadata()
        class_column = (
            self.dataset["label_folder"]
            if "label_folder" in self.dataset.column_names
            else [str(label_idx) for label_idx in self.dataset["label"]]
        )
        self.y = np.asarray(
            [self.label_to_index[str(class_name)] for class_name in class_column],
            dtype=np.int64,
        )

    def _resolve_split_name(self, flag):
        split_map = {
            "TRAIN": "train",
            "VAL": "validation",
            "VALIDATION": "validation",
            "TEST": "test",
        }
        if flag not in split_map:
            raise ValueError(
                f"Unsupported split flag {flag!r} for Hugging Face EEG-ImageNet"
            )
        return split_map[flag]

    def _load_split(self, split_name):
        cache_key = (self.dataset_id, split_name, self.cache_dir)
        cached = self._split_cache.get(cache_key)
        if cached is not None:
            return cached

        dataset = load_dataset(
            self.dataset_id,
            split=split_name,
            cache_dir=self.cache_dir,
        )
        self._split_cache[cache_key] = dataset
        return dataset

    def _load_all_dataset(self):
        """Load and concatenate all Hugging Face splits for self-splitting."""
        cache_key = (self.dataset_id, self.cache_dir)
        cached = self._all_dataset_cache.get(cache_key)
        if cached is not None:
            return cached

        if DatasetDict is None or concatenate_datasets is None:
            raise ImportError(
                "Self-splitting the Hugging Face EEG dataset requires the "
                "`datasets` package. Install it with `pip install datasets`."
            )

        dataset_dict = load_dataset(
            self.dataset_id,
            cache_dir=self.cache_dir,
        )
        if isinstance(dataset_dict, DatasetDict):
            split_datasets = [dataset_dict[name] for name in dataset_dict.keys()]
            if not split_datasets:
                raise RuntimeError(
                    f"No splits found in Hugging Face dataset {self.dataset_id}"
                )
            if len(split_datasets) == 1:
                dataset = split_datasets[0]
            else:
                dataset = concatenate_datasets(split_datasets)
        else:
            dataset = dataset_dict

        self._all_dataset_cache[cache_key] = dataset
        return dataset

    def _metadata_class_values(self, dataset):
        if "label_folder" in dataset.column_names:
            values = dataset["label_folder"]
        else:
            values = dataset["label"]
            label_feature = getattr(dataset, "features", {}).get("label")
            label_names = getattr(label_feature, "names", None)
            if label_names:
                values = [
                    label_names[int(value)]
                    if isinstance(value, (int, np.integer))
                    and 0 <= int(value) < len(label_names)
                    else value
                    for value in values
                ]
        return [str(value) for value in values]

    def _metadata_subject_values(self, dataset):
        if "subject" not in dataset.column_names:
            raise RuntimeError(
                "Self-splitting the Hugging Face EEG dataset requires a "
                "`subject` column."
            )
        return [str(value) for value in dataset["subject"]]

    def _split_subject_targets(self, num_subjects):
        """Return deterministic subject counts with nonempty splits."""
        if num_subjects < 3:
            raise RuntimeError(
                "Self-splitting requires at least 3 unique subjects, "
                f"got {num_subjects}"
            )

        ratios = self.split_ratios
        raw_targets = ratios * num_subjects
        targets = np.floor(raw_targets).astype(int)
        targets[targets == 0] = 1
        while targets.sum() > num_subjects:
            removable = np.where(targets > 1)[0]
            index = int(removable[np.argmax(targets[removable])])
            targets[index] -= 1
        fractions = raw_targets - np.floor(raw_targets)
        while targets.sum() < num_subjects:
            index = int(np.argmax(fractions))
            targets[index] += 1
            fractions[index] = -1.0
        return targets

    def _assign_subjects_to_splits(self, dataset):
        """Assign each subject once while approximately balancing class counts."""
        class_values = self._metadata_class_values(dataset)
        subjects = self._metadata_subject_values(dataset)
        class_names = list(self.class_names)
        class_to_index = {name: index for index, name in enumerate(class_names)}

        subject_class_counts = {}
        for class_name, subject in zip(class_values, subjects):
            if class_name not in class_to_index:
                continue
            if subject not in subject_class_counts:
                subject_class_counts[subject] = np.zeros(len(class_names), dtype=np.float64)
            subject_class_counts[subject][class_to_index[class_name]] += 1.0

        subject_names = list(subject_class_counts)
        if len(subject_names) < 3:
            raise RuntimeError(
                "Self-splitting requires at least 3 unique subjects, "
                f"got {len(subject_names)}"
            )

        rng = np.random.default_rng(self.split_seed)
        rng.shuffle(subject_names)
        subject_names.sort(
            key=lambda subject: float(subject_class_counts[subject].sum()),
            reverse=True,
        )

        split_names = ("train", "val", "test")
        subject_targets = self._split_subject_targets(len(subject_names))
        all_class_counts = np.sum(
            np.stack(list(subject_class_counts.values())), axis=0
        )
        sample_targets = all_class_counts[None, :] * np.asarray(
            self.split_ratios, dtype=np.float64
        )[:, None]
        assigned_subjects = [[] for _ in split_names]
        current_subject_counts = np.zeros(3, dtype=np.float64)
        current_class_counts = np.zeros((3, len(class_names)), dtype=np.float64)

        for subject in subject_names:
            subject_counts = subject_class_counts[subject]
            candidates = [
                index
                for index in range(3)
                if current_subject_counts[index] < subject_targets[index]
            ]
            if not candidates:
                candidates = list(range(3))

            scores = []
            for index in candidates:
                proposed = current_class_counts[index] + subject_counts
                class_error = np.sum(
                    ((proposed - sample_targets[index]) / np.maximum(sample_targets[index], 1.0)) ** 2
                )
                sample_error = (
                    (current_class_counts[index].sum() + subject_counts.sum())
                    / max(sample_targets[index].sum(), 1.0)
                    - 1.0
                ) ** 2
                subject_error = (
                    (current_subject_counts[index] + 1.0)
                    / max(float(subject_targets[index]), 1.0)
                    - 1.0
                ) ** 2
                scores.append(class_error + 0.10 * sample_error + 0.25 * subject_error)

            # Randomized candidate order makes equal-score assignments depend
            # only on the requested seed, while ``argmin`` remains deterministic.
            candidate_order = rng.permutation(len(candidates))
            best_position = min(
                candidate_order,
                key=lambda position: scores[int(position)],
            )
            selected_split = candidates[int(best_position)]
            assigned_subjects[selected_split].append(subject)
            current_subject_counts[selected_split] += 1
            current_class_counts[selected_split] += subject_counts

        subject_to_split = {
            subject: split_names[index]
            for index, split_subjects in enumerate(assigned_subjects)
            for subject in split_subjects
        }
        missing_classes = {
            split_names[index]: [
                class_names[class_index]
                for class_index, count in enumerate(current_class_counts[index])
                if count <= 0
            ]
            for index in range(3)
            if np.any(current_class_counts[index] <= 0)
        }
        if missing_classes:
            warnings.warn(
                "Self split could not place every class in every split because "
                "the class/subject support is insufficient: "
                f"{missing_classes}"
            )
        sample_split_names = [subject_to_split[subject] for subject in subjects]
        return sample_split_names, assigned_subjects, current_class_counts

    def _load_custom_split(self, flag):
        cache_key = (
            self.dataset_id,
            self.cache_dir,
            self.split_seed,
            tuple(np.round(self.split_ratios, 8)),
            tuple(self.class_names),
        )
        cached = self._custom_split_cache.get(cache_key)
        if cached is None:
            all_dataset = self._load_all_dataset()
            if self.max_classes:
                class_values = self._metadata_class_values(all_dataset)
                allowed_classes = set(self.class_names)
                keep_indices = [
                    index
                    for index, class_name in enumerate(class_values)
                    if class_name in allowed_classes
                ]
                all_dataset = all_dataset.select(keep_indices)

            sample_split_names, _assigned_subjects, _class_counts = self._assign_subjects_to_splits(
                all_dataset
            )
            cached = {
                split_name: all_dataset.select(
                    [
                        index
                        for index, assigned_split in enumerate(sample_split_names)
                        if assigned_split == split_name
                    ]
                )
                for split_name in ("train", "val", "test")
            }
            self._custom_split_cache[cache_key] = cached

        split_key = {
            "TRAIN": "train",
            "VAL": "val",
            "VALIDATION": "val",
            "TEST": "test",
        }[str(flag).upper()]
        return cached[split_key]

    def _resolve_class_names(self):
        cache_key = (
            self.dataset_id,
            self.cache_dir,
            self.max_classes,
            self.split_mode,
        )
        cached = self._class_name_cache.get(cache_key)
        if cached is not None:
            return cached

        reference_split = (
            self._load_all_dataset()
            if self.split_mode == "self"
            else self._load_split("train")
        )
        if "label_folder" in reference_split.column_names:
            class_names = natsorted(
                {str(class_name) for class_name in reference_split["label_folder"]}
            )
        else:
            class_names = [str(label_idx) for label_idx in sorted(set(reference_split["label"]))]

        if not class_names:
            raise RuntimeError(
                f"No classes found in Hugging Face dataset {self.dataset_id}"
            )

        if self.max_classes:
            if self.max_classes < 0:
                raise ValueError(
                    f"eeg_num_classes must be >= 0, got {self.max_classes}"
                )
            if self.max_classes > len(class_names):
                raise ValueError(
                    f"Requested {self.max_classes} classes, but only "
                    f"{len(class_names)} classes exist in {self.dataset_id}"
                )
            class_names = class_names[: self.max_classes]

        self._class_name_cache[cache_key] = class_names
        return class_names

    def _sample_class_name(self, sample):
        if "label_folder" in sample and sample["label_folder"] is not None:
            return str(sample["label_folder"])
        return str(sample["label"])

    def _label_index_from_sample(self, sample):
        class_name = self._sample_class_name(sample)
        try:
            return self.label_to_index[class_name]
        except KeyError as exc:
            raise KeyError(
                f"Class {class_name!r} is not part of the configured class set "
                f"for {self.dataset_id}"
            ) from exc

    def _prepare_dataset(self, dataset):
        if self.max_classes:
            allowed_classes = set(self.class_names)
            class_column = (
                dataset["label_folder"]
                if "label_folder" in dataset.column_names
                else [str(label_idx) for label_idx in dataset["label"]]
            )
            keep_indices = [
                idx for idx, class_name in enumerate(class_column) if str(class_name) in allowed_classes
            ]
            dataset = dataset.select(keep_indices)

        keep_columns = {"conditioning_image", "label", "subject", "label_folder"}
        keep_columns.update(self._image_related_columns(dataset))
        drop_columns = [
            column_name
            for column_name in dataset.column_names
            if column_name not in keep_columns
        ]
        if drop_columns:
            dataset = dataset.remove_columns(drop_columns)
        return dataset

    @staticmethod
    def _resolve_image_column(dataset, required=False):
        """Find the paired image column without assuming one HF feature name.

        The public EEG-ImageNet releases have used ``image``, ``image_raw`` and
        path-like names in different revisions.  We intentionally do not use a
        local image root or reconstruct an image from a class label: generation
        training requires the exact image paired with each EEG sample.
        """
        columns = list(getattr(dataset, "column_names", []))
        preferred = (
            "image",
            "image_raw",
            "target_image",
            "image_path",
            "image_filename",
            "image_file",
            "image_url",
        )
        for name in preferred:
            if name in columns:
                return name
        for name in columns:
            lower = str(name).lower()
            if lower in {"conditioning_image", "eeg", "eeg_signal", "signal"}:
                continue
            if "image" in lower or lower.endswith(("_jpg", "_jpeg", "_png")):
                return name
        if required:
            raise RuntimeError(
                "Hugging Face dataset does not contain an image field. "
                "Exact EEG-image generation requires paired image data."
            )
        return None

    @staticmethod
    def _image_related_columns(dataset):
        columns = list(getattr(dataset, "column_names", []))
        return [
            name
            for name in columns
            if str(name).lower() not in {"conditioning_image", "eeg", "eeg_signal", "signal"}
            and (
                "image" in str(name).lower()
                or str(name).lower().endswith(("_jpg", "_jpeg", "_png"))
            )
        ]

    @staticmethod
    def _to_pil_image(value):
        """Convert common Hugging Face image feature representations to PIL."""
        if Image is None:
            raise ImportError(
                "Pillow is required for Stage-3 image generation; install Pillow."
            )
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            if value.get("path"):
                return Image.open(value["path"]).convert("RGB")
        if isinstance(value, (bytes, bytearray, memoryview)):
            return Image.open(io.BytesIO(bytes(value))).convert("RGB")
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)):
            value = np.asarray(value)
        if isinstance(value, np.ndarray):
            array = value
            if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
                array = np.moveaxis(array, 0, -1)
            if array.dtype != np.uint8:
                array = np.clip(array, 0.0, 1.0) * 255.0 if array.max() <= 1.0 else np.clip(array, 0.0, 255.0)
                array = array.astype(np.uint8)
            return Image.fromarray(array).convert("RGB")
        if isinstance(value, (str, os.PathLike)):
            value = os.fspath(value)
            if str(value).lower().startswith(("http://", "https://")):
                with urllib.request.urlopen(value, timeout=30) as response:
                    return Image.open(io.BytesIO(response.read())).convert("RGB")
            return Image.open(value).convert("RGB")
        raise TypeError(f"Unsupported Hugging Face image value type: {type(value)!r}")

    def get_image(self, index):
        """Return the paired image for a sample as an RGB PIL image."""
        if self.image_column is None:
            self.image_column = self._resolve_image_column(self.dataset, required=True)
        sample = self.dataset[index]
        errors = []
        for column_name in [self.image_column] + [
            name for name in self._image_related_columns(self.dataset)
            if name != self.image_column
        ]:
            if column_name not in sample:
                continue
            try:
                return self._to_pil_image(sample[column_name])
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"{column_name}: {exc}")
        raise RuntimeError(
            f"Could not decode a paired image for Hugging Face sample {index}. "
            "Expected a PIL image, encoded bytes, image path or URL. "
            f"Tried columns: {errors}"
        )

    def get_subject(self, index):
        sample = self.dataset[index]
        return str(sample.get("subject", "unknown"))

    def get_label_name(self, index):
        sample = self.dataset[index]
        return self._sample_class_name(sample)

    def _time_first_shape(self, shape):
        if shape[0] < shape[1]:
            return shape[1], shape[0]
        return shape[0], shape[1]

    def _resample_sample(self, sample):
        if not self.adaptive_seq_len or sample.shape[0] == self.target_seq_len:
            return sample

        sample_tensor = torch.from_numpy(sample.T).unsqueeze(0)
        sample_tensor = F.interpolate(
            sample_tensor,
            size=self.target_seq_len,
            mode="linear",
            align_corners=False,
        )
        return sample_tensor.squeeze(0).transpose(0, 1).contiguous().numpy()

    def _normalize_sample(self, sample):
        mean = sample.mean(axis=0, keepdims=True)
        std = sample.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (sample - mean) / std

    def _eeg_to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _resolve_dataset_metadata(self):
        cache_key = (
            self.dataset_id,
            self.split_name,
            tuple(self.class_names),
            self.target_seq_len if self.adaptive_seq_len else None,
        )
        cached = self._metadata_cache.get(cache_key)
        if cached is not None:
            return cached

        max_seq_len = None
        feature_dim = None

        for idx in range(len(self.dataset)):
            sample = self._eeg_to_numpy(self.dataset[idx]["conditioning_image"])
            if sample.ndim != 2:
                raise ValueError(
                    f"Expected 2D EEG array from Hugging Face sample #{idx}, "
                    f"got shape {sample.shape}"
                )

            seq_len, feat_dim = self._time_first_shape(sample.shape)
            if feature_dim is None:
                feature_dim = feat_dim
            elif feature_dim != feat_dim:
                raise ValueError(
                    f"Inconsistent feature dimension in {self.dataset_id}: "
                    f"expected {feature_dim}, got {feat_dim} for sample #{idx}"
                )

            if max_seq_len is None or seq_len > max_seq_len:
                max_seq_len = seq_len

        if max_seq_len is None or feature_dim is None:
            raise RuntimeError(
                f"Failed to infer metadata from Hugging Face dataset {self.dataset_id}"
            )

        if self.adaptive_seq_len:
            metadata = (self.target_seq_len, feature_dim)
        else:
            metadata = (max_seq_len, feature_dim)
        self._metadata_cache[cache_key] = metadata
        return metadata

    def __getitem__(self, index):
        sample = self.dataset[index]
        eeg = self._eeg_to_numpy(sample["conditioning_image"]).astype(
            np.float32, copy=False
        )
        if eeg.ndim != 2:
            raise ValueError(
                f"Expected 2D EEG array, got shape {eeg.shape} from "
                f"Hugging Face dataset sample #{index}"
            )
        if eeg.shape[0] < eeg.shape[1]:
            eeg = eeg.T

        eeg = self._resample_sample(eeg)
        if self.normalize:
            eeg = self._normalize_sample(eeg)
        label = np.asarray(self._label_index_from_sample(sample), dtype=np.int64)
        return torch.from_numpy(eeg), torch.from_numpy(label)

    def __len__(self):
        return len(self.dataset)


class EEGImageNetLoader(Dataset):
    _metadata_cache = {}

    def __init__(self, args, root_path, flag=None):
        self.args = args
        self.root_path = Path(root_path)
        self.data_root = self._resolve_data_root(self.root_path)
        self.flag = flag
        self.normalize = getattr(args, "eeg_normalize", True)
        self.max_classes = getattr(args, "eeg_num_classes", 0)
        self.target_seq_len = getattr(args, "requested_seq_len", getattr(args, "seq_len", None))
        self.adaptive_seq_len = (
            getattr(args, "eeg_adaptive_seq_len", False)
            and self.target_seq_len is not None
            and self.target_seq_len > 0
        )

        self.class_names = self._discover_class_names()
        self.label_to_index = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }
        self.num_class = len(self.class_names)

        all_subjects = self._discover_subjects()
        self.train_ids, self.val_ids, self.test_ids = self._split_subjects(
            all_subjects, seed=getattr(args, "seed", 42)
        )

        if flag == "TRAIN":
            active_subjects = set(self.train_ids)
        elif flag == "VAL":
            active_subjects = set(self.val_ids)
        elif flag == "TEST":
            active_subjects = set(self.test_ids)
        else:
            active_subjects = set(all_subjects)

        self.samples = self._build_samples(active_subjects)
        if not self.samples:
            raise RuntimeError(
                f"No EEG-ImageNet samples found for split {flag!r} under {self.data_root}"
            )

        self.max_seq_len, self.feature_dim = self._resolve_dataset_metadata()
        self.y = np.asarray([label_idx for _, label_idx, _ in self.samples], dtype=np.int64)

    def _resolve_data_root(self, root_path):
        if root_path.is_dir() and any(root_path.glob("*/*/*.npy")):
            return root_path

        extracted_root = root_path / "extracted_npy"
        if extracted_root.is_dir() and any(extracted_root.glob("*/*/*.npy")):
            return extracted_root

        raise FileNotFoundError(
            "Could not find extracted EEG-ImageNet NPY files under "
            f"{root_path} or {extracted_root}"
        )

    def _discover_class_names(self):
        class_dirs = [path for path in self.data_root.iterdir() if path.is_dir()]
        class_names = natsorted([path.name for path in class_dirs])
        if not class_names:
            raise RuntimeError(f"No class folders found under {self.data_root}")
        if self.max_classes:
            if self.max_classes < 0:
                raise ValueError(
                    f"eeg_num_classes must be >= 0, got {self.max_classes}"
                )
            if self.max_classes > len(class_names):
                raise ValueError(
                    f"Requested {self.max_classes} classes, but only "
                    f"{len(class_names)} classes exist under {self.data_root}"
                )
            class_names = class_names[: self.max_classes]
        return class_names

    def _discover_subjects(self):
        subjects = set()
        for class_name in self.class_names:
            class_dir = self.data_root / class_name
            for subject_dir in class_dir.iterdir():
                if subject_dir.is_dir():
                    subjects.add(subject_dir.name)

        subject_ids = natsorted(subjects)
        if len(subject_ids) < 3:
            raise RuntimeError(
                f"Expected at least 3 subjects for train/val/test splitting, got {len(subject_ids)}"
            )
        return subject_ids

    def _split_subjects(self, subject_ids, seed=42):
        rng = random.Random(seed)
        shuffled_subjects = list(subject_ids)
        rng.shuffle(shuffled_subjects)

        total = len(shuffled_subjects)
        train_end = max(1, int(total * 0.6))
        val_end = max(train_end + 1, int(total * 0.8))
        val_end = min(val_end, total - 1)

        train_ids = shuffled_subjects[:train_end]
        val_ids = shuffled_subjects[train_end:val_end]
        test_ids = shuffled_subjects[val_end:]

        if not val_ids:
            val_ids = test_ids[:1]
            test_ids = test_ids[1:]
        if not test_ids:
            test_ids = val_ids[-1:]
            val_ids = val_ids[:-1]

        return train_ids, val_ids, test_ids

    def _build_samples(self, active_subjects):
        samples = []
        for class_name in self.class_names:
            class_dir = self.data_root / class_name
            label_idx = self.label_to_index[class_name]
            for subject_dir in natsorted(
                [path for path in class_dir.iterdir() if path.is_dir()],
                key=lambda path: path.name,
            ):
                if subject_dir.name not in active_subjects:
                    continue
                for sample_path in natsorted(subject_dir.glob("*.npy")):
                    samples.append((sample_path, label_idx, subject_dir.name))
        return samples

    def _resolve_dataset_metadata(self):
        cache_key = (str(self.data_root.resolve()), tuple(self.class_names))
        cached = self._metadata_cache.get(cache_key)
        if cached is not None:
            max_seq_len, feature_dim = cached
            if self.adaptive_seq_len:
                return self.target_seq_len, feature_dim
            return cached

        max_seq_len = None
        feature_dim = None

        for class_name in self.class_names:
            class_dir = self.data_root / class_name
            for subject_dir in [path for path in class_dir.iterdir() if path.is_dir()]:
                for sample_path in subject_dir.glob("*.npy"):
                    sample = np.load(sample_path, mmap_mode="r")
                    if sample.ndim != 2:
                        raise ValueError(
                            f"Expected 2D EEG array, got shape {sample.shape} from {sample_path}"
                        )

                    seq_len, feat_dim = self._time_first_shape(sample.shape)
                    if feature_dim is None:
                        feature_dim = feat_dim
                    elif feature_dim != feat_dim:
                        raise ValueError(
                            f"Inconsistent feature dimension: expected {feature_dim}, "
                            f"got {feat_dim} from {sample_path}"
                        )

                    if max_seq_len is None or seq_len > max_seq_len:
                        max_seq_len = seq_len

        if max_seq_len is None or feature_dim is None:
            raise RuntimeError(f"Failed to infer dataset metadata under {self.data_root}")

        metadata = (max_seq_len, feature_dim)
        self._metadata_cache[cache_key] = metadata
        if self.adaptive_seq_len:
            return self.target_seq_len, feature_dim
        return metadata

    def _time_first_shape(self, shape):
        if shape[0] < shape[1]:
            return shape[1], shape[0]
        return shape[0], shape[1]

    def _resample_sample(self, sample):
        if not self.adaptive_seq_len or sample.shape[0] == self.target_seq_len:
            return sample

        sample_tensor = torch.from_numpy(sample.T).unsqueeze(0)
        sample_tensor = F.interpolate(
            sample_tensor,
            size=self.target_seq_len,
            mode="linear",
            align_corners=False,
        )
        return sample_tensor.squeeze(0).transpose(0, 1).contiguous().numpy()

    def _normalize_sample(self, sample):
        mean = sample.mean(axis=0, keepdims=True)
        std = sample.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (sample - mean) / std

    def __getitem__(self, index):
        sample_path, label_idx, _ = self.samples[index]
        eeg = np.load(sample_path, allow_pickle=False).astype(np.float32, copy=False)
        if eeg.ndim != 2:
            raise ValueError(f"Expected 2D EEG array, got shape {eeg.shape} from {sample_path}")
        if eeg.shape[0] < eeg.shape[1]:
            eeg = eeg.T

        eeg = self._resample_sample(eeg)
        if self.normalize:
            eeg = self._normalize_sample(eeg)
        label = np.asarray(label_idx, dtype=np.int64)
        return torch.from_numpy(eeg), torch.from_numpy(label)

    def __len__(self):
        return len(self.samples)

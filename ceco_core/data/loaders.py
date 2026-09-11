import logging
import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch.utils.data import DataLoader

from ceco_core.data.preprocessor import Preprocessor
from ceco_core.utils.random_state import get_random_state


class HDFSSegLoader(object):
    _CONFIG_PATH = 'configs/training/hdfs.yaml'

    def __init__(self, ensemble_param, data_path, win_size, step, data_seq_len, mode='train'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        preprocessor = Preprocessor(length=data_seq_len, timeout=float('inf'))

        path_train = os.path.join(data_path, 'hdfs_train.txt')
        path_test_normal = os.path.join(data_path, 'hdfs_test_normal.txt')
        path_test_abnormal = os.path.join(data_path, 'hdfs_test_abnormal.txt')

        X_train, _, _, _ = preprocessor.text(path_train, verbose=True)
        X_test, _, _, _ = preprocessor.text(path_test_normal, verbose=True)
        X_test_anomaly, _, _, _ = preprocessor.text(path_test_abnormal, verbose=True)

        data = X_train.numpy()
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        if self.mode == 'train':
            random_number = get_random_state(
                self._CONFIG_PATH,
                ensemble_param[0], ensemble_param[1], ensemble_param[2], ensemble_param[3],
            )
            data, _ = resample(data, data, replace=True, n_samples=len(data), random_state=random_number)

        test_normal = X_test.numpy()
        test_abnormal = X_test_anomaly.numpy()
        test_data = np.concatenate((test_normal, test_abnormal), axis=0)

        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test

        test_normal_labels = np.full(len(test_normal), 0, dtype=int)
        test_abnormal_labels = np.full(len(test_abnormal), 1, dtype=int)
        self.test_labels = np.concatenate((test_normal_labels, test_abnormal_labels), axis=None)

        logging.info(f"test data shape: {self.test.shape}")
        logging.info(f"train data shape: {self.train.shape}")
        logging.info(f"test_labels shape: {self.test_labels.shape}")

    def __len__(self):
        if self.mode == 'train':
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == 'train':
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            return np.float32(self.test[start:start + self.win_size]), np.float32(self.test_labels[start:start + self.win_size])


class BGLSegLoader(object):
    _CONFIG_PATH = 'configs/training/bgl.yaml'

    def __init__(self, ensemble_param, data_path, win_size, step, data_seq_len, mode='train'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        preprocessor = Preprocessor(length=data_seq_len, timeout=float('inf'))

        path_train = os.path.join(data_path, 'bgl_train.txt')
        path_test_normal = os.path.join(data_path, 'bgl_test_normal.txt')
        path_test_abnormal = os.path.join(data_path, 'bgl_test_abnormal.txt')

        X_train, _, _, _ = preprocessor.text(path_train, verbose=True)
        X_test, _, _, _ = preprocessor.text(path_test_normal, verbose=True)
        X_test_anomaly, _, _, _ = preprocessor.text(path_test_abnormal, verbose=True)

        data = X_train.numpy()
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        if self.mode == 'train':
            random_number = get_random_state(
                self._CONFIG_PATH,
                ensemble_param[0], ensemble_param[1], ensemble_param[2], ensemble_param[3],
            )
            data, _ = resample(data, data, replace=True, n_samples=len(data), random_state=random_number)

        test_normal = X_test.numpy()
        test_abnormal = X_test_anomaly.numpy()
        test_data = np.concatenate((test_normal, test_abnormal), axis=0)

        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test

        test_normal_labels = np.full(len(test_normal), 0, dtype=int)
        test_abnormal_labels = np.full(len(test_abnormal), 1, dtype=int)
        self.test_labels = np.concatenate((test_normal_labels, test_abnormal_labels), axis=None)

        logging.info(f"test data shape: {self.test.shape}")
        logging.info(f"train data shape: {self.train.shape}")
        logging.info(f"test_labels shape: {self.test_labels.shape}")

    def __len__(self):
        if self.mode == 'train':
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == 'train':
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            return np.float32(self.test[start:start + self.win_size]), np.float32(self.test_labels[start:start + self.win_size])


class OpenStackSegLoader(object):
    _CONFIG_PATH = 'configs/training/os.yaml'

    def __init__(self, ensemble_param, data_path, win_size, step, data_seq_len, mode='train'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        preprocessor = Preprocessor(length=data_seq_len, timeout=float('inf'))

        path_train = os.path.join(data_path, 'train.txt')
        path_test_normal = os.path.join(data_path, 'test_normal.txt')
        path_test_abnormal = os.path.join(data_path, 'test_abnormal.txt')

        X_train, _, _, _ = preprocessor.text(path_train, verbose=True)
        X_test, _, _, _ = preprocessor.text(path_test_normal, verbose=True)
        X_test_anomaly, _, _, _ = preprocessor.text(path_test_abnormal, verbose=True)

        data = X_train.numpy()
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        if self.mode == 'train':
            random_number = get_random_state(
                self._CONFIG_PATH,
                ensemble_param[0], ensemble_param[1], ensemble_param[2], ensemble_param[3],
            )
            data, _ = resample(data, data, replace=True, n_samples=len(data), random_state=random_number)

        test_normal = X_test.numpy()
        test_abnormal = X_test_anomaly.numpy()
        test_data = np.concatenate((test_normal, test_abnormal), axis=0)

        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test

        test_normal_labels = np.full(len(test_normal), 0, dtype=int)
        test_abnormal_labels = np.full(len(test_abnormal), 1, dtype=int)
        self.test_labels = np.concatenate((test_normal_labels, test_abnormal_labels), axis=None)

        logging.info(f"test data shape: {self.test.shape}")
        logging.info(f"train data shape: {self.train.shape}")
        logging.info(f"test_labels shape: {self.test_labels.shape}")

    def __len__(self):
        if self.mode == 'train':
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == 'train':
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            return np.float32(self.test[start:start + self.win_size]), np.float32(self.test_labels[start:start + self.win_size])


_DATASET_MAP = {
    'Openstack': OpenStackSegLoader,
    'HDFS':      HDFSSegLoader,
    'BGL':       BGLSegLoader,
}


def get_loader_segment(
    ensemble_param,
    data_path: str,
    batch_size: int,
    win_size: int = 100,
    step: int = 100,
    data_seq_len: int = 10,
    mode: str = 'train',
    dataset: str = 'BGL',
) -> DataLoader:
    if dataset not in _DATASET_MAP:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {list(_DATASET_MAP)}.")
    ds = _DATASET_MAP[dataset](ensemble_param, data_path, win_size, step, data_seq_len, mode)
    shuffle = mode == 'train'
    return DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

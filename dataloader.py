import os
import pickle
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch


class FederatedDataset(Dataset):
    """Federated dataset loader for CICIDS2017.

    Expects pre-processed per-device pickle files at:
        data/<n_devices>_nodes/device_<id>/dataset_<mode>.txt

    Each pickle contains: {'dataset': DataFrame, 'labels': np.array}

    To generate these files, run `python dataloader.py` or provide your own
    following the same format.
    """

    def __init__(self, device_id, n_devices=11, data_path='data/',
                 mode='train', two_class=True):
        self.device_id = device_id
        self.mode = mode
        self.two_class = two_class

        file_path = os.path.join(
            data_path, f'{n_devices}_nodes',
            f'device_{device_id}', f'dataset_{mode}.txt'
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Dataset not found at {file_path}. "
                f"Run `python dataloader.py` first or place data files manually. "
                f"See README for expected directory structure."
            )

        with open(file_path, 'rb') as f:
            a = pickle.load(f)
        self.data = a['dataset']
        self.labels = a['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = np.array(self.data.iloc[idx])
        label = self.labels[idx]
        if self.two_class:
            label = 1 if label > 0 else 0
        data = data.reshape(1, -1)
        return torch.tensor(data, dtype=torch.float32), label


class CICIDSDatasetCreator:
    """Creates federated dataset splits from raw CICIDS2017 CSVs.

    Raw CSVs should be placed in raw_data_path (default: 'raw_data/').
    """

    def __init__(self, raw_data_path='raw_data/', output_path='data/',
                 n_devices=11, frac_train=0.8, bootstrap=False):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.n_devices = n_devices
        self.frac_train = frac_train
        self.bootstrap = bootstrap

    def create(self):
        dataset = self._read_csvs()
        x_dataset, y = self._clean_dataset(dataset)
        y_encoded = self._label_encode(y)
        x_scaled = self._scale_data(x_dataset)
        y_attacks = self._get_eligible_attacks(y_encoded)

        y_encoded = np.array(y_encoded)
        frac = 1 / self.n_devices
        benign_indices = [[] for _ in range(self.n_devices)]

        if self.bootstrap:
            ben_len = []

        for attack in y_attacks:
            indices = np.where(y_encoded == attack)[0]
            if attack == 0:
                for dev in range(self.n_devices):
                    start = int(dev * frac * len(indices))
                    end = int((dev + 1) * frac * len(indices))
                    benign_indices[dev].extend(indices[start:end])
                    if self.bootstrap:
                        ben_len.append(len(benign_indices[dev]))
            if attack > 0:
                target_dev = np.where(y_attacks == attack)[0][0] - 1
                benign_indices[target_dev].extend(indices)
                if self.bootstrap:
                    n = math.ceil(ben_len[target_dev] / len(indices))
                    for _ in range(n - 1):
                        benign_indices[target_dev].extend(indices)

        base_path = os.path.join(self.output_path, f'{self.n_devices}_nodes')
        for dev in range(self.n_devices):
            dev_data = x_scaled[np.array(benign_indices[dev])]
            dev_labels = y_encoded[np.array(benign_indices[dev])]
            self._create_splits(dev_data, dev_labels, dev, base_path)

        print(f"Datasets created at {base_path}/")

    def _create_splits(self, data, labels, device_id, base_path):
        labels = np.array(labels)
        modes = {'train': self.frac_train, 'val': 0.1, 'test': 0.1}
        indices_dict = {m: [] for m in modes}

        for attack in np.unique(labels):
            indices = np.where(labels == attack)[0]
            offset = 0
            for mode, frac in modes.items():
                end = int(offset + frac * len(indices))
                indices_dict[mode].extend(indices[offset:end])
                offset = end

        dev_path = os.path.join(base_path, f'device_{device_id}')
        os.makedirs(dev_path, exist_ok=True)

        for mode in modes:
            ind = np.array(indices_dict[mode])
            a = {
                'dataset': pd.DataFrame(data[ind]),
                'labels': labels[ind],
            }
            with open(os.path.join(dev_path, f'dataset_{mode}.txt'), 'wb') as f:
                pickle.dump(a, f)

    def _read_csvs(self):
        csv_files = [
            'Monday-WorkingHours.pcap_ISCX.csv',
            'Tuesday-WorkingHours.pcap_ISCX.csv',
            'Wednesday-workingHours.pcap_ISCX.csv',
            'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        ]
        frames = []
        for f in csv_files:
            path = os.path.join(self.raw_data_path, f)
            if os.path.exists(path):
                frames.append(pd.read_csv(path))
            else:
                print(f"Warning: {path} not found, skipping.")
        return pd.concat(frames, ignore_index=True)

    def _clean_dataset(self, dataset):
        dataset[dataset == np.inf] = np.nan
        dataset.fillna(0, inplace=True)
        x = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        return x, y

    def _label_encode(self, y):
        le = LabelEncoder()
        return le.fit_transform(y)

    def _scale_data(self, x):
        scaler = MinMaxScaler()
        return scaler.fit_transform(x)

    def _get_eligible_attacks(self, y_encoded):
        y_val, y_count = np.unique(y_encoded, return_counts=True)
        return y_val[y_count > 100]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create federated datasets from CICIDS2017 CSVs')
    parser.add_argument('--raw_data_path', type=str, default='raw_data/')
    parser.add_argument('--output_path', type=str, default='data/')
    parser.add_argument('--n_devices', type=int, default=11)
    parser.add_argument('--frac_train', type=float, default=0.8)
    parser.add_argument('--bootstrap', action='store_true')
    args = parser.parse_args()

    creator = CICIDSDatasetCreator(
        raw_data_path=args.raw_data_path,
        output_path=args.output_path,
        n_devices=args.n_devices,
        frac_train=args.frac_train,
        bootstrap=args.bootstrap,
    )
    creator.create()

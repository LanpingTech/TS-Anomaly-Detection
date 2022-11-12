import os
import numpy as np
import pandas as pd
from scipy.io import arff
from torch.utils.data import Dataset, Subset


class ECG5000(Dataset):
    def __init__(self, dataset):
        super(ECG5000, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return self.dataset[index, :-1], self.dataset[index, -1]


def data_preprocessing(data_root, train_val_split=0.8):
    with open(os.path.join(data_root, 'ECG5000_TRAIN.arff')) as f:
        dataset1, meta1 = arff.loadarff(f)
    with open(os.path.join(data_root, 'ECG5000_TEST.arff')) as f:
        dataset2, meta2 = arff.loadarff(f)
    dataset = pd.concat([pd.DataFrame(dataset1), pd.DataFrame(dataset2)])
    dataset["target"] = pd.to_numeric(dataset["target"])
    dataset["target"] = dataset["target"].apply(lambda x: 0 if x == 1 else 1)

    idx_list = np.arange(dataset.shape[0])
    np.random.shuffle(idx_list)

    total_num = dataset.shape[0]
    train_val_num = int(total_num * train_val_split)
    train_num = int(train_val_num * train_val_split)

    train_dataset = dataset.iloc[idx_list[:train_num]].values
    mean = np.mean(train_dataset[:, :-1])
    std = np.std(train_dataset[:, :-1])
    train_dataset[:, :-1] = (train_dataset[:, :-1] - mean) / std
    val_dataset = dataset.iloc[idx_list[train_num:train_val_num]].values
    val_dataset[:, :-1] = (val_dataset[:, :-1] - mean) / std
    test_dataset = dataset.iloc[idx_list[train_val_num:]].values
    test_dataset[:, :-1] = (test_dataset[:, :-1] - mean) / std
    np.save(os.path.join(data_root, 'test.npy'), test_dataset)

    train_dataset = ECG5000(train_dataset)
    val_dataset = ECG5000(val_dataset)
    test_dataset = ECG5000(test_dataset)


    return train_dataset, val_dataset, test_dataset

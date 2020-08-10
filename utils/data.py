#!/usr/bin/python3
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import time

class CollisionDataset(Dataset):
    """
    Abstract class for the collion detection

    Args
        path: (string) path to the dataset
    """
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        self._data = data.values
        # self._data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        input_num = self._data.shape[1]-1
        # inputs = torch.FloatTensor(self._data.iloc[idx,0:input_num])
        # labels = torch.IntTensor([self._data.iat[idx,input_num]])
        inputs = torch.from_numpy(self._data[idx,0:input_num]).float()
        labels = torch.from_numpy(np.asarray(self._data[idx,input_num],dtype=int))

        return inputs, labels

    @property
    def input_dim_(self):
        return len(self[0][0])


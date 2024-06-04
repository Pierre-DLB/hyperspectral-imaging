import scipy.io as sio
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from typing import Dict, List
import torchvision.transforms as T

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from torchinfo import summary

from sklearn.metrics import accuracy_score
import seaborn as sns
from tqdm.notebook import tqdm

import utils as u
import RPNetRFextractor as rp

from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
import pickle


from sklearn.model_selection import train_test_split

configuration = yaml.safe_load(open("config.yaml"))
data_path = configuration["data"]
parameters = configuration["parameters"]


# create data tools : data loader and data set


class HSI_Dataset(Dataset):

    def __init__(self, X, y, transform=None):
        """X is a FLATTENED HSI volume dataset, y is a 1D FLATTENED tensor with the ground truth"""
        super().__init__()
        if transform is None:
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            self.X = transform(X)  # should be transformed to tensors + normalized
        self.y = torch.tensor(
            y, dtype=torch.float32
        )  # should be transformed to tensor + normalized
        self.transform = transform

        assert X.shape[0] == len(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class HSI_AEDataset(Dataset):
    """Dataset class returning the X values and noise(X)"""

    def __init__(self, X, noise, transform=torch.tensor):
        """X is a FLATTENED HSI volume dataset, y is a 1D FLATTENED tensor with the ground truth"""
        super().__init__()
        self.X = transform(X)  # should be transformed to tensors + normalized
        self.transform = transform
        self.noise = noise

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.noise is None:
            return self.X[i], self.X[i]
        else:
            return self.noise(self.X[i]), self.X[i]


# train_ds = HSI_Dataset(X_train, y_train, transform=in_transform)
# test_ds = HSI_Dataset(X_test, y_test, transform=in_transform)


def scaler(X, vals=None):
    x = torch.tensor(X, dtype=torch.float32)
    m, std = torch.std_mean(x, axis=0)
    if vals is None:
        return (x - m) / std
    else:
        return (x - m) / std, (m, std)


def min_max_scaler(X, vals=None):
    x = torch.tensor(X, dtype=torch.float32)
    min, max = torch.min(x, axis=0), torch.max(x, axis=0)
    if vals is None:
        return (x - min) / (max - min)
    else:
        return (x - min) / (max - min), (max, min)


def gaussian_noise(x, m=0, s=0.1):
    x = x + torch.normal(m * torch.zeros_like(x), std=s * torch.ones_like(x))
    return x


def zero_noise(x, p=10):
    """randomly sets p coordinates to 0"""
    size = x.shape[0]
    idx = np.random.choice(range(size), size=p, replace=False)
    x[idx] = 0.0
    return x


train_ds = HSI_Dataset(X_train, y_train)
test_ds = HSI_Dataset(X_test, y_test)


train_dl = DataLoader(train_ds, batch_size=1000, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=1000, shuffle=False)

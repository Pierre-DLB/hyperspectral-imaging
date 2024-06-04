import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import ae_architecture as ae
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import utils as u
import pickle


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


def test_different_sae(ratios, lr=0.001, n_in=100, bottleneck=14):

    results = {r: [] for r in ratios}

    for r in tqdm(ratios, desc="Processing Ratios"):
        sae = ae.ShallowAE(n_in, int(n_in * r), bottleneck)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(params=sae.parameters(), lr=lr)
        loss_train, loss_test = ae.train_ae(
            sae, optimizer, loss_fn, train_dl, test_dl, epochs=30, verbose=False
        )
        results[r] = [loss_train, loss_test]
        torch.save(sae.state_dict(), f"models/sae_{r}.pt")
    epoch_results = {
        r: (
            [results[r][0][epoch].mean() for epoch in range(30)],
            [results[r][1][epoch].mean() for epoch in range(30)],
        )
        for r in ratios
    }

    with open(f"models/resultats_lr{lr}.pkl", "wb") as file:
        pickle.dump(epoch_results, file)
    print(f"Dictionary saved to models/results_lr{lr}.pkl")

    return epoch_results


def test_shallow_ae(
    ratio: float, lr: float, n_in: int = 100, bottleneck: int = 14, epochs: int = 30
):
    sae = ae.ShallowAE(n_in, int(n_in * ratio), bottleneck)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=sae.parameters(), lr=lr)
    loss_train, loss_test = ae.train_ae(
        sae, optimizer, loss_fn, train_dl, test_dl, epochs=epochs, verbose=False
    )
    results = np.array([lt.mean() for lt in loss_train]), np.array(
        [lt.mean() for lt in loss_test]
    )
    # torch.save(sae.state_dict(), f"models/sae_{r}.pt")
    return results


def plot_ae(
    ratio: float, lr: float, n_in: int = 100, bottleneck: int = 14, epochs: int = 30
):
    results = test_shallow_ae(ratio, lr, n_in, bottleneck, epochs)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.plot(range(epochs), results[0], c="orange", label="Train")
    plt.plot(range(epochs), results[1], c="b", label="Test")
    plt.legend()
    plt.show()
    plt.savefig(f"figures/ae{ratio}_{lr}_{n_in}_{bottleneck}_{epochs}.png")
    return results


if __name__ == "__main__":

    data = u.load_data("KSC")
    X, y = data
    train_data, test_data, unsup_data = u.data_split(data[1])
    X_train, y_train = X[train_data + unsup_data], y[train_data + unsup_data]
    # X_train, y_train = X[train_data], y[train_data]
    X_test, y_test = X[test_data], y[test_data]

    train_ds = HSI_AEDataset(X_train, noise=None, transform=scaler)
    test_ds = HSI_AEDataset(X_test, noise=None, transform=scaler)

    train_dl = DataLoader(train_ds, batch_size=400, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=400, shuffle=False)

    # CONSTANTS

    n_in = X_train.shape[0]  # next(iter(train_dl))[0].shape[1]
    bottleneck = 14  # because 14 classes

    ratio = 7.0
    lr = 0.001
    epochs = 50

    plot_ae(ratio, lr, n_in, bottleneck, epochs)

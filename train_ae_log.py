import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from torch import nn

import torch
from torch.utils.data import Dataset, DataLoader
import ae_architecture as ae
from sklearn.metrics import accuracy_score

from tqdm.notebook import tqdm
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import utils as u

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
    x = torch.tensor(X, dtype=torch.float32).clone().detach()
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


def encode_x(x, autoencoder, preprocessing=None, device="cpu"):
    """Encodes the data using the autoencoder"""
    autoencoder.to(device).eval()
    x = torch.tensor(x, device=device, dtype=torch.float32)
    if preprocessing is not None:
        x = preprocessing(x)
    x_encoded = autoencoder.encode(x)
    return x_encoded.to("cpu").detach().numpy()


# auto encoder build
def draft_plot_ae(EPOCHS, r, lr, verbose=False, sae=None):
    if sae is None:
        sae_log = ae.ShallowAE(n_in, int(r * n_in), bottleneck)
    else:
        sae_log = sae
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=sae_log.parameters(), lr=lr)

    # Auto encoder training
    loss_train, loss_test, loss_unsup = ae.train_ae(
        sae_log,
        optimizer,
        loss_fn,
        train_dl2,
        test_dl2,
        EPOCHS,
        unsup_dl=unsup_dl2,
        device="mps",
        verbose=verbose,
    )
    results = np.array(loss_train), np.array(loss_test), np.array(loss_unsup)

    # figure plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(range(EPOCHS), results[0], c="orange", label="Train", linestyle="--")
    ax[0].plot(range(EPOCHS), results[1], c="b", label="Test", linestyle="-.")
    ax[0].plot(range(EPOCHS), results[2], c="r", label="Unsup")

    ax[1].plot(
        range(EPOCHS), np.log(results[0]), c="orange", label="logTrain", linestyle="--"
    )
    ax[1].plot(
        range(EPOCHS), np.log(results[1]), c="b", label="logTest", linestyle="-."
    )
    ax[1].plot(range(EPOCHS), np.log(results[2]), c="r", label="logUnsup")
    plt.legend()

    # save chart
    plt.savefig(f"figures/ae_log_r{r}_lr{lr}.png")
    # plt.show()

    return sae_log


if __name__ == "__main__":
    params = {
        "r": 2.0,
        "epochs": 30,
        "lr": 0.002,
        "seed": 7,
        "dataset": "PU",
        "version": "v2",
    }

    data = u.load_data(params["dataset"])
    X, y = data
    train_data, test_data, unsup_data = u.data_split(data[1], seed=params["seed"])
    X_log = np.log(X / 2 + 1)

    X_train_full, y_train_full = (
        X_log[train_data + unsup_data],
        y[train_data + unsup_data],
    )
    X_train2, y_train = X_log[train_data], y[train_data]
    X_unsup2, y_unsup = X_log[unsup_data], y[unsup_data]
    X_test2, y_test = X_log[test_data], y[test_data]

    # print("Full : ", X_train_full.shape, y_train_full.shape)
    # print("Not Full : ", X_train2.shape, y_train.shape)

    # train_ds = HSI_AEDataset(X_train_full, noise=None, transform=scaler)
    train_ds2 = HSI_AEDataset(X_train2, noise=None, transform=scaler)
    test_ds2 = HSI_AEDataset(X_test2, noise=None, transform=scaler)
    unsup_ds2 = HSI_AEDataset(X_unsup2, noise=None, transform=scaler)

    # batch size à 522 pour avoir qqch qui divise la taille du dataset

    train_dl2 = DataLoader(train_ds2, batch_size=522, shuffle=True)
    unsup_dl2 = DataLoader(unsup_ds2, batch_size=2048, shuffle=True)
    test_dl2 = DataLoader(test_ds2, batch_size=1043, shuffle=False)
    # CONSTANTS

    n_in = X_train2.shape[1]  # next(iter(train_dl))[0].shape[1]
    bottleneck = 14  # because 14 classes

    # Model :
    sae = draft_plot_ae(
        EPOCHS=params["epochs"], r=params["r"], lr=params["lr"], verbose=True
    )
    torch.save(
        sae.state_dict(),
        f'models/sae_{params["r"]}_{params["lr"]}_{params["epochs"]}_log_{params["version"]}.pt',
    )

    # test on classif :

    latent_x_unsup = encode_x(X_unsup2, sae, preprocessing=scaler, device="mps")
    latent_x_train = encode_x(X_train2, sae, preprocessing=scaler, device="mps")
    latent_x_test = encode_x(X_test2, sae, preprocessing=scaler, device="mps")

    np.save("data_otp/latent_x_unsup_log.npy", latent_x_unsup)
    np.save("data_otp/latent_x_train_log.npy", latent_x_train)
    np.save("data_otp/latent_x_test_log.npy", latent_x_test)
    np.save("data_otp/y_train_log.npy", y_train)
    np.save("data_otp/y_test_log.npy", y_test)
    np.save("data_otp/y_unsup_log.npy", y_unsup)

    ## Assess on the classification

    # C, kernel, degree, gamma.

    rf = RandomForestClassifier(n_estimators=100, max_depth=13, random_state=42)
    svc = SVC(C=100, kernel="rbf", degree=3, gamma=2)
    xgb = XGBClassifier(max_depth=12)

    # Needed for XGBoost : needs labels starting at 0 and we removed the "unsup" pixels
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    svc.fit(latent_x_train, y_train)
    rf.fit(latent_x_train, y_train)
    print("train xgb")
    xgb.fit(latent_x_train, y_train)

    print(
        f"With seed {params['seed']} and SVC, accuracy is : {accuracy_score(y_test, svc.predict(latent_x_test))}"
    )
    print(
        f"With seed {params['seed']} and RF , accuracy is : {accuracy_score(y_test, rf.predict(latent_x_test))}"
    )
    print(
        f"With seed {params['seed']} and XGB, accuracy is : {accuracy_score(y_test, xgb.predict(latent_x_test))}"
    )


"""
KSC
Test 1:
    With seed 7 and SVC, accuracy is : 0.8609779482262704
    With seed 7 and RF , accuracy is : 0.8561840843720039
    With seed 7 and XGB, accuracy is : 0.850431447746884
Test 2: 
    With seed 7 and SVC, accuracy is : 0.8648130393096836
    With seed 7 and RF , accuracy is : 0.8312559923298178
    With seed 7 and XGB, accuracy is : 0.825503355704698
Test 3:
    With seed 7 and SVC, accuracy is : 0.8600191754554171
    With seed 7 and RF , accuracy is : 0.8427612655800575
    With seed 7 and XGB, accuracy is : 0.8283796740172579

IP
Test 1:
    With seed 7 and SVC, accuracy is : 0.5117073170731707
    With seed 7 and RF , accuracy is : 0.6351219512195122
    With seed 7 and XGB, accuracy is : 0.6351219512195122
Test 2: 
    With seed 7 and SVC, accuracy is : 0.38097560975609757
    With seed 7 and RF , accuracy is : 0.5975609756097561
    With seed 7 and XGB, accuracy is : 0.5765853658536585

PU:
Test 1:
    With seed 1 and SVC, accuracy is : 0.8062178588125292
    With seed 1 and RF , accuracy is : 0.823632538569425
    With seed 1 and XGB, accuracy is : 0.8104254324450678
Test 2:
    With seed 7 and SVC, accuracy is : 0.8020102851799906
    With seed 7 and RF , accuracy is : 0.8085553997194951
    With seed 7 and XGB, accuracy is : 0.7944132772323516
"""

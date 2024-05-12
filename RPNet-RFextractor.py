import utils as u
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from torch import nn
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

configuration = yaml.safe_load(open("config.yaml"))
data_path = configuration["data"]
parameters = configuration["parameters"]


def RPNet_extractor(X, y, params=None):
    """Feature extraction using RPNet method"""

    if params is None:
        params = parameters

    p = params["p"]  # number of PC in PCA
    L = params["L"]  # network depth
    k = params["k"]  # number of patches
    w = params["w"]  # size of patches
    long, larg, _ = X.shape

    feature_maps = []
    # L times feature extraction with convolutionn filters
    for i in range(L):
        # Scale and center the data, then apply pca to it
        scaler = StandardScaler()
        pca_white = PCA(n_components=p, whiten=True)
        A = X.reshape(long * larg, -1)
        A = scaler.fit_transform(A)
        feat_map = pca_white.fit_transform(A).reshape(long, larg, p)

        # extract patches:
        patches = u.extract_tiles(feat_map, k, w)
        patches_tensor = torch.from_numpy(patches).unsqueeze(1)
        res = nn.functional.conv3d(
            torch.from_numpy(feat_map).to(torch.float32).unsqueeze(0),
            weight=patches_tensor.to(torch.float32),
            bias=None,
            padding=(w // 2, w // 2, 0),
            dilation=1,
            groups=1,
        )

        extracted_map = res.squeeze(0).squeeze(-1).permute(1, 2, 0)

        feature_maps.append(extracted_map)

    feature_maps_tensor = torch.cat(feature_maps, dim=2)

    # final PCA to find the right number of PC ("Q" in the paper)

    pca = PCA()
    S = StandardScaler()
    feature_maps_tensor = S.fit_transform(feature_maps_tensor.reshape(-1, L * k))
    pca.fit(feature_maps_tensor)
    Q = pca.explained_variance_ratio_.cumsum().searchsorted(0.9995, side="right")

    print(f"We keep {Q+1} principal components to keep 99.95% of the variance.")
    pca = PCA(n_components=Q + 1)
    extracted_maps = pca.fit_transform(feature_maps_tensor)

    return torch.from_numpy(extracted_maps.reshape(long, larg, -1)).to(torch.float32)


def RF_horizontal(tensor, delta_s=parameters["delta_s"], delta_r=parameters["delta_r"]):
    """Recursive filtering in the horizontal direction
    Paper formulas :
        J[n] = (1-a**d)*I[n] + a**d*J[n-1]
        a = exp(-sqrt(2)/delta_s)

        d = Integral(x_(n-1), x_n, 1+ delta_s/delta_r * |I'[n]|)
            i.e. d = 1+ delta_s/delta_r* |I[n] - I[n-1]| (as x_n-1 and x_n are adjacent pixels)
    Return J, the filtered tensor
    """
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(2)
    _, larg, _ = tensor.shape

    filtered_tensor = np.zeros_like(tensor)

    # pb : recursive filtering is not
    for n in range(larg):
        if n == 0:
            filtered_tensor[:, n, :] = tensor[:, n, :]
        else:
            d = 1 + delta_s / delta_r * torch.abs(tensor[:, n, :] - tensor[:, n - 1, :])
            a_d = torch.exp(-torch.sqrt(torch.tensor(2)) * d / delta_s)
            filtered_tensor[:, n, :] = (1 - a_d) * tensor[
                :, n, :
            ] + a_d * filtered_tensor[:, n - 1, :]
    return torch.tensor(filtered_tensor)


def RF_vertical(tensor, delta_s=parameters["delta_s"], delta_r=parameters["delta_r"]):
    """Recursive filtering in the horizontal direction
    Paper formulas :
        J[n] = (1-a**d)*I[n] + a**d*J[n-1]
        a = exp(-sqrt(2)/delta_s)

        d = Integral(x_(n-1), x_n, 1+ delta_s/delta_r * |I'[n]|)
            i.e. d = 1+ delta_s/delta_r* |I[n] - I[n-1]| (as x_n-1 and x_n are adjacent pixels)
    Return J, the filtered tensor
    """
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(2)
    long, _, _ = tensor.shape

    filtered_tensor = np.zeros_like(tensor)

    # pb : recursive filtering is not
    for n in range(long):
        if n == 0:
            filtered_tensor[n, :, :] = tensor[n, :, :]
        else:
            d = 1 + delta_s / delta_r * torch.abs(tensor[n, :, :] - tensor[n - 1, :, :])
            a_d = torch.exp(-torch.sqrt(torch.tensor(2)) * d / delta_s)
            filtered_tensor[n, :, :] = (1 - a_d) * tensor[
                n, :, :
            ] + a_d * filtered_tensor[n - 1, :, :]
    return torch.tensor(filtered_tensor)


def RF(tensor, delta_s=parameters["delta_s"], delta_r=parameters["delta_r"]):
    """Applies 3 passes of recursive filtering, as mentionned in the paper : 3 vertical and 3 horizontal.
    Orignial paper on this recommends to modify the value of deltas between iterations,
    but replicated paper doesn't so not done here.

    """

    tensor1h = RF_horizontal(tensor, delta_s, delta_r)
    tensor1v = RF_vertical(tensor1h, delta_s, delta_r)
    tensor2h = RF_horizontal(tensor1v, delta_s, delta_r)
    tensor2v = RF_vertical(tensor2h, delta_s, delta_r)
    tensor3h = RF_horizontal(tensor2v, delta_s, delta_r)
    tensor3v = RF_vertical(tensor3h, delta_s, delta_r)
    return tensor3v


def RPNet_RF(X, y, params=parameters):
    """Full RPNet-RF method, from feature extraction to recursive filtering"""

    tensor = RPNet_extractor(X, y, params)
    filtered_tensor = RF(tensor, params["delta_s"], params["delta_r"])

    return torch.tensor(X).cat(filtered_tensor, dim=2)

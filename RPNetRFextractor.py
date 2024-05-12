import utils as u
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from torch import nn
import yaml

configuration = yaml.safe_load(open("config.yaml"))
data_path = configuration["data"]
parameters = configuration["parameters"]


def one_extraction(A, params=parameters):
    """One extraction of features in the RPNet method. A is in the shape (long, larg, prof)"""
    device = "cuda" if torch.cuda.is_available() else "mps"
    long, larg, _ = A.shape
    p = params["p"]  # number of PC in PCA
    k = params["k"]  # number of patches
    w = params["w"]  # size of patches

    scaler = StandardScaler()
    pca_white = PCA(n_components=p, whiten=True)

    A = A.reshape(long * larg, -1)
    A = scaler.fit_transform(A)
    feat_map = pca_white.fit_transform(A).reshape(long, larg, p)  # feature map: PCA(X)
    patches = u.extract_tiles(feat_map, k, w)

    tensor_patches = torch.from_numpy(patches).to(torch.float32).to(device).unsqueeze(1)
    tensor_map = torch.from_numpy(feat_map).to(torch.float32).to(device).unsqueeze(0)

    res = nn.functional.conv3d(
        tensor_map,
        weight=tensor_patches,
        bias=None,
        padding=(w // 2, w // 2, 0),
        dilation=1,
        groups=1,
    )
    # reshape the tensor to apply the next PCA
    extracted_map = res.squeeze(0).squeeze(-1).permute(1, 2, 0).to("cpu")

    return extracted_map


def RPNet_extractor(X, params=parameters):
    """Feature extraction using RPNet method"""

    L = params["L"]  # network depth
    k = params["k"]  # number of patches
    long, larg, _ = X.shape
    extracted_map = X.copy()
    feature_maps = []

    for i in range(L):
        # input : X, or last extracted map (long, larg, k), output : new extracted map (long, larg, k) also appended to feature maps
        extracted_map = one_extraction(extracted_map, params)
        print(f"Conv {i+1} done, shape : {extracted_map.shape}")
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


def RF_horizontal(tensor, params=parameters):
    """Recursive filtering in the horizontal direction
    Paper formulas :
        J[n] = (1-a**d)*I[n] + a**d*J[n-1]
        a = exp(-sqrt(2)/delta_s)

        d = Integral(x_(n-1), x_n, 1+ delta_s/delta_r * |I'[n]|)
            i.e. d = 1+ delta_s/delta_r* |I[n] - I[n-1]| (as x_n-1 and x_n are adjacent pixels)
    Return J, the filtered tensor
    """
    delta_s, delta_r = params["delta_s"], params["delta_r"]

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
    return torch.tensor(filtered_tensor).to(torch.float32)


def RF_vertical(tensor, params=parameters):
    """Recursive filtering in the horizontal direction
    Paper formulas :
        J[n] = (1-a**d)*I[n] + a**d*J[n-1]
        a = exp(-sqrt(2)/delta_s)

        d = Integral(x_(n-1), x_n, 1+ delta_s/delta_r * |I'[n]|)
            i.e. d = 1+ delta_s/delta_r* |I[n] - I[n-1]| (as x_n-1 and x_n are adjacent pixels)
    Return J, the filtered tensor
    """

    delta_s, delta_r = params["delta_s"], params["delta_r"]
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
    return torch.tensor(filtered_tensor).to(torch.float32)


def RF(tensor, params=parameters):
    """Applies 3 passes of recursive filtering, as mentionned in the paper : 3 vertical and 3 horizontal.
    Orignial paper on this recommends to modify the value of deltas between iterations,
    but replicated paper doesn't so not done here.

    """
    tensor1h = RF_horizontal(tensor, params)
    tensor1v = RF_vertical(tensor1h, params)
    tensor2h = RF_horizontal(tensor1v, params)
    tensor2v = RF_vertical(tensor2h, params)
    tensor3h = RF_horizontal(tensor2v, params)
    tensor3v = RF_vertical(tensor3h, params)
    return tensor3v


def RPNet_RF(X, params=parameters):
    """Full RPNet-RF method, from feature extraction to recursive filtering"""

    tensor = RPNet_extractor(X, params)
    filtered_tensor = RF(tensor, params)

    # return torch.tensor(X).to(torch.float32), filtered_tensor
    return torch.cat([torch.tensor(X).to(torch.float32), filtered_tensor], dim=2)

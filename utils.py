import scipy.io as sio
import yaml
import numpy as np


from sklearn.model_selection import train_test_split

configuration = yaml.safe_load(open("config.yaml"))
data_path = configuration["data"]


def load_data(ds: str):
    """Load data from the dataset specified by the user"""
    data = sio.loadmat(data_path[ds]["img"])
    labels = sio.loadmat(data_path[ds]["labels"])
    return data[list(data.keys())[-1]], labels[list(labels.keys())[-1]]


def extract_random_pixel(X, label):
    long, larg, depth = X.shape
    assert long == label.shape[0] and larg == label.shape[1]
    x = np.random.randint(long)
    y = np.random.randint(larg)
    return (X[x, y, :], label[x, y])


def balanced_split(y, test_size=0.2):
    """Returns the mask for a balanced split train-test
    y : labels
    test_size : proportion of the test set

    returns : train_mask, test_mask, unsup_mask

    """
    classes = np.unique(y)
    unsup_mask = y == 0  # some pixels are not that usefull : did not get any label

    train_mask = y == -1  # false everywhere, shaped as y
    test_mask = y == -1  # false everywhere, shaped as y

    for i in classes:
        if i > 0:
            col, rows = np.where(y == i)
            col_train, col_test, row_train, row_test = train_test_split(
                col, rows, test_size=test_size
            )
            train_mask[col_train, row_train] = True
            test_mask[col_test, row_test] = True
    return train_mask, test_mask, unsup_mask


def extract_tiles(X, number, w):

    patches_top_left_corners = [
        (np.random.randint(X.shape[0] - w), np.random.randint(X.shape[1] - w))
        for i in range(number)
    ]
    X_patches = []
    y_patches = []
    for a, b in patches_top_left_corners:
        X_patches.append(X[a : a + w, b : b + w, :])
    return np.array(X_patches)


def random_split(y, test_size=0.2):
    """Returns the mask for a balanced split train-test
    y : labels
    test_size : proportion of the test set

    returns : train_mask, test_mask, unsup_mask

    """
    unsup_mask = y == 0  # some pixels are not that usefull : did not get any label

    train_mask = y == -1  # false everywhere, shaped as y
    test_mask = y == -1  # false everywhere, shaped as y

    col, rows = np.where(y != 0)
    col_train, col_test, row_train, row_test = train_test_split(
        col, rows, test_size=test_size
    )
    train_mask[col_train, row_train] = True
    test_mask[col_test, row_test] = True
    return train_mask, test_mask, unsup_mask


def data_split(y, test_size=0.2, style="random"):
    if style == "balanced":
        return balanced_split(y, test_size)
    else:
        return random_split(y, test_size)

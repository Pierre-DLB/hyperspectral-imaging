from torch import nn
from typing import List
import numpy as np
import torch


# Auto-encoders
# https://www.jeremyjordan.me/autoencoders/


class Encoder(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hidden: List[int]):

        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden

        in_list = [n_in] + n_hidden
        out_list = n_hidden + [n_out]

        self.encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(x, y), nn.ReLU())
                for x, y in zip(in_list, out_list)
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hidden: List[int]):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden

        in_list = [n_in] + n_hidden
        out_list = n_hidden + [n_out]

        self.decoder = nn.Sequential(
            *[
                nn.Sequential(nn.ReLU(), nn.Linear(x, y))
                for x, y in zip(in_list, out_list)
            ],
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)


class AE(nn.Module):
    def __init__(
        self,
        n_in: int,
        bottleneck: int,
        n_hidden_encoder: List[int],
        n_hidden_decoder: List[int] = None,
    ):
        super().__init__()
        self.bottleneck = bottleneck
        self.n_hidden_encoder = n_hidden_encoder
        if n_hidden_decoder is None:
            # basecase : symetrical
            self.n_hidden_decoder = n_hidden_encoder[::-1]
        else:
            self.n_hidden_decoder = n_hidden_decoder

        self.encoder = Encoder(n_in, bottleneck, n_hidden_encoder)
        self.decoder = Decoder(bottleneck, n_in, self.n_hidden_decoder)

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, y: torch.Tensor):
        return self.decode(y)


class ShallowAE(AE):
    def __init__(self, n_in: int, n_hidden: int, bottleneck: int):
        super().__init__(n_in, bottleneck, [n_hidden])

    def forward(self, x: torch.Tensor):
        return super().forward(x)

    def encode(self, x: torch.Tensor):
        return super().encode(x)

    def decode(self, y: torch.Tensor):
        return super().decode(y)


# def train_ae_step(model, optimizer, loss_fn, train_dl, debug=False):

#     model.train()
#     running_loss_mean = []
#     running_loss_std = []
#     for batch, (noised_x, x) in enumerate(train_dl):
#         if debug:
#             print(f"train batch {batch+1}/{len(train_dl)}")
#         x2 = model(noised_x)
#         loss_intermediary = loss_fn(x2, x)

#         optimizer.zero_grad()
#         loss_intermediary.backward()
#         optimizer.step()
#         avg_loss = torch.mean(loss_intermediary)
#         std_loss = torch.std(loss_intermediary)

#         running_loss_mean.append(avg_loss.item())
#         running_loss_std.append(std_loss.item())
#     return running_loss_mean, running_loss_std


# def test_ae(model, loss_fn, test_dl, debug=False):

#     model.eval()
#     loss_batch_mean = []
#     loss_batch_std = []
#     for batch, (noised_x, x) in enumerate(test_dl):
#         if debug:
#             print(f"test batch {batch+1}/{len(test_dl)}")
#         x2 = model(noised_x)
#         loss = loss_fn(x2, x)
#         avg_loss = torch.mean(loss)
#         std_loss = torch.std(loss)
#         loss_batch_mean.append(avg_loss.item())
#         loss_batch_std.append(std_loss.item())
#     return loss_batch_mean, loss_batch_std


def test_ae(model, loss_fn, test_dl, device="cpu", debug=False):
    model.to(device)

    model.eval()
    loss_batch_mean = []

    for batch, (noised_x, x) in enumerate(test_dl):
        if debug:
            print(f"test batch {batch+1}/{len(test_dl)}")
        noised_x = noised_x.to(device)
        x = x.to(device)

        x2 = model(noised_x)
        loss = loss_fn(x2, x)
        avg_loss = torch.mean(loss)
        # std_loss = torch.std(loss)
        loss_batch_mean.append(avg_loss.item())
        # loss_batch_std.append(std_loss.item())
    return loss_batch_mean


def train_ae_step(model, optimizer, loss_fn, train_dl, device="cpu", debug=False):

    model.to(device)
    model.train()
    running_loss_mean = []
    # running_loss_std = []
    for batch, (noised_x, x) in enumerate(train_dl):
        if debug:
            print(f"train batch {batch+1}/{len(train_dl)}")

        noised_x = noised_x.to(device)
        x = x.to(device)

        x2 = model(noised_x)
        loss_intermediary = loss_fn(x2, x)

        optimizer.zero_grad()
        loss_intermediary.backward()
        optimizer.step()
        avg_loss = torch.mean(loss_intermediary)
        # std_loss = torch.std(loss_intermediary)

        running_loss_mean.append(avg_loss.item())
        # running_loss_std.append(std_loss.item())
    return running_loss_mean


def train_ae(
    model,
    optimizer,
    loss_fn,
    train_dl,
    test_dl,
    epochs,
    unsup_dl=None,
    device="cpu",
    debug=False,
    verbose=False,
):
    losses_train = []
    losses_test = []
    losses_unsup = []
    for i in range(epochs):
        if verbose:
            print(f"Epoch {i+1} :")

        m_train = train_ae_step(model, optimizer, loss_fn, train_dl, device, debug)
        m_test = test_ae(model, loss_fn, test_dl, device, debug)
        m_train, m_test = (
            np.array(m_train).mean(),
            np.array(m_test).mean(),
        )

        if verbose:
            print(f"\tLOSSES : {np.mean(m_train)} - {np.mean(m_test)}")
        losses_train.append(m_train)
        losses_test.append(m_test)

        if unsup_dl is not None:
            m_unsup = test_ae(model, loss_fn, unsup_dl, device, debug)
            m_unsup = np.array(m_unsup).mean()
            losses_unsup.append(m_unsup)

    if unsup_dl is not None:
        return losses_train, losses_test, losses_unsup
    else:
        return losses_train, losses_test

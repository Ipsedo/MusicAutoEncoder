import torch as th
import torch.nn as nn

import values


# Pourri
class ConstantUnpool1d(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size

        if stride != kernel_size:
            raise NotImplementedError("Different kernel size and stride is not implemented")

    def forward(self, x):
        return x.view(x.size(0), x.size(1), x.size(2), 1) \
            .repeat(1, 1, 1, self.kernel_size) \
            .view(x.size(0), x.size(1), x.size(2) * self.kernel_size)


# Pourri
class RandomZeroUnpool1d(ConstantUnpool1d):
    def __init__(self, kernel_size: int):
        super().__init__(kernel_size, kernel_size)

    def forward(self, x):
        res = super().forward(x).view(x.size(0), x.size(1), x.size(2), self.kernel_size)

        dev = th.device("cuda") if x.is_cuda else th.device("cpu")

        rand_v = th.rand(x.size(0), x.size(1), x.size(2), self.kernel_size, device=dev).to(th.float)
        _, mask_idx = th.max(rand_v, dim=-1)

        mask = th.zeros(*rand_v.size(), device=dev).to(th.float)
        mask = mask.scatter_(-1, mask_idx.view(x.size(0), x.size(1), x.size(2), 1), 1)

        return (res * mask).flatten(-2, -1)


class Encoder(nn.Module):
    def __init__(self, n_channel):
        super().__init__()

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(n_channel, n_channel + 32,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(n_channel + 32),
            nn.Conv1d(n_channel + 32, n_channel + 64,
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(n_channel + 64),
            nn.Conv1d(n_channel + 64, n_channel + 128,
                      kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(n_channel + 128)
        )

        self.n_channel = n_channel

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.n_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.n_channel}."
        return self.cnn_enc(x)


class Decoder(nn.Module):
    def __init__(self, n_channel):
        super().__init__()

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(n_channel, n_channel - 128,
                               kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(n_channel - 128),
            nn.ConvTranspose1d(n_channel - 128, n_channel - (128 + 64),
                               kernel_size=5, stride=2, output_padding=1, padding=2),
            nn.BatchNorm1d(n_channel - (128 + 64)),
            nn.ConvTranspose1d(n_channel - (128 + 64), values.N_FFT * 2,
                               kernel_size=3, padding=1)
        )

        self.n_channel = n_channel

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.n_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.n_channel}."
        return self.cnn_tr_dec(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channel_enc = values.N_FFT * 2
        self.in_channel_dec = values.N_FFT * 2 + 128

        self.enc = Encoder(self.in_channel_enc)
        self.dec = Decoder(self.in_channel_dec)

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.in_channel_enc, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.in_channel_enc}."

        out_enc = self.enc(x)
        out_dec = self.dec(out_enc)

        return out_enc, out_dec


class Discriminator(nn.Module):
    def __init__(self, n_channel):
        super().__init__()

        self.n_channel = n_channel

        self.lin_dicr = nn.Sequential(
            nn.Linear(self.n_channel, self.n_channel * 2),
            nn.BatchNorm1d(self.n_channel * 2),
            nn.Linear(self.n_channel * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        assert len(x.size()) == 2, \
            f"Wrong size len, actual : {len(x.size())}, needed : 2."
        assert x.size(1) == self.n_channel, \
            f"Wrong size, actual : {x.size()}, needed : (N, C)."

        return self.lin_dicr(x).view(-1)


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d_z_prime, d_z):
        assert len(d_z_prime.size()) == 1, \
            f"Wrong z_prime size, actual : {d_z_prime.size()}, needed : (N)."
        assert len(d_z.size()) == 1, \
            f"Wrong z size, actual : {d_z.size()}, needed : (N)."
        assert d_z_prime.size(0) == d_z.size(0), \
            f"z_prime en z must have the same batch size, z_fake : {d_z_prime.size(0)} and z_real : {d_z.size(0)}"

        return th.mean(th.log2(d_z_prime) + th.log2(1. - d_z), dim=0)

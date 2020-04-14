import abc

import torch as th
import torch.nn as nn


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


####################################################
# Encoder & Decoder class template
####################################################

class Coder(nn.Module):
    def __init__(self, n_fft: int):
        super().__init__()

        self._n_fft = n_fft

    @abc.abstractmethod
    def get_hidden_size(self) -> int:
        return -1

    @abc.abstractmethod
    def division_factor(self) -> int:
        return 0

    @abc.abstractmethod
    def p(self):
        pass


####################################################
# Auto Encoder Small - designed for n_fft = 150
####################################################

class EncoderSmall(nn.Module):
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

    def __str__(self):
        return self.__get_str()

    def __repr__(self):
        return self.__get_str()

    def __get_str(self):
        return f"EncoderSmall_{self.n_channel}"


class DecoderSmall(nn.Module):
    def __init__(self, n_channel):
        super().__init__()

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(n_channel, n_channel + 64,
                               kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(n_channel + 64),
            nn.ConvTranspose1d(n_channel + 64, n_channel + 32,
                               kernel_size=5, stride=2, output_padding=1, padding=2),
            nn.BatchNorm1d(n_channel + 32),
            nn.ConvTranspose1d(n_channel + 32, n_channel,
                               kernel_size=3, padding=1)
        )

        self.n_channel = n_channel

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.n_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.n_channel}."
        return self.cnn_tr_dec(x)

    def __str__(self):
        return self.__get_str()

    def __repr__(self):
        return self.__get_str()

    def __get_str(self):
        return f"DecoderSmall_{self.n_channel}"


####################################################
# Auto Encoder 1 - designed for n_fft = 147
####################################################

class Encoder1(nn.Module):
    def __init__(self, n_channel):
        super().__init__()

        n_layer = 4

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(n_channel, n_channel + int(n_channel / n_layer),
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(n_channel + int(n_channel / n_layer)),
            nn.Conv1d(n_channel + int(n_channel / n_layer),
                      n_channel + int(2 * n_channel / n_layer),
                      kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(n_channel + int(2 * n_channel / n_layer)),
            nn.Conv1d(n_channel + int(2 * n_channel / n_layer),
                      n_channel + int(3 * n_channel / n_layer),
                      kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(n_channel + int(3 * n_channel / n_layer)),
            nn.Conv1d(n_channel + int(3 * n_channel / n_layer),
                      n_channel + int(4 * n_channel / n_layer),
                      kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(n_channel + int(4 * n_channel / n_layer))

        )

        self.n_channel = n_channel

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.n_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.n_channel}."
        return self.cnn_enc(x)

    def __str__(self):
        return self.__get_str()

    def __repr__(self):
        return self.__get_str()

    def __get_str(self):
        return f"Encoder1_{self.n_channel}"


class Decoder1(nn.Module):
    def __init__(self, n_channel):
        super().__init__()

        n_layer = 4

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(n_channel + int(4 * n_channel / n_layer),
                               n_channel + int(3 * n_channel / n_layer),
                               kernel_size=11, stride=5, padding=3),
            nn.BatchNorm1d(n_channel + int(3 * n_channel / n_layer)),
            nn.ConvTranspose1d(n_channel + int(3 * n_channel / n_layer),
                               n_channel + int(2 * n_channel / n_layer),
                               kernel_size=9, stride=4, padding=3, output_padding=1),
            nn.BatchNorm1d(n_channel + int(2 * n_channel / n_layer)),
            nn.ConvTranspose1d(n_channel + int(2 * n_channel / n_layer),
                               n_channel + int(n_channel / n_layer),
                               kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(n_channel + int(n_channel / n_layer)),
            nn.ConvTranspose1d(n_channel + int(n_channel / n_layer),
                               n_channel,
                               kernel_size=3, padding=1)
        )

        self.n_channel = n_channel + int(4 * n_channel / n_layer)

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.n_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.n_channel}."
        return self.cnn_tr_dec(x)

    def __str__(self):
        return self.__get_str()

    def __repr__(self):
        return self.__get_str()

    def __get_str(self):
        return f"Decoder1_{self.n_channel}"


####################################################
# Auto Encoder 2 - designed for n_fft = 147
####################################################

class Encoder2(nn.Module):
    def __init__(self, n_channel: int):
        super().__init__()

        n_layer = 5

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(n_channel, n_channel + int(n_channel / n_layer),
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(n_channel + int(n_channel / n_layer),
                      n_channel + int(2 * n_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(n_channel + int(2 * n_channel / n_layer),
                      n_channel + int(3 * n_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(n_channel + int(3 * n_channel / n_layer),
                      n_channel + int(4 * n_channel / n_layer),
                      kernel_size=7, stride=3, padding=2),
            nn.ReLU(),
            nn.Conv1d(n_channel + int(4 * n_channel / n_layer),
                      n_channel + int(5 * n_channel / n_layer),
                      kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(n_channel + int(5 * n_channel / n_layer))
        )

        self.n_channel = n_channel

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.n_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.n_channel}."
        return self.cnn_enc(x)

    def __str__(self):
        return self.__get_str()

    def __repr__(self):
        return self.__get_str()

    def __get_str(self):
        return f"Encoder2_{self.n_channel}"


class Decoder2(nn.Module):
    def __init__(self, n_channel: int):
        super().__init__()

        n_layer = 5

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(n_channel * 2,
                               n_channel + int(4 * n_channel / n_layer),
                               kernel_size=11, stride=5, padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(n_channel + int(4 * n_channel / n_layer),
                               n_channel + int(3 * n_channel / n_layer),
                               kernel_size=7, stride=3, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(n_channel + int(3 * n_channel / n_layer),
                               n_channel + int(2 * n_channel / n_layer),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_channel + int(2 * n_channel / n_layer),
                               n_channel + int(n_channel / n_layer),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(n_channel + int(n_channel / n_layer)),
            nn.ConvTranspose1d(n_channel + int(n_channel / n_layer),
                               n_channel,
                               kernel_size=3, padding=1)
        )

        self.n_channel = n_channel * 2

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.n_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.n_channel}."
        return self.cnn_tr_dec(x)

    def __str__(self):
        return self.__get_str()

    def __repr__(self):
        return self.__get_str()

    def __get_str(self):
        return f"Decoder2_{self.n_channel}"


####################################################
# Auto Encoder 2 - designed for n_fft = 147
####################################################

class Encoder3(nn.Module):
    def __init__(self, n_channel: int):
        super().__init__()

        n_layer = 4
        self.cnn_enc = nn.Sequential(
            nn.Conv1d(n_channel,
                      n_channel + int(n_channel / n_layer),
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(n_channel + int(n_channel / n_layer),
                      n_channel + int(2 * n_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(n_channel + int(2 * n_channel / n_layer),
                      n_channel + int(3 * n_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(n_channel + int(3 * n_channel / n_layer),
                      n_channel + int(4 * n_channel / n_layer),
                      kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(n_channel + int(4 * n_channel / n_layer))
        )

        self.n_channel = n_channel

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.n_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.n_channel}."
        return self.cnn_enc(x)

    def __str__(self):
        return self.__get_str()

    def __repr__(self):
        return self.__get_str()

    def __get_str(self):
        return f"Encoder3_{self.n_channel}"


class Decoder3(nn.Module):
    def __init__(self, n_channel: int):
        super().__init__()

        n_layer = 4

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(n_channel + int(4 * n_channel / n_layer),
                               n_channel + int(3 * n_channel / n_layer),
                               kernel_size=7, stride=3, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(n_channel + int(3 * n_channel / n_layer),
                               n_channel + int(2 * n_channel / n_layer),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_channel + int(2 * n_channel / n_layer),
                               n_channel + int(n_channel / n_layer),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(n_channel + int(n_channel / n_layer)),
            nn.ConvTranspose1d(n_channel + int(n_channel / n_layer),
                               n_channel,
                               kernel_size=3, padding=1)
        )

        self.n_channel = n_channel + int(4 * n_channel / n_layer)

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.n_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.n_channel}."
        return self.cnn_tr_dec(x)

    def __str__(self):
        return self.__get_str()

    def __repr__(self):
        return self.__get_str()

    def __get_str(self):
        return f"Decoder3_{self.n_channel}"


####################################################
# Discriminator stuff
####################################################

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

    def __str__(self):
        return self.__get_str()

    def __repr__(self):
        return self.__get_str()

    def __get_str(self):
        return f"Discriminator_{self.n_channel}"


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

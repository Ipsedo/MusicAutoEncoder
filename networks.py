import abc
from typing import Tuple

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
    def _get_str(self):
        return "Coder"

    def __str__(self):
        return self._get_str()

    def __repr__(self):
        return self._get_str()


####################################################
# Auto Encoder Small - designed for n_fft = 150
####################################################

class EncoderSmall(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft)

        n_channel = n_fft * 2

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

    def _get_str(self):
        return f"EncoderSmall_{self.n_channel}"

    def get_hidden_size(self) -> int:
        return self.n_channel + 128

    def division_factor(self) -> int:
        return 2 * 3


class DecoderSmall(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft)

        n_channel = n_fft * 2

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(n_channel + 128, n_channel + 64,
                               kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(n_channel + 64),
            nn.ConvTranspose1d(n_channel + 64, n_channel + 32,
                               kernel_size=5, stride=2, output_padding=1, padding=2),
            nn.BatchNorm1d(n_channel + 32),
            nn.ConvTranspose1d(n_channel + 32, n_channel,
                               kernel_size=3, padding=1)
        )

        self.n_channel = n_channel + 128

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.n_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.n_channel}."
        return self.cnn_tr_dec(x)

    def _get_str(self):
        return f"DecoderSmall_{self.n_channel}"

    def get_hidden_size(self) -> int:
        return self.n_channel

    def division_factor(self) -> int:
        return 3 * 2


####################################################
# Auto Encoder 1 - designed for n_fft = 147
####################################################

class Encoder1(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft)

        n_channel = n_fft * 2

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

    def _get_str(self):
        return f"Encoder1_{self.n_channel}"

    def get_hidden_size(self) -> int:
        return self.n_channel * 2

    def division_factor(self) -> int:
        return 3 * 4 * 5


class Decoder1(Coder):
    def __init__(self, n_fft):
        super().__init__(n_fft)

        n_channel = n_fft * 2
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

    def _get_str(self):
        return f"Decoder1_{self.n_channel}"

    def get_hidden_size(self) -> int:
        return self.n_channel

    def division_factor(self) -> int:
        return 3 * 4 * 5


####################################################
# Auto Encoder 2 - designed for n_fft = 147
####################################################

class Encoder2(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft)

        n_channel = n_fft * 2
        n_layer = 5

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(n_channel, n_channel + int(n_channel / n_layer),
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(n_channel + int(n_channel / n_layer)),
            nn.Conv1d(n_channel + int(n_channel / n_layer),
                      n_channel + int(2 * n_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(n_channel + int(2 * n_channel / n_layer)),
            nn.Conv1d(n_channel + int(2 * n_channel / n_layer),
                      n_channel + int(3 * n_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(n_channel + int(3 * n_channel / n_layer)),
            nn.Conv1d(n_channel + int(3 * n_channel / n_layer),
                      n_channel + int(4 * n_channel / n_layer),
                      kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(n_channel + int(4 * n_channel / n_layer)),
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

    def _get_str(self):
        return f"Encoder2_{self.n_channel}"

    def get_hidden_size(self) -> int:
        return self.n_channel * 2

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 5


class Decoder2(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft)

        n_channel = n_fft * 2
        n_layer = 5

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(n_channel * 2,
                               n_channel + int(4 * n_channel / n_layer),
                               kernel_size=11, stride=5, padding=3),
            nn.BatchNorm1d(n_channel + int(4 * n_channel / n_layer)),
            nn.ConvTranspose1d(n_channel + int(4 * n_channel / n_layer),
                               n_channel + int(3 * n_channel / n_layer),
                               kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(n_channel + int(3 * n_channel / n_layer)),
            nn.ConvTranspose1d(n_channel + int(3 * n_channel / n_layer),
                               n_channel + int(2 * n_channel / n_layer),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(n_channel + int(2 * n_channel / n_layer)),
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

    def _get_str(self):
        return f"Decoder2_{self.n_channel}"

    def get_hidden_size(self) -> int:
        return self.n_channel

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 5


####################################################
# Auto Encoder 3 - designed for n_fft = 147
####################################################

class Encoder3(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft)

        n_channel = n_fft * 2
        n_layer = 4

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(n_channel,
                      n_channel + int(n_channel / n_layer),
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(n_channel + int(n_channel / n_layer)),
            nn.Conv1d(n_channel + int(n_channel / n_layer),
                      n_channel + int(2 * n_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(n_channel + int(2 * n_channel / n_layer)),
            nn.Conv1d(n_channel + int(2 * n_channel / n_layer),
                      n_channel + int(3 * n_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(n_channel + int(3 * n_channel / n_layer)),
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

    def _get_str(self):
        return f"Encoder3_{self.n_channel}"

    def get_hidden_size(self) -> int:
        return self.n_channel * 2

    def division_factor(self) -> int:
        return 2 * 2 * 3


class Decoder3(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft)

        n_channel = n_fft * 2
        n_layer = 4

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(n_channel + int(4 * n_channel / n_layer),
                               n_channel + int(3 * n_channel / n_layer),
                               kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(n_channel + int(3 * n_channel / n_layer)),
            nn.ConvTranspose1d(n_channel + int(3 * n_channel / n_layer),
                               n_channel + int(2 * n_channel / n_layer),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(n_channel + int(2 * n_channel / n_layer)),
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

    def _get_str(self):
        return f"Decoder3_{self.n_channel}"

    def get_hidden_size(self) -> int:
        return self.n_channel

    def division_factor(self) -> int:
        return 2 * 2 * 3


####################################################
# Auto Encoder 4 - designed for n_fft = 175 ou 49
####################################################

class Encoder4(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft)

        self.n_channel = n_fft * 2
        n_layer = 5

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(self.n_channel, self.n_channel + int(self.n_channel / n_layer),
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_channel + int(self.n_channel / n_layer)),
            nn.Conv1d(self.n_channel + int(self.n_channel / n_layer),
                      self.n_channel + int(2 * self.n_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(self.n_channel + int(2 * self.n_channel / n_layer)),
            nn.Conv1d(self.n_channel + int(2 * self.n_channel / n_layer),
                      self.n_channel + int(3 * self.n_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(self.n_channel + int(3 * self.n_channel / n_layer)),
            nn.Conv1d(self.n_channel + int(3 * self.n_channel / n_layer),
                      self.n_channel + int(4 * self.n_channel / n_layer),
                      kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(self.n_channel + int(4 * self.n_channel / n_layer)),
            nn.Conv1d(self.n_channel + int(4 * self.n_channel / n_layer),
                      self.n_channel + int(5 * self.n_channel / n_layer),
                      kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(self.n_channel + int(5 * self.n_channel / n_layer))
        )

    def forward(self, x):
        return self.cnn_enc(x)

    def get_hidden_size(self) -> int:
        return self.n_channel * 2

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 3

    def _get_str(self):
        return f"Encoder4_{self.n_channel}"


class Decoder4(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft)

        n_channel = n_fft * 2
        n_layer = 5

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(n_channel * 2,
                               n_channel + int(4 * n_channel / n_layer),
                               kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(n_channel + int(4 * n_channel / n_layer)),
            nn.ConvTranspose1d(n_channel + int(4 * n_channel / n_layer),
                               n_channel + int(3 * n_channel / n_layer),
                               kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(n_channel + int(3 * n_channel / n_layer)),
            nn.ConvTranspose1d(n_channel + int(3 * n_channel / n_layer),
                               n_channel + int(2 * n_channel / n_layer),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(n_channel + int(2 * n_channel / n_layer)),
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

    def _get_str(self):
        return f"Decoder4_{self.n_channel}"

    def get_hidden_size(self) -> int:
        return self.n_channel

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 3


####################################################
# Auto Encoder 4 Bis - designed for n_fft = 175 ou 49
####################################################

class Encoder4Bis(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft)

        self.n_channel = n_fft * 2

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(self.n_channel, int(self.n_channel * 1.5 ** 1),
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(int(self.n_channel * 1.5 ** 1)),
            nn.Conv1d(int(self.n_channel * 1.5 ** 1),
                      int(self.n_channel * 1.5 ** 2),
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(int(self.n_channel * 1.5 ** 2)),
            nn.Conv1d(int(self.n_channel * 1.5 ** 2),
                      int(self.n_channel * 1.5 ** 3),
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(int(self.n_channel * 1.5 ** 3)),
            nn.Conv1d(int(self.n_channel * 1.5 ** 3),
                      int(self.n_channel * 1.5 ** 4),
                      kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(int(self.n_channel * 1.5 ** 4)),
            nn.Conv1d(int(self.n_channel * 1.5 ** 4),
                      int(self.n_channel * 1.5 ** 5),
                      kernel_size=7, stride=3, padding=3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.cnn_enc(x)

    def get_hidden_size(self) -> int:
        return int(self.n_channel * 1.5 ** 5)

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 3

    def _get_str(self):
        return f"EncoderBis4_{self.n_channel}"


class Decoder4Bis(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft)

        n_channel = n_fft * 2

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 5),
                               int(n_channel * 1.5 ** 4),
                               kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(int(n_channel * 1.5 ** 4)),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 4),
                               int(n_channel * 1.5 ** 3),
                               kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(int(n_channel * 1.5 ** 3)),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 3),
                               int(n_channel * 1.5 ** 2),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(int(n_channel * 1.5 ** 2)),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 2),
                               int(n_channel * 1.5 ** 1),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(int(n_channel * 1.5 ** 1)),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 1),
                               n_channel,
                               kernel_size=3, padding=1)
        )

        self.n_channel = int(n_channel * 1.5 ** 5)

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.n_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.n_channel}."
        return self.cnn_tr_dec(x)

    def _get_str(self):
        return f"DecoderBis4_{self.n_channel}"

    def get_hidden_size(self) -> int:
        return self.n_channel

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 3


####################################################
# Auto Encoder 2 Bis - designed for n_fft = 49
####################################################

class Encoder2Bis(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft)

        n_channel = n_fft * 2
        n_layer = 5

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(n_channel, int(n_channel * 1.5 ** 1),
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(int(n_channel * 1.5 ** 1)),
            nn.Conv1d(int(n_channel * 1.5 ** 1),
                      int(n_channel * 1.5 ** 2),
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(int(n_channel * 1.5 ** 2)),
            nn.Conv1d(int(n_channel * 1.5 ** 2),
                      int(n_channel * 1.5 ** 3),
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(int(n_channel * 1.5 ** 3)),
            nn.Conv1d(int(n_channel * 1.5 ** 3),
                      int(n_channel * 1.5 ** 4),
                      kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(int(n_channel * 1.5 ** 4)),
            nn.Conv1d(int(n_channel * 1.5 ** 4),
                      int(n_channel * 1.5 ** 5),
                      kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(n_channel + int(5 * n_channel / n_layer)),

        )

        self.n_channel = n_channel

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.n_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.n_channel}."
        return self.cnn_enc(x)

    def _get_str(self):
        return f"Encoder2Bis_{self.n_channel}"

    def get_hidden_size(self) -> int:
        return int(self.n_channel * 1.5 ** 5)

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 5


class Decoder2Bis(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft)

        n_channel = n_fft * 2
        n_layer = 5

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 5),
                               int(n_channel * 1.5 ** 4),
                               kernel_size=11, stride=5, padding=3),
            nn.BatchNorm1d(int(n_channel * 1.5 ** 4)),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 4),
                               int(n_channel * 1.5 ** 3),
                               kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(int(n_channel * 1.5 ** 3)),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 3),
                               int(n_channel * 1.5 ** 2),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(int(n_channel * 1.5 ** 2)),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 2),
                               int(n_channel * 1.5 ** 1),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(int(n_channel * 1.5 ** 1)),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 1),
                               n_channel,
                               kernel_size=3, padding=1)
        )

        self.n_channel = int(n_channel * 1.5 ** 5)

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.n_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.n_channel}."
        return self.cnn_tr_dec(x)

    def _get_str(self):
        return f"Decoder2Bis_{self.n_channel}"

    def get_hidden_size(self) -> int:
        return self.n_channel

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 5


####################################################
# Coder maker
####################################################

class CoderMaker:
    def __init__(self):
        self.__coder_types = ["encoder", "decoder"]
        self.__models = ["small", "1", "2", "3", "4", "4bis", "2bis"]
        self.__model_maker = {
            "small_encoder": EncoderSmall,
            "small_decoder": DecoderSmall,
            "1_encoder": Encoder1,
            "1_decoder": Decoder1,
            "2_encoder": Encoder2,
            "2_decoder": Decoder2,
            "3_encoder": Encoder3,
            "3_decoder": Decoder3,
            "4_encoder": Encoder4,
            "4_decoder": Decoder4,
            "4bis_encoder": Encoder4Bis,
            "4bis_decoder": Decoder4Bis,
            "2bis_encoder": Encoder2Bis,
            "2bis_decoder": Decoder2Bis
        }

    @property
    def coder_types(self):
        return self.__coder_types

    @property
    def models(self):
        return self.__models

    def __getitem__(self, coder_model_nfft: Tuple[str, str, int]) -> Coder:
        coder_type = coder_model_nfft[0]
        model = coder_model_nfft[1]
        n_fft = coder_model_nfft[2]
        assert coder_type in self.__coder_types, \
            f"Wrong Coder type : {coder_type}, possible = {self.__coder_types}"
        assert model in self.__models, \
            f"Wrong Coder model : {model}, possible = {self.__models}"
        assert n_fft > 0, f"Wrong FFT value : {n_fft}, must be > 0."

        return self.__model_maker[model + "_" + coder_type](n_fft)


####################################################
# Discriminator stuff
####################################################

class Discriminator(nn.Module):
    def __init__(self, n_channel):
        super().__init__()

        self.n_channel = n_channel

        self.lin_dicr = nn.Sequential(
            nn.Linear(self.n_channel, int(self.n_channel * 1.2)),
            nn.ReLU(),
            nn.Linear(int(self.n_channel * 1.2), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        assert len(x.size()) == 2, \
            f"Wrong size len, actual : {len(x.size())}, needed : 2."
        assert x.size(1) == self.n_channel, \
            f"Wrong size, actual : {x.size()}, needed : (N, {self.n_channel})."

        return self.lin_dicr(x).view(-1)

    def __str__(self):
        return self.__get_str()

    def __repr__(self):
        return self.__get_str()

    def __get_str(self):
        return f"Discriminator_{self.n_channel}"


####################################################
# Discriminator CNN
# designed for nFFT = 49 and sampling_rate = 44100
####################################################

class DiscriminatorCNN(nn.Module):
    def __init__(self, n_fft: int):
        super().__init__()

        self.n_channel = n_fft * 2

        self.cnn = nn.Sequential(
            nn.Conv1d(self.n_channel,
                      int(self.n_channel * 1.2),
                      kernel_size=3, padding=1),
            nn.CELU(),
            nn.Conv1d(int(self.n_channel * 1.2),
                      int(self.n_channel * 1.2 ** 2),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(int(self.n_channel * 1.2 ** 2),
                      int(self.n_channel * 1.2 ** 3),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(int(self.n_channel * 1.2 ** 3),
                      int(self.n_channel * 1.2 ** 4),
                      kernel_size=7, stride=3, padding=3),
            nn.CELU(),
            nn.Conv1d(int(self.n_channel * 1.2 ** 4),
                      int(self.n_channel * 1.2 ** 5),
                      kernel_size=7, stride=3, padding=3),
            nn.CELU()
        )

        self.classif = nn.Sequential(
            nn.Linear(int(self.n_channel * 1.2 ** 5) * 25, 4096 * 2),
            nn.CELU(),
            nn.Linear(4096 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.flatten(1, 2)
        out = self.classif(out)
        return out.view(-1)

    def __str__(self):
        return self.__get_str()

    def __repr__(self):
        return self.__get_str()

    def __get_str(self):
        return f"DiscriminatorCNN_{self.n_channel}"


def discriminator_loss(d_z_prime, d_z):
    assert len(d_z_prime.size()) == 1, \
        f"Wrong z_prime size, actual : {d_z_prime.size()}, needed : (N)."
    assert len(d_z.size()) == 1, \
        f"Wrong z size, actual : {d_z.size()}, needed : (N)."
    assert d_z_prime.size(0) == d_z.size(0), \
        f"z_prime en z must have the same batch size, z_fake : {d_z_prime.size(0)} and z_real : {d_z.size(0)}"

    return th.mean(-(th.log2(d_z_prime) + th.log2(1. - d_z)), dim=0)


def generator_loss(d_z):
    assert len(d_z.size()) == 1, \
        f"Wrong z size, actual : {d_z.size()}, needed : (N)."

    return -th.mean(th.log2(d_z), dim=0)


def generator_loss_2(d_z):
    assert len(d_z.size()) == 1, \
        f"Wrong z size, actual : {d_z.size()}, needed : (N)."

    return th.mean(th.log2(1. - d_z), dim=0)

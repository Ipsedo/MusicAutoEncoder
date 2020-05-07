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
    def __init__(self, n_fft: int, in_channel: int):
        super().__init__()

        self._n_fft = n_fft
        self.in_channel = in_channel

    @abc.abstractmethod
    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        raise NotImplementedError("Coder._forward is abstract class !")

    @abc.abstractmethod
    def hidden_channels(self) -> int:
        raise NotImplementedError("Coder.hidden_channels is abstract class !")

    @abc.abstractmethod
    def division_factor(self) -> int:
        raise NotImplementedError("Coder.division_factor is abstract class !")

    @abc.abstractmethod
    def _get_str(self):
        raise NotImplementedError("Coder._get_str is abstract class !")

    def forward(self, x):
        assert len(x.size()) == 3, \
            f"Wrong input size length, actual : {len(x.size())}, needed : {3}."
        assert x.size(1) == self.in_channel, \
            f"Wrong channel number, actual : {x.size(1)}, needed : {self.in_channel}."

        return self._coder_forward(x)

    def __str__(self):
        return self._get_str()

    def __repr__(self):
        return self._get_str()


####################################################
# Auto Encoder Small - designed for n_fft = 150
####################################################

class EncoderSmall(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft, n_fft * 2)

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(self.in_channel, self.in_channel + 32,
                      kernel_size=3, padding=1),
            nn.CELU(),
            nn.Conv1d(self.in_channel + 32, self.in_channel + 64,
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(self.in_channel + 64, self.in_channel + 128,
                      kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(self.in_channel + 128)
        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_enc(x)

    def _get_str(self):
        return f"EncoderSmall_{self.in_channel}"

    def hidden_channels(self) -> int:
        return self.in_channel + 128

    def division_factor(self) -> int:
        return 2 * 3


class DecoderSmall(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft, n_fft * 2 + 128)

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(self.in_channel, self.in_channel - 64,
                               kernel_size=7, stride=3, padding=2),
            nn.CELU(),
            nn.ConvTranspose1d(self.in_channel - 64, self.in_channel - (64 + 32),
                               kernel_size=5, stride=2, output_padding=1, padding=2),
            nn.CELU(),
            nn.ConvTranspose1d(self.in_channel - (64 + 32), self.in_channel - 128,
                               kernel_size=3, padding=1)
        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_tr_dec(x)

    def _get_str(self):
        return f"DecoderSmall_{self.in_channel}"

    def hidden_channels(self) -> int:
        return self.in_channel

    def division_factor(self) -> int:
        return 3 * 2


####################################################
# Auto Encoder 1 - designed for n_fft = 147
####################################################

class Encoder1(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft, n_fft * 2)

        n_layer = 4

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(self.in_channel, self.in_channel + int(self.in_channel / n_layer),
                      kernel_size=3, padding=1),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(self.in_channel / n_layer),
                      self.in_channel + int(2 * self.in_channel / n_layer),
                      kernel_size=7, stride=3, padding=3),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(2 * self.in_channel / n_layer),
                      self.in_channel + int(3 * self.in_channel / n_layer),
                      kernel_size=9, stride=4, padding=4),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(3 * self.in_channel / n_layer),
                      self.in_channel + int(4 * self.in_channel / n_layer),
                      kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(self.in_channel + int(4 * self.in_channel / n_layer))

        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_enc(x)

    def _get_str(self):
        return f"Encoder1_{self.in_channel}"

    def hidden_channels(self) -> int:
        return self.in_channel * 2

    def division_factor(self) -> int:
        return 3 * 4 * 5


class Decoder1(Coder):
    def __init__(self, n_fft):
        super().__init__(n_fft, n_fft * 4)

        n_channel = n_fft * 2
        n_layer = 4

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(n_channel + int(4 * n_channel / n_layer),
                               n_channel + int(3 * n_channel / n_layer),
                               kernel_size=11, stride=5, padding=3),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(3 * n_channel / n_layer),
                               n_channel + int(2 * n_channel / n_layer),
                               kernel_size=9, stride=4, padding=3, output_padding=1),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(2 * n_channel / n_layer),
                               n_channel + int(n_channel / n_layer),
                               kernel_size=7, stride=3, padding=2),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(n_channel / n_layer),
                               n_channel,
                               kernel_size=3, padding=1)
        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_tr_dec(x)

    def _get_str(self):
        return f"Decoder1_{self.in_channel}"

    def hidden_channels(self) -> int:
        return self.in_channel

    def division_factor(self) -> int:
        return 3 * 4 * 5


####################################################
# Auto Encoder 2 - designed for n_fft = 147
####################################################

class Encoder2(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft, n_fft * 2)

        n_layer = 5

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(self.in_channel, self.in_channel + int(self.in_channel / n_layer),
                      kernel_size=3, padding=1),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(self.in_channel / n_layer),
                      self.in_channel + int(2 * self.in_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(2 * self.in_channel / n_layer),
                      self.in_channel + int(3 * self.in_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(3 * self.in_channel / n_layer),
                      self.in_channel + int(4 * self.in_channel / n_layer),
                      kernel_size=7, stride=3, padding=2),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(4 * self.in_channel / n_layer),
                      self.in_channel + int(5 * self.in_channel / n_layer),
                      kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(self.in_channel + int(5 * self.in_channel / n_layer))
        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_enc(x)

    def _get_str(self):
        return f"Encoder2_{self.in_channel}"

    def hidden_channels(self) -> int:
        return self.in_channel * 2

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 5


class Decoder2(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft, n_fft * 4)

        n_channel = n_fft * 2
        n_layer = 5

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(n_channel * 2,
                               n_channel + int(4 * n_channel / n_layer),
                               kernel_size=11, stride=5, padding=3),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(4 * n_channel / n_layer),
                               n_channel + int(3 * n_channel / n_layer),
                               kernel_size=7, stride=3, padding=2),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(3 * n_channel / n_layer),
                               n_channel + int(2 * n_channel / n_layer),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(2 * n_channel / n_layer),
                               n_channel + int(n_channel / n_layer),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(n_channel / n_layer),
                               n_channel,
                               kernel_size=3, padding=1)
        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_tr_dec(x)

    def _get_str(self):
        return f"Decoder2_{self.in_channel}"

    def hidden_channels(self) -> int:
        return self.in_channel

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 5


####################################################
# Auto Encoder 3 - designed for n_fft = 147
####################################################

class Encoder3(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft, n_fft * 2)
        
        n_layer = 4

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(self.in_channel,
                      self.in_channel + int(self.in_channel / n_layer),
                      kernel_size=3, padding=1),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(self.in_channel / n_layer),
                      self.in_channel + int(2 * self.in_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(2 * self.in_channel / n_layer),
                      self.in_channel + int(3 * self.in_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(3 * self.in_channel / n_layer),
                      self.in_channel + int(4 * self.in_channel / n_layer),
                      kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(self.in_channel + int(4 * self.in_channel / n_layer))
        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_enc(x)

    def _get_str(self):
        return f"Encoder3_{self.in_channel}"

    def hidden_channels(self) -> int:
        return self.in_channel * 2

    def division_factor(self) -> int:
        return 2 * 2 * 3


class Decoder3(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft, n_fft * 4)

        n_channel = n_fft * 2
        n_layer = 4

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(n_channel + int(4 * n_channel / n_layer),
                               n_channel + int(3 * n_channel / n_layer),
                               kernel_size=7, stride=3, padding=2),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(3 * n_channel / n_layer),
                               n_channel + int(2 * n_channel / n_layer),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(2 * n_channel / n_layer),
                               n_channel + int(n_channel / n_layer),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(n_channel / n_layer),
                               n_channel,
                               kernel_size=3, padding=1)
        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_tr_dec(x)

    def _get_str(self):
        return f"Decoder3_{self.in_channel}"

    def hidden_channels(self) -> int:
        return self.in_channel

    def division_factor(self) -> int:
        return 2 * 2 * 3


####################################################
# Auto Encoder 4 - designed for n_fft = 175 ou 49
####################################################

class Encoder4(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft, n_fft * 2)

        n_layer = 5

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(self.in_channel, self.in_channel + int(self.in_channel / n_layer),
                      kernel_size=3, padding=1),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(self.in_channel / n_layer),
                      self.in_channel + int(2 * self.in_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(2 * self.in_channel / n_layer),
                      self.in_channel + int(3 * self.in_channel / n_layer),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(3 * self.in_channel / n_layer),
                      self.in_channel + int(4 * self.in_channel / n_layer),
                      kernel_size=7, stride=3, padding=3),
            nn.CELU(),
            nn.Conv1d(self.in_channel + int(4 * self.in_channel / n_layer),
                      self.in_channel + int(5 * self.in_channel / n_layer),
                      kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(self.in_channel + int(5 * self.in_channel / n_layer))
        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_enc(x)

    def hidden_channels(self) -> int:
        return self.in_channel * 2

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 3

    def _get_str(self):
        return f"Encoder4_{self.in_channel}"


class Decoder4(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft, n_fft * 4)

        n_channel = n_fft * 2
        n_layer = 5

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(n_channel * 2,
                               n_channel + int(4 * n_channel / n_layer),
                               kernel_size=7, stride=3, padding=2),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(4 * n_channel / n_layer),
                               n_channel + int(3 * n_channel / n_layer),
                               kernel_size=7, stride=3, padding=2),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(3 * n_channel / n_layer),
                               n_channel + int(2 * n_channel / n_layer),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(2 * n_channel / n_layer),
                               n_channel + int(n_channel / n_layer),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.CELU(),
            nn.ConvTranspose1d(n_channel + int(n_channel / n_layer),
                               n_channel,
                               kernel_size=3, padding=1)
        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_tr_dec(x)

    def _get_str(self):
        return f"Decoder4_{self.in_channel}"

    def hidden_channels(self) -> int:
        return self.in_channel

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 3


####################################################
# Auto Encoder 4 Bis - designed for n_fft = 175 ou 49
####################################################

class Encoder4Bis(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft, n_fft * 2)

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(self.in_channel, int(self.in_channel * 1.5 ** 1),
                      kernel_size=3, padding=1),
            nn.CELU(),
            nn.Conv1d(int(self.in_channel * 1.5 ** 1),
                      int(self.in_channel * 1.5 ** 2),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(int(self.in_channel * 1.5 ** 2),
                      int(self.in_channel * 1.5 ** 3),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(int(self.in_channel * 1.5 ** 3),
                      int(self.in_channel * 1.5 ** 4),
                      kernel_size=7, stride=3, padding=3),
            nn.CELU(),
            nn.Conv1d(int(self.in_channel * 1.5 ** 4),
                      int(self.in_channel * 1.5 ** 5),
                      kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(int(self.in_channel * 1.5 ** 5))
            # TODO test batch norm only here
        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_enc(x)

    def hidden_channels(self) -> int:
        return int(self.in_channel * 1.5 ** 5)

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 3

    def _get_str(self):
        return f"EncoderBis4_{self.in_channel}"


class Decoder4Bis(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft, int(n_fft * 2 * 1.5 ** 5))

        n_channel = n_fft * 2

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 5),
                               int(n_channel * 1.5 ** 4),
                               kernel_size=7, stride=3, padding=2),
            nn.CELU(),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 4),
                               int(n_channel * 1.5 ** 3),
                               kernel_size=7, stride=3, padding=2),
            nn.CELU(),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 3),
                               int(n_channel * 1.5 ** 2),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.CELU(),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 2),
                               int(n_channel * 1.5 ** 1),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.CELU(),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 1),
                               n_channel,
                               kernel_size=3, padding=1)
        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_tr_dec(x)

    def _get_str(self):
        return f"DecoderBis4_{self.in_channel}"

    def hidden_channels(self) -> int:
        return self.in_channel

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 3


####################################################
# Auto Encoder 2 Bis - designed for n_fft = 49
####################################################

class Encoder2Bis(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft, n_fft * 2)

        n_layer = 5

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(self.in_channel, int(self.in_channel * 1.5 ** 1),
                      kernel_size=3, padding=1),
            nn.CELU(),
            nn.Conv1d(int(self.in_channel * 1.5 ** 1),
                      int(self.in_channel * 1.5 ** 2),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(int(self.in_channel * 1.5 ** 2),
                      int(self.in_channel * 1.5 ** 3),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(int(self.in_channel * 1.5 ** 3),
                      int(self.in_channel * 1.5 ** 4),
                      kernel_size=7, stride=3, padding=2),
            nn.CELU(),
            nn.Conv1d(int(self.in_channel * 1.5 ** 4),
                      int(self.in_channel * 1.5 ** 5),
                      kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(self.in_channel + int(5 * self.in_channel / n_layer)),

        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_enc(x)

    def _get_str(self):
        return f"Encoder2Bis_{self.in_channel}"

    def hidden_channels(self) -> int:
        return int(self.in_channel * 1.5 ** 5)

    def division_factor(self) -> int:
        return 2 * 2 * 3 * 5


class Decoder2Bis(Coder):
    def __init__(self, n_fft: int):
        super().__init__(n_fft, int(n_fft * 2 * 1.5 ** 5))

        n_channel = n_fft * 2

        self.cnn_tr_dec = nn.Sequential(
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 5),
                               int(n_channel * 1.5 ** 4),
                               kernel_size=11, stride=5, padding=3),
            nn.CELU(),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 4),
                               int(n_channel * 1.5 ** 3),
                               kernel_size=7, stride=3, padding=2),
            nn.CELU(),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 3),
                               int(n_channel * 1.5 ** 2),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.CELU(),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 2),
                               int(n_channel * 1.5 ** 1),
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.CELU(),
            nn.ConvTranspose1d(int(n_channel * 1.5 ** 1),
                               n_channel,
                               kernel_size=3, padding=1)
        )

    def _coder_forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn_tr_dec(x)

    def _get_str(self):
        return f"Decoder2Bis_{self.in_channel}"

    def hidden_channels(self) -> int:
        return self.in_channel

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
# and time split = 1 second
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


####################################################
# DumbDiscriminator CNN
# designed for nFFT = 49 and sampling_rate = 44100
# and time split = 1 second
####################################################

class DumbDiscriminatorCNN(nn.Module):
    def __init__(self, n_fft: int):
        super().__init__()

        self.n_channel = n_fft * 2

        self.cnn = nn.Sequential(
            nn.Conv1d(self.n_channel,
                      int(self.n_channel * 1.25),
                      kernel_size=3, padding=1),
            nn.CELU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(int(self.n_channel * 1.25),
                      int(self.n_channel * 1.25 ** 2),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(int(self.n_channel * 1.25 ** 2),
                      int(self.n_channel * 1.25 ** 3),
                      kernel_size=7, stride=3, padding=3),
            nn.CELU(),
            nn.MaxPool1d(3, 3)
        )

        self.classif = nn.Sequential(
            nn.Linear(int(self.n_channel * 1.25 ** 3) * 25, 4096 * 2),
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


####################################################
# Discriminator Hidden CNN
# designed for nFFT = 49 and sampling_rate = 44100
# and time split = 10 seconds,
# division factor equal to archi 4bis
####################################################

class DiscriminatorHidden4bisCNN(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()

        self.n_channel = hidden_channels

        self.cnn = nn.Sequential(
            nn.Conv1d(self.n_channel,
                      int(self.n_channel * 0.6),
                      kernel_size=3, padding=1),
            nn.CELU(),
            nn.Conv1d(int(self.n_channel * 0.6),
                      int(self.n_channel * 0.6 ** 2),
                      kernel_size=5, stride=2, padding=2),
            nn.CELU(),
            nn.Conv1d(int(self.n_channel * 0.6 ** 2),
                      int(self.n_channel * 0.6 ** 3),
                      kernel_size=7, stride=5, padding=3),
            nn.CELU()
        )

        self.classif = nn.Sequential(
            nn.Linear(int(self.n_channel * 0.6 ** 3) * 25, 4096),
            nn.CELU(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.flatten(1, 2)
        out = self.classif(out)
        return out.view(-1)

    def input_size(self) -> int:
        return 10 * 44100 // 49 // (2 * 2 * 3 * 3)

    def __str__(self):
        return self.__get_str()

    def __repr__(self):
        return self.__get_str()

    def __get_str(self):
        return f"DiscriminatorCNN_{self.n_channel}"


def discriminator_loss(y_real: th.Tensor, y_fake: th.Tensor) -> th.Tensor:
    assert len(y_real.size()) == 1, \
        f"Wrong y_real size, actual : {y_real.size()}, needed : (N)."
    assert len(y_fake.size()) == 1, \
        f"Wrong y_fake size, actual : {y_fake.size()}, needed : (N)."
    assert y_real.size(0) == y_fake.size(0), \
        f"y_real and y_fake must have the same batch size, y_real : {y_real.size(0)} and y_fake : {y_fake.size(0)}"

    return -th.mean(th.log2(y_real) + th.log2(1. - y_fake), dim=0)


def generator_loss(y_fake: th.Tensor) -> th.Tensor:
    assert len(y_fake.size()) == 1, \
        f"Wrong y_fake size, actual : {y_fake.size()}, needed : (N)."

    return -th.mean(th.log2(y_fake), dim=0)


def generator_loss_2(d_z):
    assert len(d_z.size()) == 1, \
        f"Wrong z size, actual : {d_z.size()}, needed : (N)."

    return th.mean(th.log2(1. - d_z), dim=0)

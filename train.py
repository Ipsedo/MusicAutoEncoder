import argparse
from os import listdir
from os.path import join, splitext
from typing import Tuple

from multiprocessing import Pool

from tqdm import tqdm

import torch as th
import torch.nn as nn

import numpy as np

import read_audio


class Encoder(nn.Module):
    def __init__(self, n_channel: int):
        super().__init__()

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(n_channel, 256 + 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(256 + 32, 256 + 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(256 + 64, 256 + 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(4, 4)
        )

    def forward(self, x):
        return self.cnn_enc(x)


class ConstantUnpool(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size

        if stride != kernel_size:
            raise NotImplementedError("Different kernel size and stride is not implemented")

    def forward(self, x):
        return th.cat([x for _ in range(self.kernel_size)], dim=2)


class Decoder(nn.Module):
    def __init__(self, n_channel):
        super().__init__()

        self.cnn_tr_dec = nn.Sequential(
            ConstantUnpool(4, 4),
            nn.ConvTranspose1d(n_channel, 256 + 64, kernel_size=5, stride=2),
            nn.ReLU(),
            ConstantUnpool(4, 4),
            nn.ConvTranspose1d(256 + 64, 256 + 32, kernel_size=5, stride=2),
            nn.ReLU(),
            ConstantUnpool(2, 2),
            nn.Conv1d(256 + 32, 256, kernel_size=3)
        )

    def forward(self, x):
        return self.cnn_tr_dec(x)


def __read_one_wav(wav_file: str, split_length: int = 10000) -> Tuple[int, np.ndarray]:
    return read_audio.open_wav(wav_file, split_length)


def __fft_one_sample(t: Tuple[int, np.ndarray]) -> np.ndarray:
    return read_audio.fft_raw_audio(t[1], 128)


def main() -> None:
    parser = argparse.ArgumentParser("Train audio auto-encoder")

    parser.add_argument("-d", "--data-root", type=str, required=True, dest="data_root")
    parser.add_argument("--split-length", type=int, default=10000, dest="split_length")

    args = parser.parse_args()

    data_root_path = args.data_root
    split_length = args.split_length

    pool = Pool(16)

    print("Reading wav....")
    wav_files = [join(data_root_path, f) for f in listdir(data_root_path) if splitext(f)[-1] == ".wav"]
    raw_data = pool.map(__read_one_wav, wav_files)

    print("Computing spectrogram...")
    fft_data = pool.map(__fft_one_sample, raw_data)
    fft_data = np.concatenate(fft_data, axis=0)
    fft_data = fft_data.transpose((0, 2, 1))

    data_real = np.real(fft_data)
    data_img = np.imag(fft_data)

    data = np.concatenate([data_real, data_img], axis=1)

    print(data.shape)

    data_th = th.tensor(data[0:2]).to(th.float)

    enc = Encoder(256)
    dec = Decoder(384)

    out_enc = enc(data_th)
    print(out_enc.size()) # torch.Size([26, 384, 26])

    out_dec = dec(out_enc)
    print(out_dec.size())


if __name__ == "__main__":
    main()

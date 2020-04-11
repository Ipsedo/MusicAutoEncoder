import argparse
from os import listdir, mkdir
from os.path import join, splitext, exists, isdir
from typing import Tuple

from multiprocessing import Pool

from tqdm import tqdm

import torch as th
import torch.nn as nn

import numpy as np

from math import ceil

import read_audio


class Encoder(nn.Module):
    def __init__(self, n_channel: int):
        super().__init__()

        # input (batch, 65, 3446)

        self.cnn_enc = nn.Sequential(
            nn.Conv1d(n_channel, 256 + 32, kernel_size=3), # (batch, 288, 3444)
            nn.ReLU(),
            nn.MaxPool1d(2, 2), # (batch, 288, 1722)
            nn.Conv1d(256 + 32, 256 + 64, kernel_size=5, stride=2), # (batch, 320, 859)
            nn.ReLU(),
            nn.MaxPool1d(4, 4), # (batch, 320, 214)
            nn.Conv1d(256 + 64, 256 + 128, kernel_size=5, stride=2), # (batch, 384, 105)
            nn.ReLU(),
            nn.MaxPool1d(4, 4) # (batch, 384, 26)
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
            ConstantUnpool(4, 4), # (batch, 384, 104)
            nn.ConstantPad1d((0, 1), 0.), # (batch, 384, 105)
            nn.ConvTranspose1d(n_channel, 256 + 64, kernel_size=5, stride=2), # (batch, 320, 213)
            nn.ConstantPad1d((0, 1), 0.),  # (batch, 320, 214)
            nn.ReLU(),
            ConstantUnpool(4, 4), # (batch 320, 856)
            nn.ConstantPad1d((1, 2), 0.), # (batch 320, 859)
            nn.ConvTranspose1d(256 + 64, 256 + 32, kernel_size=5, stride=2), # (batch, 288, 1721)
            nn.ConstantPad1d((0, 1), 0.), # (batch, 288, 1722)
            nn.ReLU(),
            ConstantUnpool(2, 2), # (batch, 288, 3442)
            nn.ConvTranspose1d(256 + 32, 256, kernel_size=3) # (batch, 288, 3446)
        )

    def forward(self, x):
        return self.cnn_tr_dec(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_channel_enc = 256
        self.in_channel_dec = 384

        self.enc = Encoder(self.in_channel_enc)
        self.dec = Decoder(self.in_channel_dec)

    def forward(self, x):
        out_enc = self.enc(x)
        out_dec = self.dec(out_enc)

        return out_dec


def __read_one_wav(wav_file: str, split_length: int = 10000) -> Tuple[int, np.ndarray]:
    return read_audio.open_wav(wav_file, split_length)


def __fft_one_sample(t: Tuple[int, np.ndarray]) -> np.ndarray:
    return read_audio.fft_raw_audio(t[1], 128)


def main() -> None:
    parser = argparse.ArgumentParser("Train audio auto-encoder")

    parser.add_argument("-d", "--data-root", type=str, required=True, dest="data_root")
    parser.add_argument("--split-length", type=int, default=10000, dest="split_length")
    parser.add_argument("--out-model-dir", type=str, required=True, dest="out_dir")

    args = parser.parse_args()

    data_root_path = args.data_root
    split_length = args.split_length
    out_dir = args.out_dir

    if exists(out_dir) and not isdir(out_dir):
        print("Model out path already exists and is not a directory !")
        exit()
    if not exists(out_dir):
        mkdir(out_dir)

    pool = Pool(16)

    print("Reading wav....")
    wav_files = [join(data_root_path, f) for f in listdir(data_root_path) if splitext(f)[-1] == ".wav"]
    raw_data = pool.map(__read_one_wav, wav_files)

    print("Computing FFT...")
    fft_data = pool.map(__fft_one_sample, raw_data)
    fft_data = np.concatenate(fft_data, axis=0)
    fft_data = fft_data.transpose((0, 2, 1))

    data_real = np.real(fft_data)
    data_img = np.imag(fft_data)

    data = np.concatenate([data_real, data_img], axis=1)

    print(data.shape)

    print("Creating pytorch stuff...")

    ae_m = AutoEncoder()
    loss_fn = nn.MSELoss()

    ae_m.cuda()
    loss_fn.cuda()

    optim = th.optim.SGD(ae_m.parameters(), lr=1e-4)

    batch_size = 8
    nb_batch = ceil(data.shape[-1] / batch_size)

    nb_epoch = 10

    print("Start learning...")
    for e in range(nb_epoch):
        sum_loss = 0
        nb_backward = 0

        tqdm_pbar = tqdm(data)
        for s in tqdm_pbar:

            s = th.tensor(s).to(th.float).cuda().unsqueeze(0)

            out = ae_m(s)

            for b_idx in range(nb_batch):
                i_min = b_idx * batch_size
                i_max = (b_idx + 1) * batch_size
                i_max = i_max if i_max < out.size(-1) else out.size(-1)

                out_b = out[:, :, i_min:i_max]
                y_b = s[:, :, i_min:i_max]

                loss = loss_fn(out_b, y_b)

                optim.zero_grad()
                loss.backward(retain_graph=True)
                optim.step()

                sum_loss += loss.item()
                nb_backward += 1

                tqdm_pbar.set_description("Loss = {:.3f}".format(loss.item()))

        print(f"Epoch {e}, avg_loss = {sum_loss / nb_backward}")

        th.save(ae_m.state_dict(), join(out_dir, f"AutoEncoder_epoch-{e}.th"))
        th.save(optim.state_dict(), join(out_dir, f"optim_epoch-{e}.th"))
        th.save(loss_fn.state_dict(), join(out_dir, f"loss_fn_epoch-{e}.th"))


if __name__ == "__main__":
    main()

import argparse
import sys
from os import mkdir
from os.path import join, exists, isdir

from tqdm import tqdm

import torch as th
import torch.nn as nn

from math import ceil
import random

import auto_encoder


def main() -> None:
    coder_maker = auto_encoder.CoderMaker()

    parser = argparse.ArgumentParser("Train audio auto-encoder")

    parser.add_argument("--archi", type=str, choices=coder_maker.models, dest="archi", required=True)
    parser.add_argument("--nfft", type=int, dest="n_fft", required=True)
    parser.add_argument("--sample-rate", type=int, default=44100, dest="sample_rate")
    parser.add_argument("--seconds", type=int, required=True, dest="seconds")
    parser.add_argument("--tensor-file", type=str, required=True, dest="tensor_file")
    parser.add_argument("--lr-ae", type=float, default=1e-6, dest="lr_auto_encoder")
    parser.add_argument("--lr-disc", type=float, default=1e-9, dest="lr_discriminator")
    parser.add_argument("--out-model-dir", type=str, required=True, dest="out_dir")

    args = parser.parse_args()

    tensor_file = args.tensor_file
    out_dir = args.out_dir
    archi = args.archi
    n_fft = args.n_fft
    sample_rate = args.sample_rate
    seconds = args.seconds
    lr_auto_encoder = args.lr_auto_encoder
    lr_discriminator = args.lr_discriminator

    if exists(out_dir) and not isdir(out_dir):
        print("Model out path already exists and is not a directory !")
        exit()
    if not exists(out_dir):
        mkdir(out_dir)

    print("Opening saved torch Tensor....")
    data = th.load(tensor_file)

    print("Shuffle data...")
    for i in tqdm(range(data.size(0) - 1)):
        j = i + random.randint(0, sys.maxsize) // (sys.maxsize // (data.size(0) - i) + 1)

        data[i, :, :], data[j, :, :] = data[j, :, :], data[i, :, :]

    # data = data[:18000]

    print(data.size())

    print("Creating pytorch stuff...")

    enc = coder_maker["encoder", archi, n_fft]
    dec = coder_maker["decoder", archi, n_fft]

    hidden_length = sample_rate // n_fft // enc.division_factor()
    hidden_channel = enc.get_hidden_size()
    enc = enc.cuda(0)
    dec = dec.cuda(0)

    hidden_length = seconds * hidden_length

    disc = auto_encoder.Discriminator(hidden_channel).cuda(0)

    print(f"Hidden layer size : {hidden_length}")

    disc_loss_fn = auto_encoder.DiscriminatorLoss().cuda(0)
    ae_loss_fn = nn.MSELoss(reduction="none").cuda(0)

    optim_disc = th.optim.Adam(list(enc.parameters()) + list(disc.parameters()), lr=lr_discriminator)
    optim_ae = th.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr_auto_encoder)

    batch_size = 4
    nb_batch = ceil(data.size(0) / batch_size)

    nb_epoch = 10

    print("Start learning...")
    for e in range(nb_epoch):
        sum_loss_ae = 0
        sum_loss_disc = 0

        nb_backward_ae = 0
        nb_backward_disc = 0

        enc.train()
        dec.train()
        disc.train()

        tqdm_pbar = tqdm(range(nb_batch))
        for b_idx in tqdm_pbar:
            i_min = b_idx * batch_size
            i_max = (b_idx + 1) * batch_size
            i_max = i_max if i_max < data.size(0) else data.size(0)

            x_batch = data[i_min:i_max].cuda()

            out_enc = enc(x_batch)

            z = out_enc.permute(0, 2, 1).flatten(0, 1)
            z_prime = th.randn(*z.size(), device=th.device("cuda"), dtype=th.float)

            out_dec = dec(out_enc)

            loss_autoencoder = ae_loss_fn(out_dec, x_batch).mean(dim=1).view(-1)

            batch_size_dec = 32
            nb_batch_dec = ceil(loss_autoencoder.size(0) / batch_size_dec)

            for b_idx_dec in range(nb_batch_dec):
                i_min_dec = b_idx_dec * batch_size_dec
                i_max_dec = (b_idx_dec + 1) * batch_size_dec
                i_max_dec = i_max_dec if i_max_dec < loss_autoencoder.size(0) else loss_autoencoder.size(0)

                loss_ae = loss_autoencoder[i_min_dec:i_max_dec].mean()

                optim_ae.zero_grad()
                loss_ae.backward(retain_graph=True)
                optim_ae.step()

                nb_backward_ae += 1
                sum_loss_ae += loss_ae.item()

            batch_size_disc = 8
            nb_batch_disc = ceil(z.size(0) / batch_size_disc)

            for b_idx_disc in range(nb_batch_disc):
                i_min_disc = b_idx_disc * batch_size_disc
                i_max_disc = (b_idx_disc + 1) * batch_size_disc
                i_max_disc = i_max_disc if i_max_disc < z.size(0) else z.size(0)

                b_z = z[i_min_disc:i_max_disc]
                b_z_prime = z_prime[i_min_disc:i_max_disc]

                d_z = disc(b_z)
                d_z_prime = disc(b_z_prime)

                loss_disc = disc_loss_fn(d_z_prime, d_z)

                optim_disc.zero_grad()
                loss_disc.backward(retain_graph=True)
                optim_disc.step()

                sum_loss_disc += loss_disc.item()
                nb_backward_disc += 1

            tqdm_pbar.set_description(f"Epoch {e:2d} : "
                                      f"loss_ae_avg = {sum_loss_ae / nb_backward_ae:.6f}, "
                                      f"loss_disc_avg = {sum_loss_disc / nb_backward_disc:.6f} ")

        th.save(enc.state_dict(), join(out_dir, f"{enc}_epoch-{e}.th"))
        th.save(dec.state_dict(), join(out_dir, f"{dec}_epoch-{e}.th"))
        th.save(optim_ae.state_dict(), join(out_dir, f"optim_ae_epoch-{e}.th"))
        th.save(ae_loss_fn.state_dict(), join(out_dir, f"loss_ae_fn_epoch-{e}.th"))

        th.save(disc.state_dict(), join(out_dir, f"{disc}_epoch-{e}.th"))
        th.save(optim_disc.state_dict(), join(out_dir, f"optim_disc_epoch-{e}.th"))
        th.save(disc_loss_fn.state_dict(), join(out_dir, f"loss_disc_fn_epoch-{e}.th"))


if __name__ == "__main__":
    main()

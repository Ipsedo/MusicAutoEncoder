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
    parser = argparse.ArgumentParser("Train audio auto-encoder")

    parser.add_argument("--archi", type=str, choices=["small", "1", "2", "3"], dest="archi", required=True)
    parser.add_argument("--n-fft", type=int, dest="n_fft", required=True)
    parser.add_argument("--sample-rate", type=int, default=44100, dest="sample_rate")
    parser.add_argument("--second", type=int, required=True, dest="second")
    parser.add_argument("-d", "--data-file", type=str, required=True, dest="data_file")
    parser.add_argument("--out-model-dir", type=str, required=True, dest="out_dir")

    args = parser.parse_args()

    data_file = args.data_file
    out_dir = args.out_dir
    archi = args.archi
    n_fft = args.n_fft
    sample_rate = args.sample_rate
    second = args.second

    if exists(out_dir) and not isdir(out_dir):
        print("Model out path already exists and is not a directory !")
        exit()
    if not exists(out_dir):
        mkdir(out_dir)

    print("Opening saved torch Tensor....")
    data = th.load(data_file)

    print("Shuffle data...")
    for i in tqdm(range(data.size(0) - 1)):
        j = i + random.randint(0, sys.maxsize) // (sys.maxsize // (data.size(0) - i) + 1)

        data[i, :, :], data[j, :, :] = data[j, :, :], data[i, :, :]

    data = data[:3000]

    print(data.size())

    print("Creating pytorch stuff...")

    if archi == "1":
        enc = auto_encoder.Encoder1(n_fft * 2)
        dec = auto_encoder.Decoder1(n_fft * 2)
        hidden_length = sample_rate // n_fft // 3 // 4 // 5
        hidden_channel = n_fft * 2 * 2
    elif archi == "2":
        enc = auto_encoder.Encoder2(n_fft * 2)
        dec = auto_encoder.Decoder2(n_fft * 2)
        hidden_length = sample_rate // n_fft // 2 // 2 // 3 // 5
        hidden_channel = n_fft * 2 * 2
    elif archi == "3":
        enc = auto_encoder.Encoder3(n_fft * 2)
        dec = auto_encoder.Decoder3(n_fft * 2)
        hidden_length = sample_rate // n_fft // 2 // 2 // 3
        hidden_channel = n_fft * 2 * 2
    elif archi == "small":
        enc = auto_encoder.EncoderSmall(n_fft * 2)
        dec = auto_encoder.DecoderSmall(n_fft * 2)
        hidden_length = sample_rate // n_fft // 2 // 3
        hidden_channel = n_fft * 2 + 128
    else:
        print(f"Unrecognized NN architecture ({archi}).")
        print(f"Will load small CNN")
        enc = auto_encoder.EncoderSmall(n_fft * 2)
        dec = auto_encoder.DecoderSmall(n_fft * 2)
        hidden_length = sample_rate // n_fft // 2 // 3
        hidden_channel = n_fft * 2 + 128

    enc = enc.cuda()
    dec = dec.cuda()

    hidden_length = second * hidden_length

    disc = auto_encoder.Discriminator(hidden_channel).cuda()

    print(f"Hidden layer size : {hidden_length}")

    disc_loss_fn = auto_encoder.DiscriminatorLoss().cuda()
    ae_loss_fn = nn.MSELoss(reduction="none").cuda()

    optim_disc = th.optim.Adam(list(enc.parameters()) + list(disc.parameters()), lr=1e-7)
    optim_ae = th.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-6)

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

            tqdm_pbar.set_description("Epoch {} : loss_ae_avg = {:.6f}, loss_disc_avg = {:.6f} "
                                      .format(e, sum_loss_ae / nb_backward_ae, sum_loss_disc / nb_backward_disc))

        th.save(enc.state_dict(), join(out_dir, f"Encoder2_epoch-{e}.th"))
        th.save(dec.state_dict(), join(out_dir, f"Decoder2_epoch-{e}.th"))
        th.save(optim_ae.state_dict(), join(out_dir, f"optim_ae_epoch-{e}.th"))
        th.save(ae_loss_fn.state_dict(), join(out_dir, f"loss_ae_fn_epoch-{e}.th"))

        th.save(disc.state_dict(), join(out_dir, f"Discriminator_epoch-{e}.th"))
        th.save(optim_disc.state_dict(), join(out_dir, f"optim_disc_epoch-{e}.th"))
        th.save(disc_loss_fn.state_dict(), join(out_dir, f"loss_disc_fn_epoch-{e}.th"))


if __name__ == "__main__":
    main()

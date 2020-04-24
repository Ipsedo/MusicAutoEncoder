import argparse
import sys
from os import mkdir
from os.path import join, exists, isdir

from tqdm import tqdm

import torch as th

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
    parser.add_argument("--lr-gen", type=float, default=6e-5, dest="lr_gen")
    parser.add_argument("--lr-disc", type=float, default=8e-5, dest="lr_discriminator")
    parser.add_argument("--out-model-dir", type=str, required=True, dest="out_dir")

    args = parser.parse_args()

    tensor_file = args.tensor_file
    out_dir = args.out_dir
    archi = args.archi
    n_fft = args.n_fft
    sample_rate = args.sample_rate
    seconds = args.seconds
    lr_gen = args.lr_gen
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

    print(data.size())

    dec = coder_maker["decoder", archi, n_fft]
    disc = auto_encoder.Discriminator(n_fft * 2)

    dec = dec.to(th.device("cuda"))
    disc = disc.to(th.device("cuda"))

    optim_gen = th.optim.Adam(dec.parameters(), lr=lr_gen)
    optim_disc = th.optim.Adam(disc.parameters(), lr=lr_discriminator)

    hidden_size = seconds * sample_rate // n_fft // dec.division_factor()

    assert data.size(2) // dec.division_factor() == hidden_size, \
        f"Wrong hidden size, data : {data.size(2) // dec.division_factor()}, expected : {hidden_size}"

    nb_epoch = 15
    batch_size = 1
    nb_batch = data.size(0) // batch_size

    for e in range(nb_epoch):

        sum_loss_disc = 0
        nb_backward_disc = 0

        sum_loss_gen = 0
        nb_backward_gen = 0

        tqdm_bar = tqdm(range(nb_batch))
        for b_idx in tqdm_bar:

            i_min = b_idx * batch_size
            i_max = (b_idx + 1) * batch_size
            i_max = i_max if i_max < data.size(0) else data.size(0)

            x_real = data[i_min:i_max, :, :].permute(0, 2, 1).flatten(0, 1)

            z_fake = th.randn(x_real.size(0), dec.get_hidden_size(), hidden_size, dtype=th.float).cuda()
            x_fake = dec(z_fake).permute(0, 2, 1).flatten(0, 1)

            batch_size_disc = data.size(2) // 12 # 3
            nb_batch_disc = x_real.size(0) // batch_size_disc

            for b_idx_disc in range(nb_batch_disc):
                i_min_disc = b_idx_disc * batch_size_disc
                i_max_disc = (b_idx_disc + 1) * batch_size_disc
                i_max_disc = i_max_disc if i_max_disc < x_real.size(0) else x_real.size(0)

                if i_max_disc - i_min_disc < 2:
                    continue

                x_r_b = x_real[i_min_disc:i_max_disc].cuda()
                x_f_b = x_fake[i_min_disc:i_max_disc]

                out_real = disc(x_r_b)
                out_fake = disc(x_f_b)

                # Discriminator
                loss = auto_encoder.discriminator_loss(out_real, out_fake)
                optim_disc.zero_grad()
                loss.backward(retain_graph=True)
                optim_disc.step()

                sum_loss_disc += loss.item()
                nb_backward_disc += 1

                # Generator
                loss = auto_encoder.generator_loss(out_fake)
                optim_gen.zero_grad()
                loss.backward(retain_graph=True)
                optim_gen.step()

                sum_loss_gen += loss.item()
                nb_backward_gen += 1

                tqdm_bar.set_description(f"Epoch {e:2d} : "
                                         f"disc_avg = {sum_loss_disc / nb_backward_disc:.6f}, "
                                         f"gen_avg = {sum_loss_gen / nb_backward_gen:.6f}")

        th.save(dec.state_dict(), join(out_dir, f"{dec}_epoch-{e}.th"))
        th.save(optim_gen.state_dict(), join(out_dir, f"optim_gen_epoch-{e}.th"))
        th.save(disc.state_dict(), join(out_dir, f"{disc}_epoch-{e}.th"))
        th.save(optim_disc.state_dict(), join(out_dir, f"optim_disc_epoch-{e}.th"))


if __name__ == '__main__':
    main()

import argparse
import sys
from os import mkdir
from os.path import join, exists, isdir

from tqdm import tqdm

import torch as th
import torch.nn as nn

from math import ceil
import random

import networks


def main() -> None:
    coder_maker = networks.CoderMaker()

    parser = argparse.ArgumentParser("Train audio auto-encoder")

    parser.add_argument("--archi", type=str, choices=coder_maker.models, dest="archi", required=True)
    parser.add_argument("--nfft", type=int, dest="n_fft", required=True)
    parser.add_argument("--sample-rate", type=int, default=44100, dest="sample_rate")
    parser.add_argument("--seconds", type=int, required=True, dest="seconds")
    parser.add_argument("--tensor-file", type=str, required=True, dest="tensor_file")
    parser.add_argument("--lr-ae", type=float, default=6e-5, dest="lr_auto_encoder")
    parser.add_argument("--lr-disc", type=float, default=8e-5, dest="lr_discriminator")
    parser.add_argument("--out-model-dir", type=str, required=True, dest="out_dir")

    subparser = parser.add_subparsers(dest="mode")

    overfit_partser = subparser.add_parser("overfit")
    overfit_partser.add_argument("--encoder-path", type=str, required=True, dest="encoder_path")
    overfit_partser.add_argument("--decoder-path", type=str, required=True, dest="decoder_path")
    overfit_partser.add_argument("--disc-path", type=str, required=True, dest="disc_path")
    overfit_partser.add_argument("--aeoptim-path", type=str, required=True, dest="aeoptim_path")
    overfit_partser.add_argument("--discoptim-path", type=str, required=True, dest="discoptim_path")
    overfit_partser.add_argument("--genoptim-path", type=str, required=True, dest="genoptim_path")

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

    print(data.size())

    print("Creating pytorch stuff...")

    enc = coder_maker["encoder", archi, n_fft]
    dec = coder_maker["decoder", archi, n_fft]

    hidden_length = seconds * sample_rate // n_fft // enc.division_factor()
    hidden_channel = enc.get_hidden_size()
    enc = enc.cuda(0)
    dec = dec.cuda(0)

    disc = networks.Discriminator(hidden_channel).cuda(0)

    print(f"Hidden layer size : {hidden_length}")

    ae_loss_fn = nn.MSELoss(reduction="none").cuda(0)

    optim_ae = th.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr_auto_encoder)
    optim_disc = th.optim.SGD(disc.parameters(), lr=lr_discriminator)
    optim_gen = th.optim.SGD(enc.parameters(), lr=lr_discriminator)

    if args.mode and args.mode == "overfit":
        encoder_path = args.encoder_path
        decoder_path = args.decoder_path
        disc_path = args.disc_path
        aeoptim_path = args.aeoptim_path
        discoptim_path = args.discoptim_path
        genoptim_path = args.genoptim_path

        enc.load_state_dict(th.load(encoder_path))
        dec.load_state_dict(th.load(decoder_path))
        disc.load_state_dict(th.load(disc_path))
        # optim_ae.load_state_dict(th.load(aeoptim_path))
        # optim_disc.load_state_dict(th.load(discoptim_path))
        # optim_gen.load_state_dict(th.load(genoptim_path))

    batch_size = 4
    nb_batch = ceil(data.size(0) / batch_size)

    nb_epoch = 15

    print("Start learning...")
    for e in range(nb_epoch):
        sum_loss_ae = 0
        sum_loss_disc = 0
        sum_loss_gen = 0

        nb_backward_ae = 0
        nb_backward_disc = 0
        nb_backward_gen = 0

        enc.train()
        dec.train()
        disc.train()

        tqdm_pbar = tqdm(range(nb_batch))
        for b_idx in tqdm_pbar:
            i_min = b_idx * batch_size
            i_max = (b_idx + 1) * batch_size
            i_max = i_max if i_max < data.size(0) else data.size(0)

            if i_max - i_min == 1:
                continue

            x_batch = data[i_min:i_max].cuda()

            # Auto Encoder
            enc.train()
            dec.train()

            out_enc = enc(x_batch)
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

            # Discriminator
            enc.eval()
            disc.train()

            out_enc = enc(x_batch)
            z = out_enc.permute(0, 2, 1).flatten(0, 1)
            z_prime = th.randn(*z.size(), device=th.device("cuda"), dtype=th.float)

            batch_size_disc = 4
            nb_batch_disc = ceil(z.size(0) / batch_size_disc)

            for b_idx_disc in range(nb_batch_disc):
                i_min_disc = b_idx_disc * batch_size_disc
                i_max_disc = (b_idx_disc + 1) * batch_size_disc
                i_max_disc = i_max_disc if i_max_disc < z.size(0) else z.size(0)

                if i_max_disc - i_min_disc == 1:
                    continue

                b_z = z[i_min_disc:i_max_disc]
                b_z_prime = z_prime[i_min_disc:i_max_disc]

                d_z = disc(b_z)
                d_z_prime = disc(b_z_prime)

                loss_disc = networks.discriminator_loss(d_z_prime, d_z)

                optim_disc.zero_grad()
                loss_disc.backward(retain_graph=True)
                optim_disc.step()

                sum_loss_disc += loss_disc.item()
                nb_backward_disc += 1

            # Generator
            enc.train()
            disc.eval()

            out_enc = enc(x_batch)
            z = out_enc.permute(0, 2, 1).flatten(0, 1)

            batch_size_gen = 4
            nb_batch_gen = ceil(z.size(0) / batch_size_gen)

            for b_idx_gen in range(nb_batch_gen):
                i_min_gen = b_idx_gen * batch_size_gen
                i_max_gen = (b_idx_gen + 1) * batch_size_gen
                i_max_gen = i_max_gen if i_max_gen < z.size(0) else z.size(0)

                if i_max_gen - i_min_gen == 1:
                    continue

                b_z = z[i_min_gen:i_max_gen]
                d_z = disc(b_z)

                loss_gen = networks.generator_loss(d_z)
                optim_gen.zero_grad()
                loss_gen.backward(retain_graph=True)
                optim_gen.step()

                sum_loss_gen += loss_gen.item()
                nb_backward_gen += 1

            tqdm_pbar.set_description(f"Epoch {e:2d} : "
                                      f"ae_avg = {sum_loss_ae / nb_backward_ae:.6f}, "
                                      f"disc_avg = {sum_loss_disc / nb_backward_disc:.6f}, "
                                      f"gen_avg = {sum_loss_gen / nb_backward_gen:.6f} ")

        th.save(enc.cpu().state_dict(), join(out_dir, f"{enc}_epoch-{e}.th"))
        th.save(dec.cpu().state_dict(), join(out_dir, f"{dec}_epoch-{e}.th"))
        th.save(optim_ae.state_dict(), join(out_dir, f"optim_ae_epoch-{e}.th"))
        th.save(ae_loss_fn.cpu().state_dict(), join(out_dir, f"loss_ae_fn_epoch-{e}.th"))
        th.save(optim_gen.state_dict(), join(out_dir, f"optim_gen_epoch-{e}.th"))

        th.save(disc.cpu().state_dict(), join(out_dir, f"{disc}_epoch-{e}.th"))
        th.save(optim_disc.state_dict(), join(out_dir, f"optim_disc_epoch-{e}.th"))

        enc = enc.cuda(0)
        dec = dec.cuda(0)
        disc = disc.cuda(0)
        ae_loss_fn = ae_loss_fn.cuda(0)


if __name__ == "__main__":
    main()

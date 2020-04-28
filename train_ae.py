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
    parser.add_argument("--out-model-dir", type=str, required=True, dest="out_dir")

    subparser = parser.add_subparsers(dest="mode")

    overfit_partser = subparser.add_parser("overfit")
    overfit_partser.add_argument("--encoder-path", type=str, required=True, dest="encoder_path")
    overfit_partser.add_argument("--decoder-path", type=str, required=True, dest="decoder_path")

    args = parser.parse_args()

    tensor_file = args.tensor_file
    out_dir = args.out_dir
    archi = args.archi
    n_fft = args.n_fft
    sample_rate = args.sample_rate
    seconds = args.seconds
    lr_auto_encoder = args.lr_auto_encoder

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
    hidden_channel = enc.hidden_channels()
    enc = enc.cuda(0)
    dec = dec.cuda(0)

    print(f"Hidden layer size : {hidden_length}")

    ae_loss_fn = nn.MSELoss(reduction="none").cuda(0)

    optim_ae = th.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr_auto_encoder)

    if args.mode and args.mode == "overfit":
        encoder_path = args.encoder_path
        decoder_path = args.decoder_path

        enc.load_state_dict(th.load(encoder_path))
        dec.load_state_dict(th.load(decoder_path))

    batch_size = 4
    nb_batch = ceil(data.size(0) / batch_size)

    nb_epoch = 15

    print("Start learning...")
    for e in range(nb_epoch):
        sum_loss_ae = 0

        nb_backward_ae = 0

        enc.train()
        dec.train()

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

            tqdm_pbar.set_description(f"Epoch {e:2d} : "
                                      f"ae_avg = {sum_loss_ae / nb_backward_ae:.6f} ")

        th.save(enc.cpu().state_dict(), join(out_dir, f"{enc}_epoch-{e}.th"))
        th.save(dec.cpu().state_dict(), join(out_dir, f"{dec}_epoch-{e}.th"))
        th.save(optim_ae.state_dict(), join(out_dir, f"optim_ae_epoch-{e}.th"))
        th.save(ae_loss_fn.cpu().state_dict(), join(out_dir, f"loss_ae_fn_epoch-{e}.th"))

        enc = enc.cuda(0)
        dec = dec.cuda(0)
        ae_loss_fn = ae_loss_fn.cuda(0)


if __name__ == "__main__":
    main()

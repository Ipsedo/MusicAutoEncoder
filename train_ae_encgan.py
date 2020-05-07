import argparse
from os import mkdir
from os.path import exists, isdir, join

from math import ceil
from random import shuffle

from tqdm import tqdm

import torch as th
import torch.nn as nn

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
    parser.add_argument("--lr-gen", type=float, default=8e-5, dest="lr_generator")
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
    lr_generator = args.lr_generator

    if exists(out_dir) and not isdir(out_dir):
        print("Model out path already exists and is not a directory !")
        exit()
    if not exists(out_dir):
        mkdir(out_dir)

    print("Opening saved torch Tensor....")
    data = th.load(tensor_file)

    print(data.size())

    print("Creating pytorch stuff...")

    enc = coder_maker["encoder", archi, n_fft]
    dec = coder_maker["decoder", archi, n_fft]

    enc.cuda(0)
    dec.cuda(0)

    ae_loss_fn = nn.MSELoss()
    ae_loss_fn.cuda(0)

    disc = networks.Discriminator(enc.hidden_channels())

    optim_ae = th.optim.SGD(list(enc.parameters()) + list(dec.parameters()), lr=lr_auto_encoder)
    optim_gen = th.optim.SGD(enc.parameters(), lr=lr_generator)
    optim_disc = th.optim.SGD(disc.parameters(), lr=lr_discriminator)

    nb_epoch = 10

    batch_size = 4
    nb_batch = ceil(data.size(0) / batch_size)

    for e in range(nb_epoch):

        sum_loss_ae = 0
        nb_backward_ae = 0

        sum_loss_disc = 0
        nb_backward_disc = 0

        sum_loss_gen = 0
        nb_backward_gen = 0

        nb_correct_real = 0
        nb_correct_fake = 0
        nb_pass_disc = 0

        batch_idx = list(range(nb_batch))
        shuffle(batch_idx)
        tqdm_bar = tqdm(batch_idx)

        for b_idx in tqdm_bar:
            i_min = b_idx * batch_size
            i_max = (b_idx + 1) * batch_size
            i_max = i_max if i_max < data.size(0) else data.size(0)

            x_batch = data[i_min:i_max]

            # Auto Encoder

            # Discriminator

            # Generator

            tqdm_bar.set_description(f"Epoch {e:2d} : "
                                      f"ae_avg = {sum_loss_ae / nb_backward_ae:.6f}, "
                                      f"disc_avg = {sum_loss_disc / nb_backward_disc:.6f}, "
                                      f"gen_avg = {sum_loss_gen / nb_backward_gen:.6f}, "
                                      f"prec_real = {nb_correct_real / nb_pass_disc:.6f}, "
                                      f"prec_fake = {nb_correct_fake / nb_pass_disc:.6f}")

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

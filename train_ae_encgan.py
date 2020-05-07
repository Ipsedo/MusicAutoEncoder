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
    parser.add_argument("--lr-ae", type=float, default=1e-5, dest="lr_auto_encoder")
    parser.add_argument("--lr-disc", type=float, default=3e-5, dest="lr_discriminator")
    parser.add_argument("--lr-gen", type=float, default=3e-5, dest="lr_generator")
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

    disc = networks.DiscriminatorHidden4bisCNN(enc.hidden_channels())
    disc.cuda(0)

    optim_ae = th.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr_auto_encoder)
    optim_gen = th.optim.SGD(enc.parameters(), lr=lr_generator)
    optim_disc = th.optim.SGD(disc.parameters(), lr=lr_discriminator)

    nb_epoch = 40

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

            hidden_size = seconds * sample_rate // n_fft // enc.division_factor()
            hidden_channels = enc.hidden_channels()

            x_batch = data[i_min:i_max].cuda(0)

            # Auto Encoder
            nb_sec_ae = 1
            x_splitted = th.stack(x_batch.split(nb_sec_ae * sample_rate // n_fft), dim=0).flatten(0, 1)
            enc.train()
            dec.train()

            ae_batch_size = 4
            nb_batch_ae = ceil(x_splitted.size(0) / ae_batch_size)

            for b_ae_idx in range(nb_batch_ae):
                i_min_ae = b_ae_idx * ae_batch_size
                i_max_ae = (b_ae_idx + 1) * ae_batch_size
                i_max_ae = i_max_ae if i_max_ae < x_splitted.size(0) else x_splitted.size(0)

                x = x_splitted[i_min_ae:i_max_ae]

                hidden_repr = enc(x)
                out_ae = dec(hidden_repr)

                loss_ae = ae_loss_fn(out_ae, x)

                optim_ae.zero_grad()
                loss_ae.backward()
                optim_ae.step()

                sum_loss_ae += loss_ae.item()

                nb_backward_ae += 1

            # Discriminator
            enc.eval()
            disc.train()

            z_real = th.randn(i_max - i_min, hidden_channels, hidden_size, device=th.device("cuda:0"), dtype=th.float)
            z_fake = enc(x_batch)

            out_real = disc(z_real)
            out_fake = disc(z_fake)

            loss_disc = networks.discriminator_loss(out_real, out_fake)

            optim_disc.zero_grad()
            loss_disc.backward()
            optim_disc.step()

            sum_loss_disc += loss_disc.item()

            nb_correct_real += (out_real > 0.5).sum().item()
            nb_correct_fake += (out_fake < 0.5).sum().item()

            nb_backward_disc += 1
            nb_pass_disc += out_real.size(0)

            # Generator
            enc.train()
            disc.eval()

            z_fake = enc(x_batch)
            out_fake = disc(z_fake)

            loss_gen = networks.generator_loss(out_fake)

            optim_gen.zero_grad()
            loss_gen.backward()
            optim_gen.step()

            sum_loss_gen += loss_gen.item()

            nb_backward_gen += 1

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

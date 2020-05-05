import argparse
import sys
from os import mkdir
from os.path import join, exists, isdir

import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch as th

import random

import networks


def main() -> None:
    coder_maker = networks.CoderMaker()

    parser = argparse.ArgumentParser("Train audio auto-encoder")

    parser.add_argument("--archi", type=str, choices=coder_maker.models, dest="archi", required=True)
    parser.add_argument("--from-decoder", type=str, dest="from_decoder")
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
    print(data.size())

    gen = coder_maker["decoder", archi, n_fft]

    if args.from_decoder is not None:
        gen.load_state_dict(th.load(args.from_decoder))

    disc = networks.DumbDiscriminatorCNN(n_fft)

    gen = gen.to(th.device("cuda"))
    disc = disc.to(th.device("cuda"))

    optim_gen = th.optim.SGD(gen.parameters(), lr=lr_gen)
    optim_disc = th.optim.SGD(disc.parameters(), lr=lr_discriminator)

    hidden_size = seconds * sample_rate // n_fft // gen.division_factor()

    assert data.size(2) // gen.division_factor() == hidden_size, \
        f"Wrong hidden size, data : {data.size(2) // gen.division_factor()}, expected : {hidden_size}"

    nb_epoch = 20
    batch_size = 4
    nb_batch = data.size(0) // batch_size

    print("Train discriminator")

    losses_disc = []
    losses_gen = []
    acc_real = []
    acc_fake = []

    # Train Disc
    for e in range(nb_epoch):

        # Discriminator
        sum_loss_disc = 0
        nb_backward_disc = 0

        sum_loss_gen = 0
        nb_backward_gen = 0

        nb_correct_real, nb_correct_fake = 0, 0
        nb_pass_disc = 0

        b_idxs = list(range(nb_batch))
        random.shuffle(b_idxs)
        tqdm_bar = tqdm(b_idxs)

        for b_idx in tqdm_bar:
            i_min = b_idx * batch_size
            i_max = (b_idx + 1) * batch_size
            i_max = i_max if i_max < data.size(0) else data.size(0)

            x_real = data[i_min:i_max, :, :].cuda()

            z_fake = th.randn(x_real.size(0), gen.hidden_channels(), hidden_size,
                              dtype=th.float, device=th.device("cuda"))

            # Discriminator
            disc.train()
            gen.eval()

            x_fake = gen(z_fake)

            out_real = disc(x_real)
            out_fake = disc(x_fake)

            nb_correct_real += (out_real > 0.5).sum().item()
            nb_correct_fake += (out_fake <= 0.5).sum().item()
            nb_pass_disc += out_real.size(0)

            optim_disc.zero_grad()
            loss_disc = networks.discriminator_loss(out_real, out_fake)
            loss_disc.backward()
            optim_disc.step()

            sum_loss_disc += loss_disc.item()
            nb_backward_disc += 1

            # Generator
            disc.eval()
            gen.train()

            z_fake = th.randn(i_max - i_min, gen.hidden_channels(), hidden_size, dtype=th.float).cuda()
            x_fake = gen(z_fake)
            out_fake = disc(x_fake)

            optim_gen.zero_grad()
            loss_gen = networks.generator_loss(out_fake)
            loss_gen.backward()
            optim_gen.step()

            sum_loss_gen += loss_gen.item()
            nb_backward_gen += 1

            tqdm_bar.set_description(f"Epoch {e:2d} : "
                                     f"disc_avg = {sum_loss_disc / nb_backward_disc:.6f}, "
                                     f"gen_avg = {sum_loss_gen / nb_backward_gen:.6f}, "
                                     f"prec_real = {nb_correct_real / nb_pass_disc:.6f}, "
                                     f"prec_fake = {nb_correct_fake / nb_pass_disc:.6f}")

            if th.isnan(loss_gen).any():
                print("NaN detected - Exiting :,(")
                exit()

        losses_disc.append(sum_loss_disc / nb_backward_disc)
        losses_gen.append(sum_loss_gen / nb_backward_gen)
        acc_fake.append(nb_correct_fake / nb_pass_disc)
        acc_real.append(nb_correct_real / nb_pass_disc)

        th.save(gen.cpu().state_dict(), join(out_dir, f"{gen}_epoch-{e}.th"))
        th.save(optim_gen.state_dict(), join(out_dir, f"optim_gen_epoch-{e}.th"))
        th.save(disc.cpu().state_dict(), join(out_dir, f"{disc}_epoch-{e}.th"))
        th.save(optim_disc.state_dict(), join(out_dir, f"optim_disc_epoch-{e}.th"))

        gen = gen.to(th.device("cuda"))
        disc = disc.to(th.device("cuda"))

    ticks = range(nb_epoch)
    plt.plot(ticks, losses_disc, color="red", label="Disc. loss")
    plt.plot(ticks, losses_gen, color="blue", label="Gen. loss")
    plt.plot(ticks, acc_fake, color="green", label="Fake precision")
    plt.plot(ticks, acc_real, color="turquoise", label="Real precision")
    plt.title(f"Train GAN - Decoder {archi}")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(join(out_dir, f"train_gan_archi{archi}_{nb_epoch}epoch.png"))


if __name__ == '__main__':
    main()

import argparse
from os.path import exists, isfile

import torch as th
import numpy as np

from scipy.io import wavfile

import networks
import read_audio
from hidden_gen import normal_dist, word_embedding


def main() -> None:
    coder_maker = networks.CoderMaker()

    parser = argparse.ArgumentParser("Generate Audio main")

    parser.add_argument("--archi", type=str, choices=coder_maker.models, dest="archi", required=True)
    parser.add_argument("--nfft", type=int, dest="n_fft", required=True)
    parser.add_argument("--sample-rate", type=int, default=44100, dest="sample_rate")
    parser.add_argument("-d", "--decoder-path", type=str, required=True, dest="decoder_path")
    parser.add_argument("-n", "--nb-sec", type=int, default=1, dest="nb_sec")
    parser.add_argument("-o", "--out-wav", type=str, default="generated_audio.wav", dest="out_wav")

    args = parser.parse_args()

    decoder_path = args.decoder_path
    nb_sec = args.nb_sec
    out_wav = args.out_wav
    archi = args.archi
    n_fft = args.n_fft
    sample_rate = args.sample_rate

    if not exists(decoder_path):
        print(f"{decoder_path} doesn't exist !")
        exit()
    if not isfile(decoder_path):
        print(f"{decoder_path} isn't a file !")
        exit()

    with th.no_grad():
        print(f"Loading model \"{decoder_path}\"")

        dec = coder_maker["decoder", archi, n_fft]

        hidden_length = sample_rate // n_fft // dec.division_factor()
        hidden_channel = dec.hidden_channels()
        dec.load_state_dict(th.load(decoder_path))

        print("Random hidden representation generation")
        # random_data = hidden_repr.rec_multivariate_gen_2(nb_sec, hidden_channel)
        # random_data = th.tensor(random_data).to(th.float).unsqueeze(0)

        #random_data = normal_dist.rec_normal_gen_mask(hidden_length, nb_sec, hidden_channel, eta=0.9)

        cov_mat = th.rand(hidden_channel, hidden_channel) * 2. - 1.
        cov_mat = th.mm(cov_mat, cov_mat.transpose(1, 0))
        means = th.rand(hidden_channel) * 0.6 - 0.3

        #random_data = normal_dist.rec_multivariate_gen(hidden_length, nb_sec, hidden_channel, means, cov_mat, eta=0.8)

        #random_data = normal_dist.rec_multivariate_different_gen(hidden_length, nb_sec, hidden_channel, eta=0.8, beta=0.5)
        #random_data = normal_dist.gen_init_normal_uni_add(hidden_length, nb_sec, hidden_channel)

        random_data = th.randn(1, hidden_channel, hidden_length * nb_sec)
        #s = "Dans le simple concept d'une chose on ne saurait trouver absolument aucun caractère de son existence. En effet, quoique ce concept soit tellement complet que rien n'y manque pour concevoir une chose avec toutes ses déterminations intérieures, l'existence n'a cependant rien à faire avec toutes ses déterminations et toute la question est de savoir si une chose de ce genre nous est donnée de telle sorte que sa perception puisse toujours précéder le concept. En effet, que le concept précède la perception, cela signifie simplement que la chose est possible, tandis que la perception qui fournit au concept la matière est le seul caractère de la réalité. Mais on peut aussi, antérieurement à la perception de la chose, et, par conséquent, relativement à priori, en connaître l'existence, pourvu qu'elle s'accorde avec quelques perceptions suivant les principes de leur liaison empirique (les analogies). Car alors l'existence de la chose est liée à nos perceptions dans une expérience possible et il nous est possible, en suivant le fil conducteur de ces analogies, d'arriver, en partant de notre perception réelle, à la chose, dans la série des perceptions possibles. C'est ainsi que nous connaissons, par la perception de la limaille de fer attirée, l'existence d'une matière magnétique qui pénètre tous les corps, quoiqu'une perception immédiate de cette matière nous soit impossible d'après la constitution de nos organes. En effet, d'après les lois de la sensibilité et d'après le contexte de nos perceptions, nous arriverions à avoir dans une expérience l'intuition empirique de cette matière, si nos sens étaient plus subtils, mais la grossièreté de nos organes ne touche en rien à la forme de l'expérience possible en général. Partout donc où s'étendent la perception et ce qui en dépend, en vertu des lois empiriques, là s'étend aussi notre connaissance de l'existence des choses. Si nous ne partions pas de l'expérience ou si nous ne procédions pas suivant les lois de l'enchaînement empirique des phénomènes, nous nous flatterions vainement de vouloir deviner et rechercher l'existence de quelque chose"
        #random_data = word_embedding.gen_embedding("/home/samuel/Documents/fastText/cc.fr.300.bin",
        #                                           s, hidden_length, hidden_channel)

        print(random_data.mean(dim=1))
        print(random_data.std(dim=1))
        print("Passing random data to decoder")
        out = dec(random_data)

        print("Processing result")
        re_out = out[:, :n_fft, :].numpy()
        img_out = out[:, n_fft:, :].numpy()

        cplx_out = (re_out + 1j * img_out).astype(np.complex128)

        raw_audio = read_audio.ifft_samples(cplx_out, n_fft).reshape(-1)
        raw_audio = raw_audio / np.max(np.abs(raw_audio))

        print(f"Writing WAV audio file in \"{out_wav}\"")
        wavfile.write(out_wav, sample_rate, raw_audio)


if __name__ == "__main__":
    main()

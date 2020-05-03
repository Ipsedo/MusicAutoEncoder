import argparse

import numpy as np
import torch as th

from networks import CoderMaker
import read_audio
from hidden_gen import word_embedding

import matplotlib.pyplot as plt


def main() -> None:
    coder_maker = CoderMaker()

    parser = argparse.ArgumentParser("Visualize hidden")

    parser.add_argument("--archi", type=str, choices=coder_maker.models, dest="archi", required=True)
    parser.add_argument("--nfft", type=int, dest="nfft", required=True)
    parser.add_argument("-e", "--encoder-path", type=str, required=True, dest="encoder_path")
    parser.add_argument("-d", "--decoder-path", type=str, required=True, dest="decoder_path")
    parser.add_argument("-i", "--input-wav", type=str, required=True, dest="input_wav")
    parser.add_argument("--from-to", nargs=2, type=int, required=True, dest="from_to")

    args = parser.parse_args()

    assert args.from_to[0] < args.from_to[1], \
        f"Negative time delta : from = {args.from_to[0]}, to = {args.from_to[1]}."

    enc = coder_maker["encoder", args.archi, args.nfft]
    dec = coder_maker["decoder", args.archi, args.nfft]

    enc.load_state_dict(th.load(args.encoder_path))
    dec.load_state_dict(th.load(args.decoder_path))

    sample_rate, raw_data = read_audio.open_wav(args.input_wav, 1)

    hidden_length = sample_rate // args.nfft // enc.division_factor()

    raw_data = raw_data[args.from_to[0]:args.from_to[1]]
    print(raw_data.shape)

    length = (args.from_to[1] - args.from_to[0]) * 44100

    plt.plot(range(length), raw_data.reshape(-1))
    plt.title(f"Raw audio 44100Hz - {args.from_to[1] - args.from_to[0]} seconds")
    plt.xlabel("time")
    plt.ylabel("signal value")
    plt.tight_layout()
    plt.savefig(f"raw_audio.png")

    fft_data = read_audio.fft_raw_audio(raw_data, args.nfft)
    data_real = np.real(fft_data)
    data_img = np.imag(fft_data)

    data = th.tensor(np.concatenate([data_real, data_img], axis=2)).to(th.float).permute(0, 2, 1)

    data_copy = data.clone()
    data_copy = data_copy / data_copy.std(dim=1).unsqueeze(1) - data_copy.mean(dim=1).unsqueeze(1)

    plt.matshow(data_copy.permute(0, 2, 1)[0].numpy().T)
    plt.title(f"FFT values (window = {args.nfft}), 1 second", pad=20.)
    plt.xlabel("time")
    plt.ylabel("FFT values - real & imag part")
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1 - 20, y2))
    plt.savefig("fft_values.png")

    data = data.permute(1, 0, 2).flatten(1, 2).unsqueeze(0)
    out = enc(data)
    print(out.size())

    plt.matshow(out.detach().permute(0, 2, 1).view(-1, hidden_length, enc.hidden_channels())[0].numpy().T)
    plt.title(f"Hidden space - archi {args.archi}, 1 second")
    plt.xlabel("time")
    plt.ylabel("Latent vector values")
    plt.savefig(f"hidden_space_archi{args.archi}.png")

    out_dec = dec(out)
    print(out_dec.size())
    out_dec = out_dec.view(args.nfft * 2, -1, sample_rate // args.nfft).permute(1, 0, 2).detach()

    out_dec_copy = out_dec.clone()
    out_dec_copy = out_dec_copy / out_dec_copy.std(dim=1).unsqueeze(1) - out_dec_copy.mean(dim=1).unsqueeze(1)

    plt.matshow(out_dec_copy.permute(0, 2, 1)[0].numpy().T)
    plt.title(f"Decoder output - archi {args.archi}, 1 second", pad=20.)
    plt.xlabel("time")
    plt.ylabel("recovered FFT values")
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1 - 20, y2))
    plt.savefig(f"output_dec_archi{args.archi}.png")

    print(out_dec.size())

    re_out = out_dec[:, :args.nfft, :].numpy()
    img_out = out_dec[:, args.nfft:, :].numpy()
    cplx_out = (re_out + 1j * img_out).astype(np.complex128)
    print( cplx_out.shape)

    plt.figure()
    raw_audio = read_audio.ifft_samples(cplx_out, args.nfft).astype(np.float32).reshape(-1)
    print(raw_audio.shape)
    plt.plot(range(raw_audio.shape[0]), raw_audio)
    plt.title(f"Recovered raw audio 44100Hz - {args.from_to[1] - args.from_to[0]} seconds")
    plt.xlabel("time")
    plt.ylabel("signal value")
    plt.savefig(f"recovered_raw_audio.png")

    """
    s = "Dans le simple concept d'une chose on ne saurait trouver absolument aucun caractère de son existence. En effet, quoique ce concept soit tellement complet que rien n'y manque pour concevoir une chose avec toutes ses déterminations intérieures, l'existence n'a cependant rien à faire avec toutes ses déterminations et toute la question est de savoir si une chose de ce genre nous est donnée de telle sorte que sa perception puisse toujours précéder le concept. En effet, que le concept précède la perception, cela signifie simplement que la chose est possible, tandis que la perception qui fournit au concept la matière est le seul caractère de la réalité. Mais on peut aussi, antérieurement à la perception de la chose, et, par conséquent, relativement à priori, en connaître l'existence, pourvu qu'elle s'accorde avec quelques perceptions suivant les principes de leur liaison empirique (les analogies). Car alors l'existence de la chose est liée à nos perceptions dans une expérience possible et il nous est possible, en suivant le fil conducteur de ces analogies, d'arriver, en partant de notre perception réelle, à la chose, dans la série des perceptions possibles. C'est ainsi que nous connaissons, par la perception de la limaille de fer attirée, l'existence d'une matière magnétique qui pénètre tous les corps, quoiqu'une perception immédiate de cette matière nous soit impossible d'après la constitution de nos organes. En effet, d'après les lois de la sensibilité et d'après le contexte de nos perceptions, nous arriverions à avoir dans une expérience l'intuition empirique de cette matière, si nos sens étaient plus subtils, mais la grossièreté de nos organes ne touche en rien à la forme de l'expérience possible en général. Partout donc où s'étendent la perception et ce qui en dépend, en vertu des lois empiriques, là s'étend aussi notre connaissance de l'existence des choses. Si nous ne partions pas de l'expérience ou si nous ne procédions pas suivant les lois de l'enchaînement empirique des phénomènes, nous nous flatterions vainement de vouloir deviner et rechercher l'existence de quelque chose"
    random_data = word_embedding.gen_embedding("/home/samuel/Documents/fastText/cc.fr.300.bin",
                                               s, hidden_length, enc.hidden_channels())

    random_data = random_data.squeeze(0).numpy()

    plt.matshow(random_data)
    plt.show()"""


if __name__ == '__main__':
    main()

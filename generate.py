import argparse
from os.path import exists, isfile

import torch as th
import numpy as np

from scipy.io import wavfile

import auto_encoder
import read_audio
import hidden_repr


def main() -> None:
    parser = argparse.ArgumentParser("Generate Audio main")

    parser.add_argument("--archi", type=str, choices=["small", "1", "2", "3"], dest="archi", required=True)
    parser.add_argument("--n-fft", type=int, dest="n_fft", required=True)
    parser.add_argument("--sample-rate", type=int, default=44100, dest="sample_rate")
    parser.add_argument("-m", "--model-path", type=str, required=True, dest="model_path")
    parser.add_argument("-n", "--nb-sec", type=int, default=1, dest="nb_sec")
    parser.add_argument("-o", "--out-wav", type=str, default="generated_audio.wav", dest="out_wav")

    args = parser.parse_args()

    model_path = args.model_path
    nb_sec = args.nb_sec
    out_wav = args.out_wav
    archi = args.archi
    n_fft = args.n_fft
    sample_rate = args.sample_rate

    if not exists(model_path):
        print(f"{model_path} doesn't exist !")
        exit()
    if not isfile(model_path):
        print(f"{model_path} isn't a file !")
        exit()

    with th.no_grad():
        print(f"Loading model \"{model_path}\"")

        if archi == "1":
            dec = auto_encoder.Decoder1(n_fft * 2)
            hidden_length = sample_rate // n_fft // 3 // 4 // 5
            hidden_channel = n_fft * 2 * 2
        elif archi == "2":
            dec = auto_encoder.Decoder2(n_fft * 2)
            hidden_length = sample_rate // n_fft // 2 // 2 // 3 // 5
            hidden_channel = n_fft * 2 * 2
        elif archi == "3":
            dec = auto_encoder.Decoder3(n_fft * 2)
            hidden_length = sample_rate // n_fft // 2 // 2 // 3
            hidden_channel = n_fft * 2 * 2
        elif archi == "small":
            dec = auto_encoder.DecoderSmall(n_fft * 2)
            hidden_length = sample_rate // n_fft // 2 // 3
            hidden_channel = n_fft * 2 + 128
        else:
            print(f"Unrecognized NN architecture ({archi}).")
            print(f"Will load small CNN")
            dec = auto_encoder.DecoderSmall(n_fft * 2)
            hidden_length = sample_rate // n_fft // 2 // 3
            hidden_channel = n_fft * 2 + 128

        dec.load_state_dict(th.load(model_path))

        print("Random hidden representation generation")
        # random_data = hidden_repr.rec_multivariate_gen_2(nb_sec, hidden_channel)
        # random_data = th.tensor(random_data).to(th.float).unsqueeze(0)

        # random_data = hidden_repr.rec_normal_gen(values.HIDDEN_LENGTH, nb_sec, hidden_channel, eta=1e-2)

        cov_mat = th.rand(hidden_channel, hidden_channel) * 8. - 4.
        cov_mat = th.mm(cov_mat, cov_mat.transpose(1, 0))
        means = th.rand(hidden_channel) * 0.6 - 0.3

        #random_data = hidden_repr.rec_multivariate_gen(hidden_length, nb_sec, hidden_channel, means, cov_mat, eta=1 - 1e-3)

        random_data = hidden_repr.rec_multivariate_different_gen(hidden_length, nb_sec, hidden_channel, eta=0.7, beta=0.7)

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

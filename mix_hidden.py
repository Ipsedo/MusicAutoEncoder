import argparse

import torch as th
import numpy as np

from tqdm import tqdm

from scipy.io import wavfile

import networks
import read_audio


def main() -> None:
    coder_maker = networks.CoderMaker()

    parser = argparse.ArgumentParser("Generate Audio main")

    parser.add_argument("--archi", type=str, choices=coder_maker.models, dest="archi", required=True)
    parser.add_argument("--nfft", type=int, dest="n_fft", required=True)
    parser.add_argument("-e", "--encoder-path", type=str, required=True, dest="encoder_path")
    parser.add_argument("-d", "--decoder-path", type=str, required=True, dest="decoder_path")
    parser.add_argument("-m", "--mode", type=str, choices=["mean", "random", "alternated"])
    parser.add_argument("--input-wavs", nargs="+", type=str, required=True, dest="input_wavs")
    parser.add_argument("--sample-index", type=int, required=True, dest="sample_index")
    parser.add_argument("--nb-sample", type=int, required=True, dest="nb_sample")

    args = parser.parse_args()

    encoder_path = args.encoder_path
    decoder_path = args.decoder_path
    input_wavs = args.input_wavs
    archi = args.archi
    n_fft = args.n_fft
    sample_index = args.sample_index
    nb_sample = args.nb_sample

    datas = []
    for w in tqdm(input_wavs):
        audio_data = read_audio.open_wav(w, 1000)[1]
        fft_data = read_audio.fft_raw_audio(audio_data, n_fft)

        data_real = np.real(fft_data)
        data_img = np.imag(fft_data)

        data = th.tensor(np.concatenate([data_real, data_img], axis=2)).to(th.float).permute(0, 2, 1)

        datas.append(data)

    with th.no_grad():
        enc = coder_maker["encoder", archi, n_fft]
        dec = coder_maker["decoder", archi, n_fft]

        enc.load_state_dict(th.load(encoder_path))
        dec.load_state_dict(th.load(decoder_path))

        hidden = th.stack([enc(d[sample_index:sample_index + nb_sample, :, :]) for d in datas], dim=0)

        print(hidden.size())

        if args.mode == "mean":
            hidden = hidden.mean(dim=0)

        elif args.mode == "random":
            rand_idx = th.randint(0, hidden.size(0), (hidden.size(1) * hidden.size(3),))
            hidden = hidden.permute(1, 3, 0, 2).flatten(0, 1)
            res = th.zeros(hidden.size(0), hidden.size(2))

            for i in tqdm(range(res.size(0))):
                res[i] = hidden[i, rand_idx[i], :]
            hidden = res.permute(1, 0).unsqueeze(0)

        elif args.mode == "alternated":
            size = hidden.size(1) * hidden.size(3)

            idx = th.cat([th.arange(len(input_wavs)) for _ in range(size // len(input_wavs))])
            idx = th.cat([idx, th.zeros(size - idx.size(0)).to(th.long)])

            hidden = hidden.permute(1, 3, 0, 2).flatten(0, 1)
            res = th.zeros(hidden.size(0), hidden.size(2))

            for i in tqdm(range(res.size(0))):
                res[i] = hidden[i, idx[i], :]

            hidden = res.permute(1, 0).unsqueeze(0)

        else:
            print("Unrecognized mode, will start mean")
            hidden = hidden.mean(dim=0)

        out_dec = dec(hidden)

        re_out = out_dec[:, :n_fft, :].numpy()
        img_out = out_dec[:, n_fft:, :].numpy()
        cplx_out = (re_out + 1j * img_out).astype(np.complex128)

        print(cplx_out.shape)

        raw_audio = read_audio.ifft_samples(cplx_out, n_fft)
        raw_audio = raw_audio  # / np.linalg.norm(raw_audio)
        print(raw_audio.shape)

        wavfile.write("mix.wav", 44100, raw_audio.reshape(-1))


if __name__ == "__main__":
    main()

import argparse

import torch as th
import numpy as np

from typing import List

from tqdm import tqdm

from scipy.io import wavfile

import auto_encoder
import read_audio


def main() -> None:
    parser = argparse.ArgumentParser("Generate Audio main")

    parser.add_argument("--archi", type=str, choices=["small", "1", "2", "3"], dest="archi", required=True)
    parser.add_argument("--n-fft", type=int, dest="n_fft", required=True)
    parser.add_argument("-e", "--encoder-path", type=str, required=True, dest="encoder_path")
    parser.add_argument("-d", "--decoder-path", type=str, required=True, dest="decoder_path")
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
        if archi == "1":
            enc = auto_encoder.Encoder1(n_fft * 2)
            dec = auto_encoder.Decoder1(n_fft * 2)
        elif archi == "2":
            enc = auto_encoder.Encoder2(n_fft * 2)
            dec = auto_encoder.Decoder2(n_fft * 2)
        elif archi == "3":
            enc = auto_encoder.Encoder3(n_fft * 2)
            dec = auto_encoder.Decoder3(n_fft * 2)
        elif archi == "small":
            enc = auto_encoder.EncoderSmall(n_fft * 2)
            dec = auto_encoder.DecoderSmall(n_fft * 2)
        else:
            print(f"Unrecognized NN architecture ({archi}).")
            print(f"Will load small CNN")
            enc = auto_encoder.EncoderSmall(n_fft * 2)
            dec = auto_encoder.DecoderSmall(n_fft * 2)

        enc.load_state_dict(th.load(encoder_path))
        dec.load_state_dict(th.load(decoder_path))

        hidden_mean = th.stack([enc(d[sample_index:sample_index + nb_sample, :, :]) for d in datas], dim=0).mean(dim=0)

        out_dec = dec(hidden_mean)

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

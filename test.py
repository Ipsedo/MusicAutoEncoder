import argparse

import torch as th
import numpy as np

from scipy.io import wavfile

from auto_encoder import Encoder, Decoder
import read_audio
import values


def main() -> None:
    parser = argparse.ArgumentParser("Generate Audio main")

    parser.add_argument("-e", "--encoder-path", type=str, required=True, dest="encoder_path")
    parser.add_argument("-d", "--decoder-path", type=str, required=True, dest="decoder_path")
    parser.add_argument("-i", "--input-wav", type=str, required=True, dest="input_wav")

    args = parser.parse_args()

    encoder_path = args.encoder_path
    decoder_path = args.decoder_path
    input_wav = args.input_wav

    audio_data = read_audio.open_wav(input_wav, 1000)[1]
    print(audio_data.shape)
    fft_data = read_audio.fft_raw_audio(audio_data, values.N_FFT)

    data_real = np.real(fft_data)
    data_img = np.imag(fft_data)

    data = th.tensor(np.concatenate([data_real, data_img], axis=2)).to(th.float).permute(0, 2, 1)

    with th.no_grad():
        enc = Encoder(values.N_FFT * 2)
        dec = Decoder(values.N_FFT * 2 + 128)

        enc.load_state_dict(th.load(encoder_path))
        dec.load_state_dict(th.load(decoder_path))

        out_enc = enc(data)
        print(out_enc.size())
        print(out_enc.max(), out_enc.min())
        print("#################################################################")
        print(out_enc.std(dim=1).min(), out_enc.std(dim=1).max())
        print("#################################################################")
        print(out_enc.mean(dim=1).min(), out_enc.mean(dim=1).max())
        out_dec = dec(out_enc)

        re_out = out_dec[:, :values.N_FFT, :].numpy()
        img_out = out_dec[:, values.N_FFT:, :].numpy()
        cplx_out = (re_out + 1j * img_out).astype(np.complex128)

        print(cplx_out.shape)

        raw_audio = read_audio.ifft_samples(cplx_out, values.N_FFT)
        raw_audio = raw_audio# / np.linalg.norm(raw_audio)
        print(raw_audio.shape)

        wavfile.write("inference.wav", 44100, raw_audio.reshape(-1))


if __name__ == "__main__":
    main()
import argparse
from os import walk
from os.path import splitext, isfile, isdir, basename, join
import subprocess

import numpy as np
import scipy
from scipy.signal import spectrogram
from scipy.io import wavfile

from math import ceil, floor

from typing import Tuple

from tqdm import tqdm


def open_wav(wav_path: str, split_length: int) -> Tuple[int, np.ndarray]:
    assert split_length > 1, f"Split length must be > 1 (actual == {split_length})."

    sampling_rate, data = wavfile.read(wav_path)

    split_size = sampling_rate * split_length // 1000
    nb_split = floor(data.shape[0] / split_size)

    splitted_audio = np.asarray(np.split(data[:split_size * nb_split], nb_split))

    int_size = splitted_audio.itemsize * 8.

    splitted_audio = splitted_audio.astype(np.float16)
    splitted_audio[:, :, 0] = splitted_audio[:, :, 0] / (2. ** int_size) * 2.
    splitted_audio[:, :, 1] = splitted_audio[:, :, 1] / (2. ** int_size) * 2.

    return sampling_rate, splitted_audio.mean(axis=2)


def spectro_raw_audio(raw_audio_split: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    assert len(raw_audio_split.shape) == 2, \
        f"Wrong audio shape len (actual == {len(raw_audio_split.shape)}, needed == 2)."

    assert raw_audio_split.dtype == np.float16, \
        f"Wrong ndarray dtype (actual == {raw_audio_split.dtype}, neede == {np.float16})."

    max_value = raw_audio_split.max()
    min_value = raw_audio_split.min()
    assert max_value <= 1.0 and min_value >= -1., \
        f"Raw audio values must be normlized between [-1., 0.] (actual == [{min_value}, {max_value}])."

    padded_raw_audio_split = np.pad(raw_audio_split, ((0, 0), (0, noverlap)), "constant", constant_values=0)
    return np.apply_along_axis(lambda split: spectrogram(split, nperseg=nperseg, noverlap=noverlap)[-1], 1,
                               padded_raw_audio_split)


def fft_raw_audio(raw_audio_split: np.ndarray, nfft: int) -> np.ndarray:
    assert len(raw_audio_split.shape) == 2, \
        f"Wrong audio shape len (actual == {len(raw_audio_split.shape)}, needed == 2)."

    assert raw_audio_split.dtype == np.float16, \
        f"Wrong ndarray dtype (actual == {raw_audio_split.dtype}, neede == {np.float16})."

    max_value = raw_audio_split.max()
    min_value = raw_audio_split.min()
    assert max_value <= 1.0 and min_value >= -1., \
        f"Raw audio values must be normlized between [-1., 0.] (actual == [{min_value}, {max_value}])."

    pad = nfft - raw_audio_split.shape[-1] % nfft
    padded_data = np.pad(raw_audio_split,
                         ((0, 0), (0, pad)),
                         mode="constant", constant_values=0)
    splitted_data = np.stack(np.hsplit(padded_data, padded_data.shape[-1] / nfft), axis=-2)
    return np.apply_along_axis(lambda sub_split: scipy.fft(sub_split), 2, splitted_data)


def convert_mp3_to_wav(root_dir: str, out_dir: str, limit: int) -> None:
    cpt = 0
    for dirname, dirnames, filenames in walk(root_dir):
        for filename in tqdm(filenames):
            if splitext(filename)[-1] == ".mp3":
                subprocess.call(["ffmpeg", "-v", "0", "-i", join(dirname, filename),
                                 "-y", "-ar", "44100", join(out_dir, basename(filename) + ".wav")])
                cpt += 1
                if cpt >= limit >= 0:
                    return


def ifft_samples(fft_samples: np.ndarray, nfft: int) -> np.ndarray:
    assert len(fft_samples.shape) == 3, \
        f"Wrong spectrogram shape len (actual == {len(fft_samples.shape)}, needed == {3}"
    assert fft_samples.dtype == np.complex128, \
        f"Wrong ndarray dtype (actual == {fft_samples.dtype}, needed == {np.complex128})"

    return np.real(np.apply_along_axis(lambda fft_values: scipy.ifft(fft_values, n=nfft), 2, fft_samples)) \
        .reshape(fft_samples.shape[0], -1)


def main() -> None:
    parser = argparse.ArgumentParser("Read audio main")

    subparser = parser.add_subparsers(dest="mode")

    test_parser = subparser.add_parser("test")
    test_parser.add_argument("-i", "--input-audio", type=str, required=True, dest="input_audio")
    test_parser.add_argument("--split-length", type=int, default=10000, dest="split_length")

    process_parser = subparser.add_parser("process")
    process_parser.add_argument("--mp3-root", type=str, dest="audio_root", required=True)
    process_parser.add_argument("-o", "--out-dir", type=str, dest="out_dir", required=True)
    process_parser.add_argument("-l", "--limit", type=int, default=100)

    args = parser.parse_args()

    if args.mode == "test":
        input_audio = args.input_audio

        # in miliseconds
        split_length = args.split_length

        if splitext(input_audio)[-1].lower() != ".wav":
            parser.error("input-audio must be a WAV audio file !")
            exit()

        sampling_rate, data = wavfile.read(input_audio)

        print(f"{sampling_rate} {data.shape}")
        print(f"{int(data.shape[0] / sampling_rate / 60)}min{int(data.shape[0] / sampling_rate) % 60}s")

        split_size = sampling_rate * split_length / 1000.
        nb_split = ceil(data.shape[0] / split_size) - 1

        splitted_audio = np.asarray(np.split(data[:int(split_size * nb_split)], nb_split))

        if len(splitted_audio.shape) == 3 and splitted_audio.shape[2] == 2:
            print("Stereo WAV")
            print(f"actual shape = {splitted_audio.shape}, dtype = {splitted_audio.dtype}")
        else:
            print("Mono WAV - unsuported")
            exit()

        int_size = splitted_audio.itemsize * 8.

        splitted_audio = splitted_audio.astype(np.float16)

        splitted_audio[:, :, 0] = splitted_audio[:, :, 0] / (2. ** int_size) * 2.
        splitted_audio[:, :, 1] = splitted_audio[:, :, 1] / (2. ** int_size) * 2.

        splitted_audio = splitted_audio.mean(axis=2)

        print(splitted_audio.shape)
        print(splitted_audio.max())
        print(splitted_audio.min())

        nperseg = 128

        fft_audio = fft_raw_audio(splitted_audio, nperseg)

        print(f"FFT shape {fft_audio.shape}, max {fft_audio.max()}, min {fft_audio.min()}")

        new_raw_audio = ifft_samples(fft_audio, nperseg)

        wavfile.write("test.wav", sampling_rate, new_raw_audio.reshape(-1))

    elif args.mode == "process":
        convert_mp3_to_wav(args.audio_root, args.out_dir, args.limit)


if __name__ == "__main__":
    main()

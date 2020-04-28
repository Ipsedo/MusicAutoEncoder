import argparse
from os import walk, mkdir, listdir
from os.path import splitext, basename, join, exists, isfile
import subprocess

import torch as th
import numpy as np
import scipy
from scipy.signal import spectrogram
from scipy.io import wavfile

from random import shuffle
from math import ceil, floor

from typing import Tuple

from tqdm import tqdm


###############
# WAV stuff
###############

def compute_wav_size(wav_path: str, nb_sec) -> Tuple[int, int]:
    sampling_rate, data = wavfile.read(wav_path)
    split_size = sampling_rate * nb_sec
    nb_split = floor(data.shape[0] / split_size)
    return nb_split, split_size


def open_wav(wav_path: str, nb_sec: int) -> Tuple[int, np.ndarray]:
    assert nb_sec > 1, f"Split length must be > 1 (actual == {nb_sec})."

    sampling_rate, data = wavfile.read(wav_path)

    split_size = sampling_rate * nb_sec
    nb_split = floor(data.shape[0] / split_size)

    splitted_audio = np.asarray(np.split(data[:split_size * nb_split], nb_split))

    int_size = splitted_audio.itemsize * 8.

    splitted_audio = splitted_audio.astype(np.float16)
    splitted_audio[:, :, 0] = splitted_audio[:, :, 0] / (2. ** int_size) * 2.
    splitted_audio[:, :, 1] = splitted_audio[:, :, 1] / (2. ** int_size) * 2.

    return sampling_rate, splitted_audio.mean(axis=2)


###############
# Spectro stuff
###############

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


###############
# FFT stuff
###############

def compute_fft_size(raw_audio_split_size: Tuple[int, int], nfft: int) -> Tuple[int, int, int]:
    return raw_audio_split_size[0], nfft, raw_audio_split_size[1] // nfft


def fft_raw_audio(raw_audio_split: np.ndarray, nfft: int) -> np.ndarray:
    assert len(raw_audio_split.shape) == 2, \
        f"Wrong audio shape len (actual == {len(raw_audio_split.shape)}, needed == 2)."

    assert raw_audio_split.dtype == np.float16, \
        f"Wrong ndarray dtype (actual == {raw_audio_split.dtype}, neede == {np.float16})."

    max_value = raw_audio_split.max()
    min_value = raw_audio_split.min()
    assert max_value <= 1.0 and min_value >= -1., \
        f"Raw audio values must be normlized between [-1., 1.] (actual == [{min_value}, {max_value}])."

    splitted_data = np.stack(np.hsplit(raw_audio_split, raw_audio_split.shape[-1] / nfft), axis=-2)
    return np.apply_along_axis(lambda sub_split: scipy.fft(sub_split), 2, splitted_data)


def ifft_samples(fft_samples: np.ndarray, nfft: int) -> np.ndarray:
    assert len(fft_samples.shape) == 3, \
        f"Wrong spectrogram shape len (actual : {len(fft_samples.shape)}, needed : {3})"
    assert fft_samples.shape[1] == nfft, f"Only same nfft length for the moment"
    assert fft_samples.dtype == np.complex128, \
        f"Wrong ndarray dtype (actual : {fft_samples.dtype}, needed : {np.complex128})"

    """return np.real(np.apply_along_axis(lambda fft_values: scipy.ifft(fft_values, n=nfft), 2, fft_samples)) \
        .reshape(fft_samples.shape[0], -1)"""
    fft_samples = fft_samples.transpose((0, 2, 1)).reshape(-1, nfft)
    return np.real(np.apply_along_axis(lambda fft_values: scipy.ifft(fft_values, n=nfft), 1, fft_samples))


###############
# Main
###############

def convert_mp3_to_wav(root_dir: str, out_dir: str, limit: int) -> None:
    cpt = 0
    if not exists(out_dir):
        mkdir(out_dir)
    for dirname, dirnames, filenames in tqdm(walk(root_dir)):
        for filename in filenames:
            if splitext(filename)[-1] == ".mp3":
                subprocess.call(["ffmpeg", "-v", "0", "-i", join(dirname, filename),
                                 "-y", "-ar", "44100", join(out_dir, basename(filename) + ".wav")])
                cpt += 1
                if cpt >= limit >= 0:
                    return


def convert_mp3_to_wav_2(root_dir: str, out_dir: str, limit_per_dir: int = 1) -> None:
    cpt = {}
    for dirname, dirnames, filenames in tqdm(walk(root_dir)):
        for filename in filenames:
            if splitext(filename)[-1] == ".mp3":
                if dirname not in cpt:
                    cpt[dirname] = 0
                if cpt[dirname] < limit_per_dir:
                    subprocess.call(["ffmpeg", "-v", "0", "-i", join(dirname, filename),
                                     "-y", "-ar", "44100", join(out_dir, basename(filename) + ".wav")])
                    cpt[dirname] += 1


def copy_mp3_2(root_dir: str, out_dir: str, limit_per_dir: int = 1) -> None:
    cpt = {}
    for dirname, dirnames, filenames in tqdm(walk(root_dir)):
        for filename in filenames:
            if splitext(filename)[-1] == ".mp3":
                if dirname not in cpt:
                    cpt[dirname] = 0
                if cpt[dirname] < limit_per_dir:
                    subprocess.call(["cp", join(dirname, filename), out_dir])
                    cpt[dirname] += 1


def __read_wavs_without_copy(wav_root: str, nb_wav: int, sample_rate: int, n_fft: int, sec: int) -> th.Tensor:
    wav_files = [join(wav_root, f) for f in listdir(wav_root) if splitext(f)[-1] == ".wav"]

    shuffle(wav_files)

    if len(wav_files) > nb_wav:
        wav_files = wav_files[:nb_wav]

    n_channel = n_fft * 2
    fft_split_size = sample_rate * sec // n_fft
    n_sample = 0

    for w in tqdm(wav_files):
        n_sample += compute_wav_size(w, sec)[0]

    data = th.zeros(n_sample, n_channel, fft_split_size, dtype=th.float)

    curr_split = 0

    for w in tqdm(wav_files):
        _, raw_audio = open_wav(w, sec)
        fft_audio = fft_raw_audio(raw_audio, n_fft).transpose((0, 2, 1))

        data[curr_split:curr_split + fft_audio.shape[0], :n_fft, :] = \
            th.tensor(np.real(fft_audio), dtype=th.float)
        data[curr_split:curr_split + fft_audio.shape[0], n_fft:, :] = \
            th.tensor(np.imag(fft_audio), dtype=th.float)

        curr_split += fft_audio.shape[0]

    return data


def main() -> None:
    parser = argparse.ArgumentParser("Read audio main")

    subparser = parser.add_subparsers(dest="mode")
    subparser.required = True

    test_parser = subparser.add_parser("test")
    test_parser.add_argument("-i", "--input-audio", type=str, required=True, dest="input_audio")
    test_parser.add_argument("--split-length", type=int, default=10000, dest="split_length")

    process_parser = subparser.add_parser("process")
    process_parser.add_argument("--mp3-root", type=str, dest="audio_root", required=True)
    process_parser.add_argument("-o", "--out-dir", type=str, dest="out_dir", required=True)
    process_parser.add_argument("-l", "--limit", type=int, default=100)

    save_parser = subparser.add_parser("save")
    save_parser.add_argument("--wav-root", type=str, dest="wav_root", required=True)
    save_parser.add_argument("-n", "--nb-wav", type=int, default=400, dest="nb_wav")
    save_parser.add_argument("-o", "--out-tensor-file", type=str, dest="out_tensor_file", required=True)
    save_parser.add_argument("--nfft", type=int, dest="n_fft", required=True)
    save_parser.add_argument("--sample-rate", type=int, default=44100, dest="sample_rate")
    save_parser.add_argument("-s", "--seconds", type=int, required=True, dest="seconds")

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

        new_raw_audio = ifft_samples(fft_audio.transpose((0, 2, 1)), nperseg)

        wavfile.write("test.wav", sampling_rate, new_raw_audio.reshape(-1))

    elif args.mode == "process":
        if not exists(args.out_dir):
            mkdir(args.out_dir)
        if exists(args.out_dir) and isfile(args.out_dir):
            print(f"{args.out_dir} already exists and is a file.")
            exit()

        convert_mp3_to_wav(args.audio_root, args.out_dir, args.limit)
    elif args.mode == "save":
        data = __read_wavs_without_copy(args.wav_root, args.nb_wav, args.sample_rate, args.n_fft, args.seconds)
        th.save(data, args.out_tensor_file)


if __name__ == "__main__":
    main()

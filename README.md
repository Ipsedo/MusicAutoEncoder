# MusicAutoEncoder
_author_ : Samuel Berrien

# Requirements

Python 3.6 pip packages :
```
torch
numpy
scipy
tqdm
```

For training only : a CUDA capable GPU with at least 2GB

# Installation

First install all the dependencies listed before.

Then clone the repo :
```bash
$ git clone https://github.com/Ipsedo/MusicAutoEncoder.git
```

# Usage

The argparse module gives help information with :
```bash
$ python script.py -h
```

## Train

Create your set of wav files for training, in this example mp3 files are assumed to be in `/path/to/mp3` and we will convert `100` files in wav to `/path/to/wav` out directory :
```bash
$ python read_audio.py --mp3-root /path/to/mp3 --out-dir /path/to/wav -l 100
```

Now you are able to generate the torch.Tensor file :
```bash
$ python read_audio.py --wav-root /path/to/wav --nb-wav 30 --out-tensor out_tensor.pt --nfft 49 --sample-rate 44100 --seconds 1
```
It creates a tensor in the file `out_tensor.pt` from `30` wav files at `44100`Hz  with `49` FFT values contained in `/path/to/wav` directory.

Finally start the training :
```bash
$ python train.py -h # TODO
```

## Inference

TODO

## Generation

TODO

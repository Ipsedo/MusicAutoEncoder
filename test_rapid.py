import torch as th
import torch. nn as nn
import auto_encoder

n_channel = 300

n_channel_h = 428
x = th.rand(2, n_channel, 294)
print(x.size())

ec1=nn.Conv1d(n_channel, n_channel + 32, kernel_size=3, padding=1)
print(ec1(x).size())
ec2=nn.Conv1d(n_channel + 32, n_channel + 64, kernel_size=5, stride=2, padding=2)
print(ec2(ec1(x)).size())
ec3=nn.Conv1d(n_channel + 64, n_channel + 128, kernel_size=7, stride=3, padding=3)
print(ec3(ec2(ec1(x))).size())

h = th.rand(2, n_channel_h, 3)
dc1 = nn.ConvTranspose1d(n_channel_h, n_channel_h - 128, kernel_size=7, stride=3, padding=2)
print(dc1(h).size())
dc2=nn.ConvTranspose1d(n_channel_h - 128, n_channel_h - (128 + 64), kernel_size=5, stride=2, output_padding=1, padding=2)
print(dc2(dc1(h)).size())
dc3=nn.ConvTranspose1d(n_channel_h - (128 + 64), 256, kernel_size=3, padding=1)
print(dc3(dc2(dc1(h))).size())

enc = auto_encoder.Encoder(n_channel)
out_enc = enc(x)
print(out_enc.size())

dec=auto_encoder.Decoder(n_channel_h)
print(dec(out_enc).size())
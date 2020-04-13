import torch as th
import torch. nn as nn
import auto_encoder

n_channel = 147 * 2
x = th.rand(2, n_channel, 44100 // 147)

n_layer = 4

ec1 = nn.Conv1d(n_channel, n_channel + int(n_channel / n_layer),
                kernel_size=3, padding=1)
ec2 = nn.Conv1d(n_channel + int(n_channel / n_layer), n_channel + int(2 * n_channel / n_layer),
                kernel_size=5, stride=2, padding=2)
ec3 = nn.Conv1d(n_channel + int(2 * n_channel / n_layer), n_channel + int(3 * n_channel / n_layer),
                kernel_size=5, stride=2, padding=2)
ec4 = nn.Conv1d(n_channel + int(3 * n_channel / n_layer), n_channel + int(4 * n_channel / n_layer),
                kernel_size=7, stride=3, padding=3)

print(x.size(), ec1(x).size(), ec2(ec1(x)).size(), ec3(ec2(ec1(x))).size(), ec4(ec3(ec2(ec1(x)))).size())

dc1=nn.ConvTranspose1d(n_channel + int(4 * n_channel / n_layer),
                       n_channel + int(3 * n_channel / n_layer),
                       kernel_size=7, stride=3, padding=2)
dc2=nn.ConvTranspose1d(n_channel + int(3 * n_channel / n_layer),
                       n_channel + int(2 * n_channel / n_layer),
                       kernel_size=5, stride=2, padding=2, output_padding=1)
dc3=nn.ConvTranspose1d(n_channel + int(2 * n_channel / n_layer),
                       n_channel + int(n_channel / n_layer),
                       kernel_size=5, stride=2, padding=2, output_padding=1)
dc4=nn.ConvTranspose1d(n_channel + int(n_channel / n_layer),
                       n_channel,
                       kernel_size=3, padding=1)
out = th.rand(2, 588, 25)
print(dc1(out).size(), dc2(dc1(out)).size(), dc3(dc2(dc1(out))).size(), dc4(dc3(dc2(dc1(out)))).size())

# enc = auto_encoder.Encoder2(n_channel)
# dec = auto_encoder.Decoder2(n_channel)
# out = enc(x)
# x_p = dec(out)
# print(x.size(), out.size(), x_p.size())
# exit()
# n_layer = 5
#
# ec1=nn.Conv1d(n_channel, n_channel + int(n_channel / n_layer),
#               kernel_size=3, padding=1)
# ec2=nn.Conv1d(n_channel + int(n_channel / n_layer),
#                       n_channel + int(2 * n_channel / n_layer),
#                       kernel_size=5, stride=2, padding=2)
# ec3=nn.Conv1d(n_channel + int(2 * n_channel / n_layer),
#                       n_channel + int(3 * n_channel / n_layer),
#                       kernel_size=5, stride=2, padding=2)
# ec4=nn.Conv1d(n_channel + int(3 * n_channel / n_layer),
#                       n_channel + int(4 * n_channel / n_layer),
#                       kernel_size=7, stride=3, padding=2)
# ec5=nn.Conv1d(n_channel + int(4 * n_channel / n_layer),
#                       n_channel + int(5 * n_channel / n_layer),
#                       kernel_size=11, stride=5, padding=5)
#
# print(x.size(), ec1(x).size(), ec2(ec1(x)).size(), ec3(ec2(ec1(x))).size(), ec4(ec3(ec2(ec1(x)))).size(), ec5(ec4(ec3(ec2(ec1(x))))).size())
#
# dc1=nn.ConvTranspose1d(n_channel * 2, n_channel + int(4 * n_channel / n_layer),
#                        kernel_size=11, stride=5, padding=3)
# dc2=nn.ConvTranspose1d(n_channel + int(4 * n_channel / n_layer),
#                                n_channel + int(3 * n_channel / n_layer),
#                                kernel_size=7, stride=3, padding=2)
# dc3=nn.ConvTranspose1d(n_channel + int(3 * n_channel / n_layer),
#                                n_channel + int(2 * n_channel / n_layer),
#                                kernel_size=5, stride=2, padding=2, output_padding=1)
# dc4=nn.ConvTranspose1d(n_channel + int(2 * n_channel / n_layer),
#                                n_channel + int(n_channel / n_layer),
#                                kernel_size=5, stride=2, padding=2, output_padding=1)
# dc5=nn.ConvTranspose1d(n_channel + int(n_channel / n_layer),
#                                147*2,
#                                kernel_size=3, padding=1)
# print(dc1(out).size(), dc2(dc1(out)).size(), dc3(dc2(dc1(out))).size(), dc4(dc3(dc2(dc1(out)))).size(), dc5(dc4(dc3(dc2(dc1(out))))).size())
#
# print(x.size(), enc(x).size(), dec(enc(x)).size())

# n_channel_h = 428
# x = th.rand(2, n_channel, 294)
# print(x.size())
#
# ec1=nn.Conv1d(n_channel, n_channel + 32, kernel_size=3, padding=1)
# print(ec1(x).size())
# ec2=nn.Conv1d(n_channel + 32, n_channel + 64, kernel_size=5, stride=2, padding=2)
# print(ec2(ec1(x)).size())
# ec3=nn.Conv1d(n_channel + 64, n_channel + 128, kernel_size=7, stride=3, padding=3)
# print(ec3(ec2(ec1(x))).size())
#
# h = th.rand(2, n_channel_h, 3)
# dc1 = nn.ConvTranspose1d(n_channel_h, n_channel_h - 128, kernel_size=7, stride=3, padding=2)
# print(dc1(h).size())
# dc2=nn.ConvTranspose1d(n_channel_h - 128, n_channel_h - (128 + 64), kernel_size=5, stride=2, output_padding=1, padding=2)
# print(dc2(dc1(h)).size())
# dc3=nn.ConvTranspose1d(n_channel_h - (128 + 64), 256, kernel_size=3, padding=1)
# print(dc3(dc2(dc1(h))).size())
#
# enc = auto_encoder.Encoder(n_channel)
# out_enc = enc(x)
# print(out_enc.size())
#
# dec=auto_encoder.Decoder(n_channel_h)
# print(dec(out_enc).size())
import torch
from torch.nn.init import kaiming_uniform_
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

import numpy as np
from modules import *



def weight_init(m):
    if isinstance(m, nn.Conv2d):
        kaiming_uniform_(m.weight)


class TextEnc(nn.Module):
    """Encodes asdf text input sequence of length N into two matrices K (Key) and V (Value) of shape [b, d, N]. """

    def __init__(self):
        super(TextEnc, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(Config.vocab), embedding_dim=Config.e,
                                      padding_idx=Config.vocab_padding_index)

        self.conv1 = Conv1d(in_channels=Config.e, out_channels=2 * Config.d, kernel_size=1, dilation=1, padding="same")
        self.conv2 = Conv1d(in_channels=2 * Config.d, out_channels=2 * Config.d, kernel_size=1, dilation=1,
                            padding="same")

        self.highway_blocks = nn.ModuleList()
        self.highway_blocks.extend([HighwayConv(in_channels=2 * Config.d, out_channels=2 * Config.d, kernel_size=3,
                                                dilation=3 ** i, padding="same") for i in range(4) for _ in range(2)])
        self.highway_blocks.extend([HighwayConv(in_channels=2 * Config.d, out_channels=2 * Config.d, kernel_size=3,
                                                dilation=1, padding="same") for _ in range(2)])
        self.highway_blocks.extend([HighwayConv(in_channels=2 * Config.d, out_channels=2 * Config.d, kernel_size=1,
                                                dilation=1, padding="same") for _ in range(2)])

    def forward(self, L):   # (bs, N)
        y = self.embedding(L)   # (bs, N, Config.e)
        y = y.permute(0, 2, 1)  # (bs, Config.e, N)
        y = F.relu(self.conv1(y))   # (bs, 2 * Config.d, N)
        y = self.conv2(y)   # (bs, 2 * Config.d, N)
        for i in range(len(self.highway_blocks)):
            y = self.highway_blocks[i](y)   # (bs, 2 * Config.d, N)

        # K: (bs, Config.d, N)
        # V: (bs, Config.d, N)
        K, V = y.chunk(2, dim=1)  # Split along d axis
        return K, V


class AudioEnc(nn.Module):
    """
    Encodes an input mel spectrogram S of shape [b, F, T] (representing audio of length T) into asdf matrix Q (Query) of
    shape [b, d, T]
    """

    def __init__(self):
        super(AudioEnc, self).__init__()

        self.conv1 = Conv1d(in_channels=Config.F, out_channels=Config.d, kernel_size=1, dilation=1, padding="causal")
        self.conv2 = Conv1d(in_channels=Config.d, out_channels=Config.d, kernel_size=1, dilation=1, padding="causal")
        self.conv3 = Conv1d(in_channels=Config.d, out_channels=Config.d, kernel_size=1, dilation=1, padding="causal")

        self.highway_blocks = nn.ModuleList()
        self.highway_blocks.extend([HighwayConv(in_channels=Config.d, out_channels=Config.d, kernel_size=3,
                                                dilation=3 ** i, padding="causal") for i in range(4) for _ in range(2)])
        self.highway_blocks.extend([HighwayConv(in_channels=Config.d, out_channels=Config.d, kernel_size=3, dilation=3,
                                                padding="causal") for _ in range(2)])

    def forward(self, S):   # (bs, n_mels, ts)
        y = F.relu(self.conv1(S))   # (bs, Config.d, ts)
        y = F.relu(self.conv2(y))   # (bs, Config.d, ts)
        y = self.conv3(y)   # (bs, Config.d, ts)
        for i in range(len(self.highway_blocks)):
            y = self.highway_blocks[i](y)   # (bs, Config.d, ts)
        return y


class AudioDec(nn.Module):
    """ Estimates asdf mel spectrogram from the seed matrix R'=[R,Q] where R' has shape [b, 2d, T]. """

    def __init__(self):
        super(AudioDec, self).__init__()

        self.conv1 = Conv1d(in_channels=2 * Config.d, out_channels=Config.d, kernel_size=1, dilation=1,
                            padding="causal")

        self.highway_blocks = nn.ModuleList()
        self.highway_blocks.extend([HighwayConv(in_channels=Config.d, out_channels=Config.d, kernel_size=3,
                                                dilation=3 ** i, padding="causal") for i in range(4)])
        self.highway_blocks.extend([HighwayConv(in_channels=Config.d, out_channels=Config.d, kernel_size=3,
                                                dilation=1, padding="causal") for _ in range(2)])

        self.conv2_1 = Conv1d(in_channels=Config.d, out_channels=Config.d, kernel_size=1, dilation=1, padding="causal")
        self.conv2_2 = Conv1d(in_channels=Config.d, out_channels=Config.d, kernel_size=1, dilation=1, padding="causal")
        self.conv2_3 = Conv1d(in_channels=Config.d, out_channels=Config.d, kernel_size=1, dilation=1, padding="causal")

        self.conv3 = Conv1d(in_channels=Config.d, out_channels=Config.F, kernel_size=1, dilation=1, padding="causal")

    def forward(self, R):   # R_: (bs, 2 * Config.d, T)
        y = self.conv1(R)   # (bs, Config.d, T)
        for i in range(len(self.highway_blocks)):
            y = self.highway_blocks[i](y)   # (bs, Config.d, T)
        y = F.relu(self.conv2_1(y)) # (bs, Config.d, T)
        y = F.relu(self.conv2_2(y)) # (bs, Config.d, T)
        y = F.relu(self.conv2_3(y)) # (bs, Config.d, T)
        y = self.conv3(y)   # (bs, Config.F, T)

        # y: bs, n_mels, ts
        return y, F.sigmoid(y)


class Attention(nn.Module):
    """
    Takes the following matrices as input (N=audio length, T=text length):
        - Text key K: [b, d, N]
        - Text value V: [b, d, N]
        - Spectrogram query Q: [b, d, T]

    Returns the attention matrix A of shape [b, N, T] and the attention result R of shape [b, d, T]
    """

    def __init__(self):
        super(Attention, self).__init__()

    def forward(
            self,
            K,  # (bs, Config.d, N)
            V,  # (bs, Config.d, N)
            Q,  # (bs, Config.d, T)
            force_incremental=False,
            previous_position=None, # (1)
            previous_att=None,  # (1, N, T)
            current_time=None,  # time in range(T)
            flag=False
    ):
        # Create attention matrix. A[b,n,t] evaluates how strongly
        # the N-th character and the T-th mel spectrum are related

        # K transposed: (bs, N, Config.d)
        # Q : (bs, Config.d, T)
        # A : (bs, N, T)
        A = torch.bmm(K.transpose(1, 2), Q) / np.sqrt(Config.d)
        A = F.softmax(A, dim=1)  # Softmax along char axis

        # During inference, force A to be diagonal
        # get the current character to look at (max along N axis)
        _, current_position = torch.max(A[:, :, current_time], dim=1)  # [b]
        if force_incremental and previous_att is not None:
            # Set previous steps for attention weights
            A[:, :, :current_time] = previous_att[:, :, :current_time]

            difference = current_position - previous_position
            # For each batch, check if the attention needs to be forcibly set
            force_needed = (difference < -1) | (difference > 3)  # [b] : (1)
            print(difference)

            # Repeat the bool tensor N times, so we get a mask for the current time column
            mask = force_needed.unsqueeze(1).repeat(1, A.shape[1])  # [1, N]

            # Kronecker Delta: 1 at index previous_position+1, 0 everywhere else.
            # delta: (bs, N) : (1, N)
            delta = torch.zeros([A.shape[0], A.shape[1]], device=A.device)

            # We must use 'scatter' to index with a tensor. We want something like 'delta[:, previous_position+1] = 1'
            idx = (previous_position + 1).clamp(0, delta.shape[1] - 1).unsqueeze(1).repeat(1, A.shape[1])   # (1, N)
            delta = delta.scatter_(1, idx, torch.ones([A.shape[0], A.shape[1]], device=A.device))



            # if flag:
            #     p = min(previous_position[0] + 1, A.shape[1]-1)
            # else:
            #     p = min(previous_position[0], A.shape[1]-1)
            #
            # delta[:, p] = 1
            # print(delta)



            # For each batch, select either the original column, or the delta column
            A[:, :, current_time] = torch.where(mask, delta, A[:, :, current_time])
            _, current_position = torch.max(A[:, :, current_time], dim=1)

        R = torch.bmm(V, A)  # [b, d, T]

        # A (bs, N, T)
        # R (bs, d, T)
        return A, R, current_position


class Text2Mel(nn.Module):
    """
    Encodes asdf text L of shape [b, N] given the previously generated mel spectrogram S of shape [b, T, F] into asdf new mel
    spectrogram Y of shape [b, F, T]
    """

    def __init__(self):
        super(Text2Mel, self).__init__()
        self.textEnc = TextEnc()
        self.audioEnc = AudioEnc()
        self.audioDec = AudioDec()
        self.attention = Attention()

    def forward(
            self,
            L,  # (bs, ts)
            S,  # (bs, n_mels, ts)
            force_incremental_att=False,
            previous_att_position=None,
            previous_att=None,
            current_time=None,
            flag=False
    ):
        # K: (bs, Config.d, N)
        # V: (bs, Config.d, N)
        K, V = self.textEnc(L)

        # Q: (bs, Config.d, ts)
        Q = self.audioEnc(S)

        # A (bs, N, T)
        # R (bs, d, T)
        A, R, current_position = self.attention(K, V, Q, force_incremental_att, previous_att_position, previous_att,
                                                current_time, flag)

        # R_: (bs, 2 * Config.d, T)
        R_ = torch.cat((Q, R), dim=1)  # Concatenate along channels

        # Y_logits: (bs, n_mels, ts)
        # Y: (bs, n_mels, ts)
        Y_logits, Y = self.audioDec(R_)
        return Y_logits, Y, A, current_position


class SSRN(nn.Module):
    """
    Spectrogram super resolution network. Converts asdf mel spectrogram Y of shape [b, F, T] to asdf linear STFT spectrogram
    Z of shape [b, F', T]
    """

    def __init__(self):
        super(SSRN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(Conv1d(in_channels=Config.F, out_channels=Config.c, kernel_size=1, dilation=1,
                                  normalize=True))
        self.layers.extend([HighwayConv(in_channels=Config.c, out_channels=Config.c, kernel_size=3, dilation=3 ** i)
                            for i in range(2)])

        for _ in range(2):
            self.layers.append(ConvTranspose1d(Config.c, Config.c, kernel_size=2, dilation=1, stride=2))
            self.layers.append(HighwayConv(in_channels=Config.c, out_channels=Config.c, kernel_size=3, dilation=1))
            self.layers.append(HighwayConv(in_channels=Config.c, out_channels=Config.c, kernel_size=3, dilation=3))

        self.layers.append(Conv1d(in_channels=Config.c, out_channels=2 * Config.c, kernel_size=1, dilation=1,
                                  normalize=True))
        self.layers.extend([HighwayConv(in_channels=2 * Config.c, out_channels=2 * Config.c, kernel_size=3, dilation=1)
                            for _ in range(2)])

        self.layers.append(Conv1d(in_channels=2 * Config.c, out_channels=Config.F_, kernel_size=1, dilation=1,
                                  normalize=True))

        for _ in range(2):
            self.layers.append(Conv1d(in_channels=Config.F_, out_channels=Config.F_, kernel_size=1, dilation=1,
                                      normalize=True))
            self.layers.append(nn.ReLU())

        self.layers.append(Conv1d(in_channels=Config.F_, out_channels=Config.F_, kernel_size=1, dilation=1,
                                  normalize=True))

    def forward(self, Y):
        y = Y
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        Z_logits = y
        Z = F.sigmoid(Z_logits)
        return Z_logits, Z


class MELSSRN(nn.Module):
    """
    Spectrogram super resolution network. Converts asdf mel spectrogram Y of shape [b, F, T] to asdf linear STFT spectrogram
    Z of shape [b, F', T]
    """

    def __init__(self):
        super(MELSSRN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(Conv1d(in_channels=Config.F, out_channels=Config.d, kernel_size=1, dilation=1,
                                  normalize=True))
        self.layers.extend([HighwayConv(in_channels=Config.d, out_channels=Config.d, kernel_size=3, dilation=3 ** i)
                            for i in range(2)])

        for _ in range(2):
            self.layers.append(ConvTranspose1d(Config.d, Config.d, kernel_size=2, dilation=1, stride=2))
            self.layers.append(HighwayConv(in_channels=Config.d, out_channels=Config.d, kernel_size=3, dilation=1))
            self.layers.append(HighwayConv(in_channels=Config.d, out_channels=Config.d, kernel_size=3, dilation=3))

        self.layers.append(Conv1d(in_channels=Config.d, out_channels=Config.d, kernel_size=1, dilation=1,
                                  normalize=True))
        self.layers.extend([HighwayConv(in_channels=Config.d, out_channels=Config.d, kernel_size=3, dilation=1)
                            for _ in range(2)])

        self.layers.append(Conv1d(in_channels=Config.d, out_channels=Config.F, kernel_size=1, dilation=1,
                                  normalize=True))

        for _ in range(2):
            self.layers.append(Conv1d(in_channels=Config.F, out_channels=Config.F, kernel_size=1, dilation=1,
                                      normalize=True))
            self.layers.append(nn.ReLU())

        self.layers.append(Conv1d(in_channels=Config.F, out_channels=Config.F, kernel_size=1, dilation=1,
                                  normalize=True))

    def forward(self, Y):
        y = Y
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        Z_logits = y
        Z = F.sigmoid(Z_logits)
        return Z_logits, Z


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                dilation=dilation,
            ),
            nn.LeakyReLU(0.2),
            WNConv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1,
            ),
        )
        self.shortcut = WNConv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1,
            )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class Generator(nn.Module):
    def __init__(
            self,
            input_size=80, # 80
            ngf=32,    # 32
            n_residual_layers=3   # 3
    ):
        super().__init__()

        ratios = [8, 8, 2, 2]
        # TODO: In Config hop length is set to 276
        self.hop_length  = np.prod(ratios) # 256
        mult = int(2 ** len(ratios))    # 16


        # conv_layer: in=80, out=512, kernel=(7,), stride=(1,)
        self.conv_layer1 = nn.Sequential(
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0)
        )

        self.upsampling_blocks = nn.ModuleList()
        for i, ratio in enumerate(ratios):
            self.upsampling_blocks.extend([
                nn.LeakyReLU(0.2),

                # The output sequence is multiplied by factor 8, 8, 2, 2 => 256
                WNConvTranspose1d(
                    in_channels=mult * ngf,
                    out_channels=mult * ngf // 2,
                    kernel_size = ratio * 2,
                    stride = ratio,
                    padding = ratio // 2 + ratio % 2,
                    output_padding = ratio % 2
                )
            ])

            for j in range(n_residual_layers):
                self.upsampling_blocks.extend([
                    ResnetBlock(
                        dim=mult * ngf // 2,
                        dilation=3 ** j
                    )
                ])

            mult //= 2

        self.output_layer = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(
                in_channels=ngf, # 32
                out_channels=1,
                kernel_size=7,
                padding=0
            ),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, mel):
        # mel: (bs, n_mels, n_frames)
        x = self.conv_layer1(mel) # x: (bs, 512, n_frames)

        for i in range(len(self.upsampling_blocks)):
            x = self.upsampling_blocks[i](x)

        # x: (bs, 32, n_frames * 256) : (bs, 32, n_frames * hop_length) : (bs, 32, n_samples)
        # out: (bs, 1, n_samples)
        out = self.output_layer(x)

        return out


class NLayerDiscriminator(nn.Module):
    def __init__(
            self,
            n_filters_start,    # 16
            n_layers,   # 4
            down_sampling_factor    # 4
    ):
        super().__init__()
        model = nn.ModuleDict()

        # layer_0
        # x: (bs, 1, n_samples)
        # out: (bs, 16, n_samples)
        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(
                in_channels=1,
                out_channels=n_filters_start,
                kernel_size=15
            ),
            nn.LeakyReLU(0.2, True)
        )

        n_filters = n_filters_start
        stride = down_sampling_factor

        # layer_1:
        # in: (bs, 16, n_samples)
        # out: (bs, 64, n_samples / 4)

        # layer_2:
        # in: (bs, 64, n_samples / 4)
        # out: (bs, 256, n_samples / 16)

        # layer_3:
        # in: (bs, 256, n_samples / 16)
        # out: (bs, 1024, n_samples / 64)

        # layer_4:
        # in: (bs, 1024, n_samples / 64)
        # out: (bs, 1024, n_samples / 256)

        for n in range(1, n_layers + 1):
            n_filters_prev = n_filters
            n_filters = min(n_filters * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    in_channels=n_filters_prev,
                    out_channels=n_filters,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding = stride * 5,
                    groups=n_filters_prev // 4
                ),
                nn.LeakyReLU(0.2, True)
            )

        n_filters = min(n_filters * 2, 1024)

        # layer_5:
        # in: (bs, 1024, n_samples / 256)
        # out: (bs, 1024, n_samples / 256)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(
                in_channels=n_filters_prev,
                out_channels=n_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.LeakyReLU(0.2, True)
        )

        # layer_6:
        # in: (bs, 1024, n_samples / 256)
        # out: (bs, 1, n_samples / 256)
        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            in_channels=n_filters,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.model = model

    def forward(self, x):
        # x: (bs, 1, n_samples)
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)

        # results: outputs of 6 layers :
        # [
        # layer_0: (bs, 16, n_samples),
        # layer_1: (bs, 64, n_samples / 4),
        # layer_2: (bs, 256, n_samples / 16),
        # layer_3: (bs, 1024, n_samples / 64),
        # layer_4: (bs, 1024, n_samples / 256),
        # layer_5: (bs, 1024, n_samples / 256),
        # layer_6: (bs, 1, n_samples / 256),
        # ]
        return results


class Discriminator(nn.Module):
    def __init__(
            self,
            num_D,
            n_filters_start,
            n_layers,
            down_sampling_factor
    ):
        super().__init__()

        self.model = nn.ModuleDict()

        for i in range(num_D):
            self.model[f'disc_{i}'] = NLayerDiscriminator(
                n_filters_start=n_filters_start,
                n_layers=n_layers,
                down_sampling_factor=down_sampling_factor
            )

        self.downsample = nn.AvgPool1d(
            kernel_size=4,
            stride=2,
            padding=1,
            count_include_pad=False
        )

        self.apply(weights_init)

    def forward(self, x):
        # x: (bs, 1, n_samples)
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))

            # x downsamples by factor 2
            x = self.downsample(x)

        # results from 4 discriminators with 7 outputs each
        return results


if __name__ == '__main__':

    netG = Generator(
        input_size=80,
        ngf=32,
        n_residual_layers=3
    )

    net_D = Discriminator(
        num_D=4,
        n_filters_start=16,
        n_layers=4,
        down_sampling_factor=4
    )

    data = torch.zeros(4, 1, 8192)
    results = net_D(data)

    print(len(results))
    print(len(results[0]))

    for i, out in enumerate(results[1]):
        print(f'layer_{i}:', out.size())






def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class HiFiGenerator(torch.nn.Module):
    def __init__(self, h):
        super(HiFiGenerator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

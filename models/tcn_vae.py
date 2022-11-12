import torch
import torch.nn as nn


class Softplus(nn.Module):
    def __init__(self, eps: float):
        super(Softplus, self).__init__()
        self.eps = eps
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(x) + self.eps


class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, final=False, forward=True):
        super(CausalConvolutionBlock, self).__init__()

        Conv1d = torch.nn.Conv1d if forward else torch.nn.ConvTranspose1d
        
        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size, forward=True):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        # double the dilation size if forward, if backward
        # we start at the final dilation and work backwards
        dilation_size = 1 if forward else 2**depth
        
        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size,
                forward,
            )]
            # double the dilation at each step if forward, otherwise
            # halve the dilation
            dilation_size = dilation_size * 2 if forward else dilation_size // 2

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Spatial(nn.Module):
    def __init__(self, channels, dropout, forward=True):
        super(Spatial, self).__init__()
        Conv1d = nn.Conv1d if forward else nn.ConvTranspose1d
        self.network = nn.Sequential(
            Conv1d(channels, channels, 1),
            nn.BatchNorm1d(num_features=channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class CausalCNNVEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, channels: int, depth: int, reduced_size: int,
                 out_channels: int, kernel_size: int, softplus_eps: float, 
                 sd_output: bool = True):
        super(CausalCNNVEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze,
        )
        self.linear_mean = torch.nn.Linear(reduced_size, out_channels)
        self.sd_output = sd_output
        if self.sd_output:
            self.linear_sd = torch.nn.Sequential(
                torch.nn.Linear(reduced_size, out_channels),
                Softplus(softplus_eps),
            )

    def forward(self, x):
        out = self.network(x)
        if self.sd_output:
            return self.linear_mean(out), self.linear_sd(out)
        return self.linear_mean(out).squeeze()

class CausalCNNVDecoder(torch.nn.Module):
    """
    Variational decoder.
    """
    def __init__(self, k, width, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNNVDecoder, self).__init__()
        self.in_channels = in_channels
        self.width = width
        self.linear1 = torch.nn.Linear(k, in_channels)
        self.linear2 = torch.nn.Linear(in_channels, in_channels * width)
        self.causal_cnn = CausalCNN(
            in_channels, channels, depth, out_channels, kernel_size,
            forward=False,
        )
        
    def forward(self, x):
        """
        Returns a reconstruction of the original 8x600 ECG, by decoding
        the given compression.
        """
        B, _ = x.shape
        # from (BxK) to (BxC)
        out = self.linear1(x)
        # from (BxC) to (Bx(C*600))
        out = self.linear2(out)
        # from (Bx(C*600)) to (BxCx600)
        out = out.view(B, self.in_channels, self.width)
        # deconvolve through the causal CNN
        out = self.causal_cnn(out)

        return out.view(B, -1)

class TCN_VAE(nn.Module):

    def __init__(self, encoder_params, decoder_params):
        super().__init__()
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.encoder = CausalCNNVEncoder(**encoder_params)
        self.decoder = CausalCNNVDecoder(**decoder_params)
    
    def reparameterize(self, mu, sd):
        """Sample from a Gaussian distribution with given mean and s.d."""
        eps = torch.randn_like(sd)
        return mu + eps * sd
    
    def forward(self, x):
        """Return reconstructed input after compressing it"""
        x = x.view(-1, self.decoder_params['out_channels'], self.decoder_params['width'])
        enc_mu, enc_sd = self.encoder(x)
        z = self.reparameterize(enc_mu, enc_sd)
        recon_x = self.decoder(z)
        return recon_x, z, (enc_mu, enc_sd)
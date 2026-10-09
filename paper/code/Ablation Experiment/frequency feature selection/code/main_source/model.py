"""RecurFlex: multiscale temporal convolutions and a residual bottleneck BiGRU."""
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel, dilation=1, pool=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(inputs, outputs, kernel, padding="same", dilation=dilation, bias=False)
        self.norm = nn.LayerNorm(outputs)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(pool, pool)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.pool(self.drop(self.gelu(x)))


class UpBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel, dropout):
        super().__init__()
        self.block = ConvBlock(inputs, outputs, kernel, dropout=dropout)
        self.up = nn.ConvTranspose1d(outputs, outputs, 2, stride=2, bias=False)

    def forward(self, x):
        return self.up(self.block(x))


class ConvEncoderDecoder(nn.Module):
    def __init__(self, electrodes, features, dropout):
        super().__init__()
        widths = [64, 64, 128, 256, 512, 512]
        kernels = [7, 7, 5, 5, 5]
        dilations = [1, 2, 3, 1, 2]
        self.reduction = ConvBlock(electrodes * features, 64, 3, dropout=dropout)
        self.encoder = nn.ModuleList(
            ConvBlock(widths[i], widths[i + 1], kernels[i], dilations[i], 2, dropout)
            for i in range(5)
        )
        self.decoder = nn.ModuleList(
            UpBlock(widths[i + 1] if i == 4 else 2 * widths[i + 1], widths[i], kernels[i], dropout)
            for i in range(4, -1, -1)
        )
        self.output = nn.Conv1d(128, 5, 1)


class RecurFlex(nn.Module):
    def __init__(self, electrodes=16, features=43, dropout=0.1):
        super().__init__()
        self.electrodes = electrodes
        self.features = features
        self.network = ConvEncoderDecoder(electrodes, features, dropout)
        self.context_norm = nn.LayerNorm(512)
        self.gru = nn.GRU(512, 128, batch_first=True, bidirectional=True)
        self.context_output = nn.Linear(256, 512)
        self.context_gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:3] != (self.electrodes, self.features) or x.shape[-1] % 32:
            raise ValueError("Expected (B, electrodes, features, T), with T divisible by 32.")
        z = self.network.reduction(x.flatten(1, 2))
        skips = []
        for block in self.network.encoder:
            skips.append(z)
            z = block(z)
        context, _ = self.gru(self.context_norm(z.transpose(1, 2)))
        z = z + self.context_gain * self.context_output(context).transpose(1, 2)
        for block, skip in zip(self.network.decoder, reversed(skips)):
            z = torch.cat([block(z), skip], dim=1)
        return self.network.output(z)

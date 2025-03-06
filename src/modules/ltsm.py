"""LTSM layer model."""

import torch.nn as nn


class SLTSM(nn.Module):
    """LTSM model without worrying about hidden state nor the layout of the data, but expect input as convolution layout."""

    def __init__(self, dimension: int, n_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.ltsm = nn.LSTM(dimension, dimension, n_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.ltsm(x)
        if self.skip:
            y += x
        return y.permute(1, 2, 0)

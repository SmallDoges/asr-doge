"""Normalization modules."""

import torch
import torch.nn as nn
import typing as tp

from einops import rearrange as r


class ConvLayerNorm(nn.LayerNorm):
    """Convolution-friendly LayerNorm that moves channels to last dimension before running normalization,
    and moves them back to original position right after.
    """

    def __init__(
        self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs
    ):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = r(x, "b ... t -> b t ...")
        super().forward(x)
        x = r(x, "b t ... -> b ... t")
        return

"""Convolutional layers wrappers and utilities."""

import math
import typing as tp
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_norm import ConvLayerNorm
from torch.nn.utils import weight_norm, spectral_norm

CONV_NORMALIZATIONS = frozenset(
    [
        "none",
        "weight_norm",
        "spectral_norm",
        "layer_norm",
        "time_layer_norm",
        "time_group_norm",
    ]
)


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    else:
        # We have already check if "norm" is in CONV_NORMALIZATIONS,
        # so any other choice doesn't need reparametrization.
        return module


def get_norm_module(
    module: nn.Module, causal: bool, norm: str = "none", **norm_kwargs
) -> nn.Module:
    """Return the proper normalization module. If the causal is true, ensure that the returned module is causal,
    or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        nn.Identity()


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames - 1) * stride) + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
):
    """Add extra padding for the 1d convolution."""
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "zero",
    value: float = 1.0,
):
    """Tiny wrapper around F.pad"""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if max_pad >= length:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unPad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert length >= padding_left + padding_right
    end = length - padding_right
    return x[..., padding_left:end]


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this convolution
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.List[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        return self.norm(x)


class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization applied to this convolution
    to provide the uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.List[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        return self.norm(x)


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied on this convolution
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.List[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm
        )
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        return self.norm(x)


class NormConvTranspose2d(nn.Module):
    """Wrapper around ConvTranspose2d and normalization applied on this convolution
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.List[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose2d(*args, **kwargs), norm
        )
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        return self.norm(x)


class SConv1d(nn.Module):
    """SConv1d with some built-in handling of asymmetric and causal padding and normalization."""

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: int = 1,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.List[str, tp.Any] = {},
        pad_mode: str = "reflect",
    ):
        super().__init__()
        # warn the user about the stride and dilation unsual setup
        if stride > 1 and dilation > 1:
            warnings.warn(
                "SConv1d has been initialized with stride > 1 and dilation > 1"
                f"kernel size: {kernel_size}, stride: {stride}, dilation: {dilation}"
            )
        self.causal = causal
        self.pad_mode = pad_mode
        self.conv = nn.Conv1d(
            in_chs,
            out_chs,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    def forward(self, x):
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(
            x, kernel_size, stride, padding_total
        )
        # left padding for causal
        if self.causal:
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        # asymmetric padding for the odd stride
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(
                x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    """SConvTranspose1d with some built-in handling of asymmetric, causal padding and normalization."""

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.List[str, tp.Any] = {},
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        self.convtr = NormConvTranspose1d(
            in_chs,
            out_chs,
            kernel_size,
            stride,
            dilation,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert (
            self.causal or self.trim_right_ratio >= 1.0
        ), "`trim_right_ratio` != 0 only make sense for causal convolution"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride
        x = self.convtr(x)
        # we will only trim fixed padding
        if self.causal:
            # if trim_right_ratio == 1.0 we will remove everything from the right
            padding_left = math.ceil(padding_total * self.trim_right_ratio)
            padding_right = padding_total - padding_left
            x = unPad1d(x, (padding_left, padding_right))
        else:
            # asymmetric padding for the odd stride
            padding_left = padding_total // 2
            padding_right = padding_total - padding_left
            x = unPad1d(x, (padding_left, padding_right))
        return x

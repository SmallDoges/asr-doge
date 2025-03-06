"""Neural Audio Encoder SEANet-based encoder for audio2tokens and decoder tokens-audio"""

import typing as tp

import numpy as np
import torch.nn as nn

from .conv import SConv1d, SConvTranspose1d
from .ltsm import SLTSM


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.
    Args:
        dimension: (int) = Dimension of the input/output
        kernel_sizes: (list) = List of kernel sizes for the convolutions.
        dilations: (list) = List of dilations for the convolutions.
        activation: (str) = Activation function.
        activation_params: (dict) = Parameters to provide to the activation function.
        norm: (str) = Normalization method.
        norm_params: (dict) = Parameters to provide to the underlying normalization used along with the convolutions.
        causal: (bool) = Whether to use fully causal convolutions.
        pad_mode: (str) = Padding mode for the convolutions.
        compress: (int) = Reduced dimentionality to the residual branches (from Demucs v3)
        true_skip: (bool) = Whether to use true skip connection or simple convolution as the skip connection.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [3, 1],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 0.1},
        norm: str = "weight_norm",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(
            dilations
        ), "Number of kernel sizes should be the same of the dilations"
        act = getattr(nn, activation)
        hidd = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidd
            out_chs = dim if len(kernel_sizes) - 1 else hidd
            block += [
                act(**activation_params),
                SConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_params=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut = nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_params=norm_params,
                pad_mode=pad_mode,
                causal=causal,
            )

    def forward(self, x):
        return self.block + self.shortcut


class SEANetEncoder(nn.Module):
    """SEANet encoder
    Args:
        channels: int = Audio channels.
        dimension: int = Intermediate representation dimension.
        n_filters: int = Base width of the model.
        n_residual_layers: int = Number of the residual layers.
        ratios: Sequence[int] = Kernel size and stride ratio. (The encoder uses downsample ratio and its normal order)
        activation: str = Activation function.
        activation_params: dict = Parameters to provide to the activation function.
        norm: str = Normalization method.
        norm_params: dict = Parameters to provide to the underelying normalization usde along with the convolution.
        kernel_size: int = Kernel size of the initial convolution.
        last_kernel_size: int = Kernel size of the final convolution.
        residual_kernel_size: int = Kernel size of all the residual layers.
        dilation_base: int = How much to increase dilation for each of the residual layer.
        causal: bool = Whether to use fully causal convolutions.
        pad_mode: str = Padding mode for the convolution.
        compress: int = Reduced dimentionality for the residual branches. (from Demucs v3)
        true_skip: bool = Whether to use true skip or simple convolution as the skip connection (streamable)
            for the residual network block.
        ltsm_layers: int = Number of the ltsm layers at the end of the encoder.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 0.1},
        norm: str = "weight_norm",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = False,
        ltsm_layers: int = 1,
    ):
        super().__init__()

        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation)
        mult = 1
        model: tp.List[nn.Module] = [
            act(**activation_params),
            SConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm[norm_params],
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        # downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # add residual layers
            for j in range(self.n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * self.n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]
            # add downsample layer
            model += [
                act(**activation_params),
                SConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2
        if ltsm_layers:
            model += [SLTSM(mult * n_filters, num_layers=ltsm_layers)]
        model += [
            SConv1d(
                mult * n_filters,
                dimension,
                kernel_size=ratio * 2,
                stride=ratio,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SEANetDecoder(nn.Module):
    """SEANet decoder.
    Args:
        channels: int = Audio channels.
        dimension: int = Intermediate representation dimension.
        n_filters: int = Base width of the model.
        n_residual_layers: int = Number of residual layers.
        ratios: Sequence[int] = Kernel size and stride ratio. (The decoder uses upsample ratio and its reverse order)
        activation: str = Activation function.
        activation_params: dict = Parameters to provide to the activation function.
        final_activation: str = Final activation function after all the convolutions.
        final_activation_params: str = Parameters to provide to the final activation function.
        norm: str = Normalization method.
        norm_params: dict = Parameters to provide for the underlying normalization used along with the convolution.
        kernel_size: int = Kernel size for the initial convolutions.
        last_kernel_size: int = Kernel size for the final convolutions.
        residual_kernel_size: int = Kernel size for the residual layers.
        dilation_base: int = How much to increase the dilation with each residual layer.
        causal: bool = Whether to use fully causal convolution.
        pad_mode: str = Padding mode for the convolution.
        compress: int = Reduced dimentionality for the residual branches. (from Demucs v3)
        true_skip: bool = Whether to use true skip connection or simple convolution (streamable)
            as the skip connection in the residual network blocks.
        ltsm_layers: int = Number of ltsm layers at the end of the decoder.
        trim_right_ratio: float = Ratio of the trimming at the right of the transposed convolutions under the causal setup.
            If equal is 1.0, that mean that all of the trimming is done at the right.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        norm: str = "weight_norm",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = False,
        ltsm_layers: int = 2,
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.n_residual_layers = n_residual_layers
        self.ratios = ratios
        del ratios
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            SConvTranspose1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        if ltsm_layers:
            model += [SLTSM(mult * n_filters, num_layers=ltsm_layers)]

        # upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # add upsampling layers
            model += [
                act(**activation_params),
                SConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            ]
            # add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]
            mult //= 2

        # add final layer
        model += [
            act(**activation_params),
            SConvTranspose1d(
                n_filters,
                channels,
                last_kernel_size,
                norm=norm,
                norm_params=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def test():
    import torch

    x = torch.rand(1, 1, 24000)
    encoder = SEANetEncoder()
    decoder = SEANetDecoder()

    e = encoder(x)
    d = decoder(x)

    assert list(e.shape) == [
        1,
        128,
        75,
    ], e.shape
    assert d.shape == x.shape, (d.shape, x.shape)


if __name__ == "__main__":
    test()

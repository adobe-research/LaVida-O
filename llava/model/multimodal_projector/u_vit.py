# ADOBE CONFIDENTIAL
# Copyright 2025 Adobe
# All Rights Reserved.
# NOTICE: All information contained herein is, and remains
# the property of Adobe and its suppliers, if any. The intellectual
# and technical concepts contained herein are proprietary to Adobe
# and its suppliers and are protected by all applicable intellectual
# property laws, including trade secret and copyright laws.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Adobe.

from diffusers.models.resnet import Downsample2D, Upsample2D
from einops import rearrange
from torch import nn
from functools import partial
# copied from ViTMin
class GeGluMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features,
            bias = False,
            drop = 0.0,
    ):
        super().__init__()
        norm_layer = partial(nn.RMSNorm, eps=1e-6,elementwise_affine=True)

        self.norm = norm_layer(in_features)
        self.w0 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU(approximate='tanh')
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x

class Simple_UVitBlock(nn.Module):
    def __init__(
        self,
        channels,
        ln_elementwise_affine,
        layer_norm_eps,
        use_bias,
        downsample: bool,
        upsample: bool,
        out_channels=None,
        add_mlp=None,
    ):
        super().__init__()
        out_channels = out_channels or channels
        if downsample:
            self.downsample = Downsample2D(
                channels,
                use_conv=True,
                padding=0,
                name="Conv2d_0",
                kernel_size=2,
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
                out_channels=out_channels,
            )
        else:
            self.downsample = None

        if upsample:
            self.upsample = Upsample2D(
                channels,
                use_conv_transpose=True,
                kernel_size=2,
                padding=0,
                name="conv",
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
                interpolate=False,
                out_channels=out_channels,
            )
        else:
            self.upsample = None
        if add_mlp:
            self.mlp = GeGluMlp(
                out_channels,
                out_channels * 2,

                bias = True,
            )
        else:
            self.mlp = None

    def forward(self, x,size=None):
        if len(x.shape) == 3:
            assert size is not None, f"Size must not be none if input is flattened"
        x = rearrange(x,'n (h w) d -> n d h w ',h=size[0],w=size[1])
        # print("before,", x.shape)
        if self.downsample is not None:
            # print('downsample')
            x = self.downsample(x)

        if self.upsample is not None:
            # print('upsample')
            x = self.upsample(x)
        x = rearrange(x,'n d h w  -> n (h w) d')
        if self.mlp is not None:
            x =self.mlp(x)
        return x

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

import torch
import torch.nn as nn
import math


class SpatialPool(nn.Module):
    def __init__(self, model_args, vision_tower):
        super().__init__()

        self.mode = model_args.mm_spatial_pool_mode
        self.stride = model_args.mm_spatial_pool_stride
        self.out_channels = getattr(model_args, "mm_spatial_pool_out_channels", vision_tower.hidden_size)

        if self.mode == "average":
            self.pool = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.mode == "conv":
            self.pool = nn.Conv2d(in_channels=vision_tower.hidden_size, out_channels=self.out_channels, kernel_size=self.stride, stride=self.stride)
            assert self.out_channels == vision_tower.hidden_size, f"self.out_channels{self.out_channels}!= vision_tower.hidden_size{vision_tower.hidden_size}  "
            with torch.no_grad():
                kernel_value = 1.0 / (self.stride * self.stride)
                weight = torch.zeros_like(self.pool.weight)
                for i in range(self.out_channels):
                    weight[i, i, :, :] = kernel_value
                self.pool.weight.copy_(weight)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pool}.")

    def forward(self, image_features, images, *args, **kwargs):
        ori_W = int(math.sqrt(image_features.shape[1] * images.shape[3] // images.shape[2]))
        ori_H = int(ori_W * images.shape[2] // images.shape[3])

        B, _, F = image_features.shape

        image_features_spatial = image_features.view(B, ori_H, ori_H, F).permute(0, 3, 1, 2)
        image_features_spatial_pool = self.pool(image_features_spatial)

        return image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous()

    @property
    def config(self):
        return {
            "mm_resampler_type": "spatial_pool",
            "mm_spatial_pool_stride": self.stride,
            "mm_spatial_pool_mode": self.mode,
            "mm_spatial_pool_out_channels": self.out_channels,
        }

    @property
    def hidden_size(self):
        return self.out_channels

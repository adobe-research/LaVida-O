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

import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def forward(self, prediction, target):
        loss = F.binary_cross_entropy_with_logits(prediction,target)
        return loss, {}


class BCELossWithQuant(nn.Module):
    def __init__(self, codebook_weight=1.):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, qloss, target, prediction, split):
        bce_loss = F.binary_cross_entropy_with_logits(prediction,target)
        loss = bce_loss + self.codebook_weight*qloss
        return loss, {"{}/total_loss".format(split): loss.clone().detach().mean(),
                      "{}/bce_loss".format(split): bce_loss.detach().mean(),
                      "{}/quant_loss".format(split): qloss.detach().mean()
                      }

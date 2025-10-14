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

from llava.model.multimodal_encoder.unitok.utils.data import normalize_01_into_pm1
from torchvision.transforms import transforms, InterpolationMode

class UnitokImageProcessor:

    def __init__(self):
        self._preprocess_fn = transforms.Compose([
            transforms.ToTensor(), normalize_01_into_pm1,
        ])

    def preprocess(self,image):
        return self._preprocess_fn(image).unsqueeze(0)

    

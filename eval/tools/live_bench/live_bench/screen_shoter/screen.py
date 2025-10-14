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

import base64
import io
from typing import List, Tuple

from live_bench.websites import Website
from PIL import Image


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class ScreenImage(object):
    def __init__(self, images: List[Image.Image], website: Website, shoter: str, screen_size: Tuple[int, int], capture_datetime: str):
        self.images = images
        self.website = website
        self.shoter = shoter
        self.screen_size = screen_size
        self.capture_datetime = capture_datetime

    def to_dict(self):
        return {"images": self.images, "website": self.website.get_info(), "shoter": self.shoter, "screen_size": self.screen_size, "capture_datetime": self.capture_datetime}

    def to_output_dict(self):
        output = self.to_dict()
        output["images"] = [image_to_base64(image) for image in self.images]
        return output

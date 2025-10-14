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

from PIL import Image, ImageOps
import cv2
import numpy as np
def pad_to_square_and_resize(img, size=1024):
    padded_img = ImageOps.pad(img, (max(img.size), max(img.size)), color=(0, 0, 0))
    resized_img = padded_img.resize((size, size))
    return resized_img
def transform_bbox(original_shape, bbox, target_size=1024):
    """
    Transforms a bounding box from the original image to the padded and resized image.

    Parameters:
    - original_shape: tuple (height, width) of the original image
    - bbox: list or tuple [x, y, width, height] in the original image
    - target_size: size of the final square image (default 1024)

    Returns:
    - Transformed bounding box [x, y, width, height] in the new image
    """
    orig_h, orig_w = original_shape
    x, y, w, h = bbox

    max_dim = max(orig_h, orig_w)
    pad_top = (max_dim - orig_h) // 2
    pad_left = (max_dim - orig_w) // 2

    x_padded = x + pad_left
    y_padded = y + pad_top

    scale = target_size / max_dim

    x_new = x_padded * scale
    y_new = y_padded * scale
    w_new = w * scale
    h_new = h * scale

    return [x_new, y_new, w_new, h_new]
    
def draw_bboxes(image, bboxes, labels, color=(0, 255, 0), thickness=2, font_scale=0.5,xyxy=False,nolabel=False,colors=None):
    """
    Draws bounding boxes and labels on an image.

    Parameters:
    - image: np.array of shape (H, W, 3)
    - bboxes: list of [x, y, width, height]
    - labels: list of strings corresponding to each bbox
    - color: color of the bounding box (default green)
    - thickness: thickness of the bounding box lines
    - font_scale: scale of the label text
    """
    for i,(bbox, label) in enumerate(zip(bboxes, labels)):
        if colors is not None:
            color = colors[i]
        x, y, w, h = map(int, bbox)
        # Draw rectangle
        if xyxy:
            cv2.rectangle(image, (x, y), ( w,  h), color, thickness)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        # Put label
        if nolabel:
            continue
        cv2.putText(image, label, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, lineType=cv2.LINE_AA)
    return image

def visualize_boxes(objects,image=None,**kwargs):
    labels = [x[0] for x in objects]
    bboxes = [x[1] for x in objects]
    if image is None:
        image = np.zeros((1024,1024,3)).astype(np.uint8)
    if isinstance(image,Image.Image):
        image = np.array(image)
    return Image.fromarray(draw_bboxes(image,bboxes, labels,xyxy=True,**kwargs))

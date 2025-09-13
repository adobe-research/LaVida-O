import json
import torch
import re
import argparse
def read_json(fp):
    with open(fp, 'r') as f:
        data = json.load(f)
    return data

def read_jsonl(fp):
    with open(fp, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def load_data(fp):
    if fp.endswith('.json'):
        return read_json(fp)
    elif fp.endswith('.jsonl'):
        return read_jsonl(fp)
    else:
        raise ValueError(f'Unsupported file format: {fp}')

PATTERN = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the Intersection over Union
    iou = intersection_area / union_area

    return iou


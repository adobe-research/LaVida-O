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



# print("Importing necessary libraries...")
import os
if __name__ == "__main__":
    os.environ['DEBUG_FIX_PADDING'] = '1'
    os.environ['NOT_ALWASY_DO_2DPOOL'] = '1'
    # groudning model always has these two env variables set
import pandas as pd
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX,SKIP_DOWN_SAMPLE
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
from accelerate import Accelerator
import copy
import torch
from llava.model.language_model.llada.generate import generate as llada_generate,cosine_schedule_2,logit_normal_schedule
from llava.model.language_model.llada.log_likelyhood import get_logits as llada_get_logits
import json
import time
import importlib
from llava.model.language_model.llada.generate import generate as llada_generate,wte,get_logits,get_num_transfer_tokens_sch
from tqdm.auto import tqdm
import numpy as np 
import torch.distributions as dists
from einops import rearrange
import torch.nn.functional as F
from llava.model.language_model.llada.generate import generate as llada_generate,wte,get_logits,get_num_transfer_tokens_sch
import re

def predict_grounding(model,image_processor,image, expressions, tokenizer,mode='refcoco',steps=None):
    conv_template = 'llada'
    # expressions = ["a cute dog","a boy", "ship"]
    mask_id = 126336
    # image = Image.open('images/port.png').convert('RGB')
    if isinstance(image, Image.Image):
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.bfloat16, device=model.device) for _image in image_tensor]
    else:
        image_tensor = image
    boxes = []
    if mode == 'refcoco':
        for expression in expressions:
            t00 = time.time()

            prompt_question = f''''<|startoftext|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\n Please locate {expression} in this image. Give bounding boxes in LOC format.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<LOC_BEGIN><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><LOC_END><|eot_id|>'''
            t01 = time.time()
            
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
            attention_mask=None
            position_ids=None
            image_sizes = [image.size]
            modalities=["image"]
            
            t0 = time.time()
            # print("Formating prompt (Total)", t0-t00,steps)
            # print("Formating prompt(String fomrating)", t01-t00,steps)
            # print("Tokenization of Text (String fomrating)", t0-t01,steps)
            (inputs, position_ids, attention_mask, _, inputs_embeds, _,raw_input_ids) = model.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, None, None, image_tensor, modalities, image_sizes=image_sizes,return_inputs=True)
            t1 = time.time()
            # print("Encoding", t1-t0,steps)
            if steps is None or steps <= 1:
                logits = get_logits(model.model,inputs_embeds)
                box_predictions = logits[raw_input_ids==mask_id].argmax(-1).view(-1,4)
            else:
                assert 4 % steps == 0, "steps must be a factor of 4"
                tokens_per_step = 4 // steps
                x_t = raw_input_ids[raw_input_ids==mask_id]
                _,x_t_pos = torch.where(raw_input_ids==mask_id)
                for step in range(steps):
                    logits = get_logits(model.model,inputs_embeds)
                    x_0 = logits[raw_input_ids==mask_id].argmax(-1)
                    probs = logits[raw_input_ids==mask_id][range(len(x_0)),x_0]
                    probs[~(x_t==mask_id)] = float('-inf')
                    transfer_index = torch.topk(probs, k=2, dim=-1).indices
                    x_t[transfer_index] = x_0[transfer_index]
                    embeds,_ = wte(model.model,x_0[transfer_index],False)
                    inputs_embeds[:,x_t_pos[transfer_index]] = embeds
                # breakpoint()
                box_predictions = x_0.view(-1,4)
            res = tokenizer.batch_decode(box_predictions)
            # print(res)
            t2  = time.time()
            # print("model.generate",t2-t1,steps)
            box = [ [int (y) for y in re.compile('<LOC_([0-9]+)>').findall(x)] for x in res]
            boxes.append(box[0])
    elif mode == 'grand':
        prompt = f"<image>\n Please describe this image with dense localized caption. Give bounding boxes in LOC format."
        mask_token = '<|mdm_mask|>'
        loc_string = [ f'<box_p> {expression} </box_p>'+f'<LOC_BEGIN>{mask_token*4}<LOC_END>' for expression in expressions]
        loc_string = ', '.join(loc_string)
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], f'There are following objects in the scene. {loc_string}')
        prompt_question = conv.get_prompt()
        prompt_question.removesuffix('<|start_header_id|>assistant<|end_header_id|>\n\n')
        prompt_question

        device = 'cuda'
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
        attention_mask=None
        position_ids=None
        image_sizes = [image.size]
        modalities=["image"]
        (inputs, position_ids, attention_mask, _, inputs_embeds, _,raw_input_ids) = model.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, None, None, image_tensor, modalities, image_sizes=image_sizes,return_inputs=True)
        logits = get_logits(model.model,inputs_embeds)
        box_predictions = logits[raw_input_ids==mask_id].argmax(-1).view(-1,4)
        res = tokenizer.batch_decode(box_predictions)
        boxes = [ [int (y) for y in re.compile('<LOC_([0-9]+)>').findall(x)] for x in res]
    return boxes


    
def load_image(image_file,image_root):
    image = None
    try:
        image_file = image_file.replace('s3://act-rec-data1/',image_root)
        image = Image.open(image_file).convert("RGB")
    except Exception as exn:
        print(f"Failed to open image {image_file}. Exception:", exn)
        raise exn
    return image


import torch

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute the IoU between two sets of boxes in xyxy format.
    boxes1: Tensor of shape (N, 4)
    boxes2: Tensor of shape (M, 4)
    Returns: Tensor of shape (N, M) with IoU values
    """
    # Intersection
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Union
    union_area = area1[:, None] + area2 - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-6)  # Add small value to avoid division by zero
    return iou

import torch

def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two tensors of shape (N, 4) in xyxy format.
    Returns a tensor of shape (N,) with IoU values.
    """
    # Intersection
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Union
    union_area = area1 + area2 - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-6)  # Add small value to avoid division by zero
    return iou


class RECDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, image_processor,image_root='data/'):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_root = image_root


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        image = load_image(item['s3_path'],self.image_root)
        expression = item['label']
        transformed_box = item['transformed_box']
        raw_gt_box = item['raw_gt_box']

        return {
            'image': np.array(image),
            'expression': expression,
            'transformed_box': transformed_box,
            'raw_gt_box': raw_gt_box,
            's3_path': item['s3_path']
        }

if __name__ == "__main__":
    import sys
    import wandb
    is_debug = '--debug' in sys.argv

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-m',default='checkpoints/lavida-o-v1.0')
    parser.add_argument('--short',action='store_true')
    parser.add_argument('--image_root',type=str,default='data/')
    args = parser.parse_args()
    pretrained = args.model

    model_name = "llava_llada"
    device = "cuda"
    device_map = "cuda"
    output_root = 'outputs/grounding/'
    accelerator = Accelerator()
    model_uuid = pretrained.replace('/', '_')
    output_path = os.path.join(output_root, f"grounding_results/{model_uuid}")
    if is_debug:
        output_path = output_path + '_debug'
    if accelerator.is_main_process:
        os.makedirs(output_path, exist_ok=True)
        # write config to output_path
        config = {
            "pretrained": pretrained,
            "model_name": model_name,
        }
        wandb.init(project="grounding-eval", name=f"grounding-{model_uuid}", config=config)
        with open(output_path+'/config.json', 'w') as f:
            json.dump(config, f, indent=4)
    accelerator.wait_for_everyone()
    print(f"Loading model from {accelerator.device}...")
    if accelerator.num_processes > 1:
        torch.cuda.set_device(accelerator.device)
    vision_kwargs = None

    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map,vision_kwargs=vision_kwargs,torch_dtype='bfloat16') # Add any other thing you want to pass in llava_model_args

    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)

    datasets = [
        'data/und_data/grounding_refcoco-unc_val.parquet',
        'data/und_data/grounding_refcoco-unc_testA.parquet',
        'data/und_data/grounding_refcoco-unc_testB.parquet',
        'data/und_data/grounding_refcocoplus-unc_val.parquet',
        'data/und_data/grounding_refcocoplus-unc_testA.parquet',
        'data/und_data/grounding_refcocoplus-unc_testB.parquet',
        'data/und_data/grounding_refcocog-umd_val.parquet',
        'data/und_data/grounding_refcocog-umd_test.parquet',     
    ]
    if args.short:
        datasets = [
            'data/und_data/grounding_refcoco-unc_val.parquet',
            'data/und_data/grounding_refcocoplus-unc_val.parquet',
            'data/und_data/grounding_refcocog-umd_val.parquet',
        ]
    all_res = []
    for data_path in datasets:
        dataset_name = os.path.basename(data_path).replace('.parquet', '')
        data = pd.read_parquet(data_path)
        seed = 42
        if is_debug:
            data = data.sample(n=100, random_state=seed).reset_index(drop=True)
        else:
            data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        rec_dataset = RECDataset(data, tokenizer, image_processor,args.image_root)
        dataloader = torch.utils.data.DataLoader(rec_dataset, batch_size=1, shuffle=False)
        dataloader = accelerator.prepare(dataloader)
        all_res = {}
        correct = 0
        total = 0
        with tqdm(dataloader, desc=f"Processing {data_path}",disable=not accelerator.is_main_process) as pbar:
            for batch in pbar:
                image = batch['image'][0]
                expression = batch['expression'][0]
                image = Image.fromarray(image.cpu().numpy()).convert("RGB")
                gt = batch['transformed_box'].cpu()
                try:
                    boxes = predict_grounding(model, image_processor, image, [expression], mode='refcoco',tokenizer=tokenizer,steps=1)
                    boxes = torch.tensor(boxes)
                    box_iou = pairwise_iou(boxes, gt).item()
                    boxes = boxes.tolist()
                except KeyboardInterrupt:
                    raise ValueError("Keyboard")
                except:
                    box_iou = 0
                    boxes = []
                # all_ious.append(box_iou)
                if box_iou >= 0.5:
                    correct += 1
                total += 1
                current_acc = correct / total if total > 0 else 0
                payload = {
                    "dataset": dataset_name,
                    "s3_path": batch['s3_path'][0],
                    "expression": expression,
                    "predicted_box": boxes,
                    "gt_box": gt.tolist(),
                    "box_iou": box_iou,
                    "correct": box_iou >= 0.5
                }
                all_res.update(payload)
                pbar.set_postfix({'IoU': box_iou, 'Acc@0.5': f'{current_acc:.4f}'})  
        # compute acc@0.5

        output_file = os.path.join(output_path, f"grounding_results_{dataset_name}_rank_{accelerator.process_index}.json")
        with open(output_file, 'w') as f:
            json.dump(all_res, f, indent=4)

        results = torch.tensor([correct,total], dtype=torch.float32).to(accelerator.device)
        results = accelerator.reduce(results, reduction = 'sum')
        correct, total = results[0].item(), results[1].item()
        acc = correct / (total + 1e-6)
        torch.cuda.empty_cache()
        print(f"{dataset_name} Acc: {acc:.4f} ({correct}/{total})")
        final_acc = {
            f"val/{dataset_name}_P@0.5": acc,
        }
        all_res.update(final_acc)
        if accelerator.is_main_process:
            with open(os.path.join(output_path, f"grounding_acc_{dataset_name}.json"), 'w') as f:
                json.dump(final_acc, f, indent=4)
        # print(f"Accuracy at IoU 0.5 for {data_path}: {acc_at_05}")
    for k,v in all_res.items():
        print(f"{k}: {v}")
    if accelerator.is_main_process:
        wandb.finish()

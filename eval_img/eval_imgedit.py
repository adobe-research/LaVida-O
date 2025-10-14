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

import datasets
import os
if __name__ == "__main__":
    os.environ['DEBUG_FIX_PADDING'] = '1'
    os.environ['NOT_ALWASY_DO_2DPOOL'] = '1'
import numpy as np
import os
import argparse
import accelerate
import torch
from llava.mm_utils import pad_to_square_and_resize,resize_and_center_crop
from tqdm.cli import tqdm
import pandas as pd
import json
from PIL import Image



parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from llava.eval.predict_t2i_edit import build_model
from llava.eval.predict_t2i_edit import text_to_image,create_plan_edit


parser = argparse.ArgumentParser()
ckpts =''
parser.add_argument('--ckpt',type=str,default=ckpts)
parser.add_argument('--output_path',type=str,default='outputs_gedit')
parser.add_argument('--fmt',type=str,default='jpg')
parser.add_argument('--cleanup',action='store_true')
parser.add_argument('--config',default='',type=str)
parser.add_argument('--name',default='',type=str)
parser.add_argument('--do_plan',action='store_true')
parser.add_argument('--dataset',default='imgedit',type=str)
parser.add_argument('--aug',default='crop',type=str)
args = parser.parse_args()
if args.dataset == 'imgedit':
    with open('data/imagedit/Benchmark/singleturn/singleturn.json') as f:
        data = json.load(f)
    all_entry = []
    for k,v in tqdm(data.items()):
        v['key'] = k
        v['input_image'] = Image.open(os.path.join('data/imagedit/Benchmark/singleturn/',v['id'])).convert('RGB')
        all_entry.append(v)
    dataset = all_entry
elif args.dataset == 'gedit':
    dataset = datasets.load_from_disk('data/gedit')
    is_en = np.array(dataset['instruction_language']) == 'en'
    is_en = np.where(is_en)[0]
    dataset = dataset.select(is_en)
else:
    raise NotImplementedError(f"Unknown dataset {args.dataset}")
accelerator = accelerate.Accelerator()
if accelerator.num_processes > 1:
    torch.cuda.set_device(accelerator.device)
tokenizer, model, image_processor = build_model(pretrained=args.ckpt)


token_map ={
        256:256,
        512:1024,
        1024:4096
    }
res = 1024
gen_dict = dict(tokenizer=tokenizer,
                sample_policy='multinomial',
                    confidence_policy='halton',
                    guidance_scale=1.2,
                    batch_size=1,
                    image_resolution=res,
                    n_tokens=token_map[1024], 
                    shift=5,
                    n_steps=64,
                    schedule="shift",
                    alg_temp=5,
                    dynamic_temperature=True,
                    schedule_temp='cosine2',
                    min_temperature=0.5,
                    micro_cond='')
if args.config:
        import yaml
        with open(args.config) as f:
            config_yaml = yaml.safe_load(f)
        gen_dict.update(config_yaml['config'])

def edit_one_image(src_image,instruction):
    if args.do_plan:
        # if 'add' in instruction.lower():
        plan = create_plan_edit(model,tokenizer,instruction,src_image,image_processor)
        gen_dict['plan'] = plan
    with torch.no_grad():
        images = text_to_image(model, instruction, edit_image=src_image,image_processor=image_processor,
                        **gen_dict
                )
    return images

local_rank = accelerator.process_index
world_size = accelerator.num_processes

n = len(dataset)
n_prompt_per_rank = n // world_size + 1
start = local_rank * n_prompt_per_rank 
all_evaluations = []

indices = np.arange(len(dataset))
rng = np.random.default_rng(420)
shuffled_indices = rng.permutation(indices)
ROOT=os.path.join(args.output_path,f'{args.dataset}_{args.name}_{args.aug}_do_plan_{args.do_plan}')
# print(ROOT)
# exit()
if accelerator.is_main_process:
    if os.path.exists(ROOT) and args.cleanup:
        import shutil
        shutil.rmtree(ROOT)
accelerator.wait_for_everyone()

os.makedirs(ROOT,exist_ok=True)
from tqdm.cli import tqdm
for i in tqdm(range(start,start+n_prompt_per_rank)):
    if i >= len(dataset):
        continue 
    
    i = shuffled_indices[i]
    i = int(i)
    entry = dataset[i]
    try:

        outpath = os.path.join(ROOT,f"{entry['key']}.jpg") #
        outpath_in = os.path.join(ROOT,f"{entry['key']}_in.jpg") #
        outpath_txt = os.path.join(ROOT,f"{entry['key']}.txt") #
        
        
        input_image = entry['input_image'].convert('RGB')
        if args.aug == 'pad':
            image = pad_to_square_and_resize(input_image,1024)
        elif args.aug == 'crop':
            image = resize_and_center_crop(input_image,1024)
        if args.dataset == 'imgedit':
            instruction = entry['prompt']
        elif args.dataset == 'gedit':
            instruction = entry['instruction']
        out = edit_one_image(image,instruction)
        image.save(outpath_in)
        out.save(outpath)
        with open(outpath_txt,'w') as f:
            f.write(instruction)
    except Exception as e:
        print(f"Error {entry['key']} {e.__traceback__}")

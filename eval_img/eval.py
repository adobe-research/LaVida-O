import sys
import os

from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs



if __name__ == "__main__":
    os.environ['DEBUG_FIX_PADDING'] = '1'
    os.environ['NOT_ALWASY_DO_2DPOOL'] = '1'
# set import dir to be the parent folder
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(parent)
# exit()
sys.path.append(parent)
import torch
from transformers import AutoTokenizer

import os


from transformers import CLIPTextModelWithProjection,CLIPVisionModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast,CLIPImageProcessor
import argparse
import pandas as pd
from argparse import ArgumentParser
import accelerate
from PIL import Image, ImageOps
import numpy as np
from eval_img.geneval_utils import SegmentationFeedback,QwenFeedback

from eval_img.dpg_utils import DPGFeedback,prepare_dpg_data
from llava.eval.predict_t2i_edit import build_model,text_to_image,create_plan,get_feedback
class DummyFeedback:
    
    def __init__(self,device,correct=1):
        self.correct = correct
        pass
    
    def evaluate_image(self,image_path,metadata):
       return dict(
            correct=self.correct,
            text_feedback='',
            prompt=metadata['prompt'],
        )
       
    def to(self,*args, **kwargs):
        pass


class ReflectionFeedback:
    
    def __init__(self,model,tokenizer,image_processor):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def evaluate_image(self,image_path,metadata):
       feedback = get_feedback(self.model,self.tokenizer,self.image_processor,metadata['prompt'],image_path)
       ok = 'is correct' in feedback
       return dict(
            correct=ok,
            text_feedback=feedback,
            prompt=metadata['prompt'],
        )
       
    def to(self,*args, **kwargs):
        pass

def read_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


import glob

parser = argparse.ArgumentParser()
ckpts =''
base_model_path='legacy_not_used'
text_vae_path='legacy_not_used'

parser.add_argument('--ckpt',type=str,default=ckpts)
parser.add_argument('--output_path',type=str,default='outputs_geneval')
parser.add_argument('--cfg',type=float,default=4)
parser.add_argument('--name',type=str,default='none')
parser.add_argument('--base_model_path',type=str,default=base_model_path)
parser.add_argument('--text_vae_path',type=str,default=text_vae_path)
parser.add_argument('--shift',type=float,default=3.0)
###########
parser.add_argument('--alg_temp',type=float,default=1.0)
parser.add_argument('--schedule',type=str,default='shift')
parser.add_argument('--dynamic_temperature',type=bool,default=False)
parser.add_argument('--schedule_temp',type=str,default='linear')
parser.add_argument('--min_temperature',type=float,default=1)
parser.add_argument('--top_p',type=float,default=None)
parser.add_argument('--top_k',type=int,default=None)
parser.add_argument('--micro_cond',type=str,default='ORIGINAL WIDTH : 1024; ORIGINAL HEIGHT : 1024; TOP : 0; LEFT : 0; SCORE : 6.5')
########
parser.add_argument('--ema',action='store_true')
parser.add_argument('--res',type=int,default=1024)
parser.add_argument('-n','--num_samples',type=int,default=4)
parser.add_argument('-k','--n_reflection_tokens_per_img',type=int,default=64)
parser.add_argument('--cont',action='store_true')
parser.add_argument('--config',default='',type=str)
parser.add_argument('--drop_feedback',action='store_true')
parser.add_argument('--steps',type=int,default=20)
parser.add_argument('--no_reflection',action='store_true')
parser.add_argument('--cleanup',action='store_true')
parser.add_argument('--do_plan',action='store_true')
parser.add_argument('--prefix',type=str,default='')
parser.add_argument('--n_feedback',type=int,default=3)
parser.add_argument('--dataset',type=str,choices=['geneval','dpg','custom','wise','mjhq'])
parser.add_argument('--add_prefix_feedback',action='store_true')
parser.add_argument('--fmt',type=str,default='png')
parser.add_argument('--scaling',type=str,default='')
parser.add_argument('--vqvae',type=str,default='')

# vlm_path
parser.add_argument('--vlm_path',type=str,default='')
parser.add_argument('--debug',action='store_true')


parser.add_argument('-i','--annotation',default='geneval/prompts/evaluation_metadata.jsonl',type=str)

args = parser.parse_args()

def build_pipeline(accelerator,args):
    tokenizer, model, image_processor = build_model(args.ckpt)
    # if args.debug:
    #     breakpoint()
    if args.vqvae:
        from safetensors.torch import load_file
        vae = load_file(os.path.join(args.vqvae,'diffusion_pytorch_model.safetensors'),device='cpu')
        # vae_filtered = {k:v for k,v in vae.items() if 'decoder' in k}
        model.model.vqvae.load_state_dict(vae,strict=False)
    def _generate_one_image(prompt, **kwargs):
        # Generate one image using the text-to-image model
        token_map ={
            256:256,
            512:1024,
            1024:4096
        }
        print(args.res,token_map[args.res])
        extra_args = {}
        if args.prefix:
            prompt = args.prefix + prompt
        gen_dict = dict(tokenizer=tokenizer,sample_policy='multinomial', confidence_policy='mmada', guidance_scale=args.cfg, batch_size=1,
                        image_resolution=args.res, n_tokens=token_map[args.res], shift=args.shift,n_steps=args.steps,
                        schedule=args.schedule,
                        alg_temp=args.alg_temp,
                        dynamic_temperature=args.dynamic_temperature,
                        schedule_temp=args.schedule_temp,
                        min_temperature=args.min_temperature,
                        micro_cond=args.micro_cond,)
        if args.top_p is not None:
            gen_dict.update(dict(
                top_p=args.top_p,
                top_k=args.top_k
            ))
        if args.config:
            import yaml
            with open(args.config) as f:
                config_yaml = yaml.safe_load(f)
            gen_dict.update(config_yaml['config'])
        gen_dict.update(kwargs)
        if args.do_plan:
            plan = create_plan(model,tokenizer,prompt, gen_dict['micro_cond'])
            gen_dict['plan'] = plan
        with torch.no_grad():
            images = text_to_image(model, prompt, 
                            **gen_dict
                    )
        return images
    return model,_generate_one_image,tokenizer,image_processor


from matplotlib import pyplot as plt
from PIL import Image

def center_crop_and_resize(image_path, desired_size):
    # Open the image
    image = Image.open(image_path)
    
    # Get dimensions
    width, height = image.size
    
    # Calculate the size of the largest square
    new_side = min(width, height)
    
    # Calculate the cropping box
    left = (width - new_side) / 2
    top = (height - new_side) / 2
    right = (width + new_side) / 2
    bottom = (height + new_side) / 2
    
    # Crop the image to the largest square
    image = image.crop((left, top, right, bottom))
    
    # Resize the image to the desired size
    image = image.resize((desired_size, desired_size))
    
    return image

from transformers import CLIPImageProcessor,SiglipVisionModel,SiglipImageProcessor
def subsample_by_unique_reason(data_list, sample_size=3):
    """
    Selects up to `sample_size` samples with unique reasons from the given list.
    
    :param data_list: List of dictionaries with 'path' and 'reason' keys
    :param sample_size: Number of unique samples to select (default is 3)
    :return: List of selected dictionaries
    """
    reason_map = {}
    for item in data_list:
        reason = item['reason']
        if reason not in reason_map:
            reason_map[reason] = item
    
    unique_samples = np.array(list(reason_map.values()))
    return list(np.random.choice(unique_samples, min(sample_size, len(unique_samples)), replace=False))


timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60*60*2))
accelerator = accelerate.Accelerator(kwargs_handlers=[timeout_kwargs])

if accelerator.num_processes > 1:
    torch.cuda.set_device(accelerator.device)
else:
    torch.cuda.set_device(0)
pipeline,generate_one_image,tokenizer,image_processor = build_pipeline(accelerator,args)
local_rank = accelerator.process_index
world_size = accelerator.num_processes
head = args.dataset
if args.scaling == 'reflection':
    head = f'{head}_reflection_n_{args.num_samples}'
elif args.scaling == 'self_verify':
    head = f'{head}_self_verify_n_{args.num_samples}'

ROOT=os.path.join(args.output_path,f'{head}_{args.name}_cfg_{args.cfg}_shift_{args.shift}_steps_{args.steps}_nsamples_{args.num_samples}_px_{args.res}_do_plan_{args.do_plan}/')
if accelerator.is_main_process:
    if os.path.exists(ROOT) and args.cleanup:
        import shutil
        shutil.rmtree(ROOT)
accelerator.wait_for_everyone()

os.makedirs(ROOT,exist_ok=True)


if args.dataset == 'geneval':
    verifier = SegmentationFeedback(accelerator.device)
elif args.dataset == 'dpg':
    verifier = DPGFeedback(device=f'cuda:{local_rank}')
else: # custom
    verifier = DummyFeedback(device=f'cuda:{local_rank}')

vlm_path = args.vlm_path
feedback = DummyFeedback('cuda',0)#QwenFeedback('cuda',vlm_path,greedy=False)
if args.scaling in ['reflection','self_verify'] :
    feedback = ReflectionFeedback(pipeline,tokenizer,image_processor)


import json

if args.dataset == 'geneval':
    with open(args.annotation) as fp:
        metadatas = [json.loads(line) for line in fp]
elif args.dataset == 'custom':
    metadatas = pd.read_csv(args.annotation).to_dict(orient='records')
elif args.dataset == 'mjhq':
    with open(args.annotation) as fp:
        data = json.loads(fp.read())
    metadatas = []
    for k,v in data.items():
        v['key'] = k
        metadatas.append(v)
elif args.dataset == 'wise':
    metadatas = pd.read_csv(args.annotation).to_dict(orient='records')
else:
    metadatas = prepare_dpg_data(csv_file=os.path.join(args.annotation,'dpg_bench.csv'),
                                 prompt_path=os.path.join(args.annotation,'prompts')
                                 )
n = len(metadatas)
n_prompt_per_rank = n // world_size + 1
start = local_rank * n_prompt_per_rank 
all_evaluations = []

acc = 0
total = 0

indices = np.arange(len(metadatas))
rng = np.random.default_rng(420)
shuffled_indices = rng.permutation(indices)
print(shuffled_indices[:100])


# reflection_transformer = pipeline.transformer.cpu()
RELFECTION_PROMPT = [
    "Given a user feed back for text-to-image generation, describe how you would fix the image given the feedback ",
    "Feedback: ",
]
for i in range(start,start+n_prompt_per_rank):
    if i >= len(metadatas):
        continue 
    i = shuffled_indices[i]
    outpath = os.path.join(ROOT,f"{i:0>5}") #

    metadata = metadatas[i]
    sample_path = os.path.join(outpath, "samples")

    try:
        os.makedirs(sample_path, exist_ok=True)
    except:
        pass

    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
        json.dump(metadata, fp)
    #     continue
    if i >= len(metadatas):
        continue 
    
    prompt = metadata['prompt']

    all_feedbacks = []

    print(f"Prompt:{prompt}")
    for j in range(args.num_samples):
        torch.cuda.empty_cache()
        pipeline.to('cuda')
        outpath_file = os.path.join(sample_path, f"{j:05}.{args.fmt}")
        outpath_file_json = outpath_file+'.feedback.json'
        if args.cont and os.path.exists(outpath_file_json):
            load_results = read_json(outpath_file_json)
            if type(load_results) == list:
                feedback_results = load_results[-1]
            else:
                feedback_results = load_results
            if feedback_results['correct_verifier']:
                print(f"{j}: OK (Replayed)")
                acc += 1
                break
            else:
                feedback_payload = dict(
                    path=outpath_file,
                    reason=feedback_results['text_feedback']
                )
                print(f"{j}: {feedback_results['text_feedback']} (Replayed)")
                all_feedbacks.append(feedback_payload)
                all_evaluations.append(feedback_payload)
            continue
        if len(all_feedbacks) == 0:
            img = generate_one_image(prompt)
        else:
            N_FEEDBACK = args.n_feedback
            if len(all_feedbacks) <= N_FEEDBACK:
                selected_feedbacks = all_feedbacks
            else:
                selected_feedbacks = subsample_by_unique_reason(all_feedbacks,N_FEEDBACK)
            feedback_imgs = list([ x['path'] for x in selected_feedbacks])
            feedback_texts = list([ x['reason'] for x in selected_feedbacks])
            if args.scaling == 'self_verify':
                feedback_imgs = feedback_texts = None
            img = generate_one_image(prompt,
                                     feedback_imgs=feedback_imgs,
                                     feedback_texts=feedback_texts,
                                     image_processor=image_processor)
        if args.dataset == 'wise':
            outpath_file = os.path.join(outpath, f"{metadata['prompt_id']}.png")
            
        img.save(outpath_file)
        if args.dataset == 'wise':
            continue
        feedback_results = feedback.evaluate_image(outpath_file,metadata)
        if verifier is not feedback:
            torch.cuda.empty_cache()
            outpath_file_abs = os.path.abspath(outpath_file)
            verifier.to('cuda')
            actual_results = verifier.evaluate_image(outpath_file_abs,metadata)
            verifier.to('cpu')
            actual_results['correct_verifier'] = feedback_results['correct']
            actual_results['text_feedback_verifier'] = feedback_results['text_feedback']
        else:
            actual_results = feedback_results
        actual_results['prompt'] = prompt
        actual_results['gen_idx_gt'] = j
        actual_results['filename'] = outpath_file
        all_evaluations.append(actual_results)
        with open(outpath_file_json,'w') as f:
            f.write(json.dumps(actual_results))
        if feedback_results['correct']:
            print(f"{j}: OK")
            acc += 1
            break
        else:
            feedback_payload = dict(
                path=outpath_file,
                reason=feedback_results['text_feedback']
            )
            print(f"{j}: {feedback_results['text_feedback']}")
            all_feedbacks.append(feedback_payload)
    total += 1
    
annotation_path = os.path.join(ROOT,f'annotations_rank_{local_rank}_of_{world_size}.json')
with open(annotation_path,'w') as f:
    f.write(json.dumps(all_evaluations))

config_path = os.path.join(ROOT,f'config_rank_{local_rank}_of_{world_size}.json')
with open(annotation_path,'w') as f:
    f.write(json.dumps(vars(args)))
        
accelerator.wait_for_everyone()
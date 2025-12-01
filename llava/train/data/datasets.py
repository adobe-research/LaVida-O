from llava.train.data.process_functions import PROCESS_FUNCTIONs
import transformers
from llava.utils import rank0_print
import yaml
import re
import json
from torch.utils.data import ConcatDataset,Dataset
import os
import pandas as pd
try:
    import piat
    # import internal library for loading s3 image
except:
    pass
from PIL import Image
import math
import random
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN,DEFAULT_IMAGE_GEN_TOKEN,DEFAULT_IMAGE_GEN_TOKEN_XTD
import torch
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, resize_and_center_crop,resize_and_random_crop,tokenizer_image_token,pad_to_square_and_resize,resize_and_crop
import time
from typing import Dict
import numpy as np
from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord
import copy
from llava.train.data.preprocess import *
from functools import partial
from llava.train.data.get_mask import diff_mask,comput_pixel_diff
def get_image_piat(strImagehash):
    strImagehash = strImagehash.replace('piat://','')
    try:
        try:
            img = piat.get_image({'strSource': '1024-pil-antialias'},strImagehash)
        except Exception as e:
            img = piat.get_image({'strSource': 'raw'},strImagehash)
        img = Image.fromarray(img).convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to load hash {strImagehash}")
    return img

def build_dataset_lazy(data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args,prepend_folder: bool = True):
    if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            list_data_dict = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    list_data_dict.extend(cur_data_dict)
            return LazySupervisedDataset(data_path, tokenizer, data_args, prepend_folder=prepend_folder, list_data=list_data_dict)
    elif data_path.endswith(".yaml"):
        prepend_folder = False # we load image root from yaml and ignore args
        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets")
            group_weights_dict = yaml_data.get("weights", {})
            group_bs_factor_dict = yaml_data.get("batch_sizes", {})
            # file should be in the format of:
            # datasets:
            #   - json_path: xxxx1.json
            #     sampling_strategy: first:1000
            #   - json_path: xxxx2.json
            #     sampling_strategy: end:3000
            #   - json_path: xxxx3.json
            #     sampling_strategy: random:999
            data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
            all_dataset_objs = []
            for dataset in datasets:
                json_path = dataset.get("json_path")
                sampling_strategy = dataset.get("sampling_strategy", "all")
                image_root = dataset.get("image_root",None)
                length_group = dataset.get("length_group", 'default')
                sampling_number = None
                preprocess_fn = None
                max_enc_images =  dataset.get("max_enc_images",None)
                if "preprocess_fn" in dataset:
                    preprocess_fn = PROCESS_FUNCTIONs.get(dataset["preprocess_fn"], None)
                    if preprocess_fn is None:
                        raise ValueError(f"Preprocess function {dataset['preprocess_fn']} not found in PROCESS_FUNCTIONs.")
                    if dataset.get('process_fn_kwargs') is not None:
                        preprocess_fn = partial(preprocess_fn,**dataset['process_fn_kwargs'])
                rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                if json_path.endswith(".jsonl"):
                    cur_data_dict = []
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            cur_data_dict.append(json.loads(line.strip()))
                elif json_path.endswith(".json"):
                    with open(json_path, "r") as json_file:
                        cur_data_dict = json.load(json_file)
                elif json_path.endswith(".parquet"):
                    df = pd.read_parquet(json_path,columns=dataset.get("columns",None))
                    if 'fltLaionAesthScore' in df.columns:
                        cutoff = dataset.get('aes_cutoff',5)
                        df = df[df.fltLaionAesthScore > cutoff]
                    # if 'longLLA_captions' in df.columns:
                    #     df = df[['longLLA_captions','strImagehash','']]
                    if 'min_size' in dataset:
                        assert 'intHeight' in df.columns and 'intWidth' in df.columns, "min_size requires intHeight and intWidth columns"
                        df = df[(df.intHeight >= dataset['min_size']) & (df.intWidth >= dataset['min_size'])]

                    if max_enc_images is not None:
                        image_lens = df[max_enc_images['column']].apply(len)
                        df = df[image_lens <=max_enc_images['value'] ]
                    df.reset_index(drop=True, inplace=True)
                    cur_data_dict = df
                else:
                    raise ValueError(f"Unsupported file type: {json_path}")

                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)

                # Apply the sampling strategy
                raw_length = len(cur_data_dict)
                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    cur_data_dict = cur_data_dict.sample(n=sampling_number, random_state=42)
                    cur_data_dict.reset_index(drop=True, inplace=True)
                    # random.shuffle(cur_data_dict)
                    # cur_data_dict = cur_data_dict[:sampling_number]
                if image_root is not None:
                    for sample in cur_data_dict:
                        if "image" in sample:
                            image_file = sample["image"]
                            if not os.path.isabs(image_file):
                                image_file = os.path.join(image_root, image_file)
                            sample["image"] = image_file
                duplicate_factor = 0
                if sampling_strategy == 'dup':
                    duplicate_factor = sampling_number
                    cur_data_dict = cur_data_dict.loc[cur_data_dict.index.repeat(duplicate_factor)]
                rank0_print(f"Loaded {len(cur_data_dict)} / {raw_length} samples from {json_path}")
                dataset_name = dataset.get('name','none')
                this_dataset = LazySupervisedDataset(
                    json_path, tokenizer, data_args, prepend_folder=prepend_folder, list_data=cur_data_dict, preprocess_fn=preprocess_fn,name=dataset_name
                )
                _ = this_dataset[0]  # Trigger the loading of the first sample for debugging
                all_dataset_objs.append((this_dataset,length_group))


            if data_args.group_by_random_length:
                all_dataset_objs = sorted(all_dataset_objs, key=lambda x: x[1]) 
                length_groups = {} # key -> int length
                for dataset, length_group in all_dataset_objs:
                    if length_group not in length_groups:
                        length_groups[length_group] = 0
                    length_groups[length_group] += len(dataset)

                group_names = []
                group_lengths = []
                group_weights = []
                group_bs_factor = []
                for dataset, length_group in all_dataset_objs:
                    if length_group not in group_names:
                        group_names.append(length_group)
                        group_lengths.append(length_groups[length_group])
                        group_weights.append(group_weights_dict.get(length_group, 100))
                        group_bs_factor.append(group_bs_factor_dict.get(length_group, 1))
                data_args.group_names = group_names
                data_args.group_lengths = group_lengths
                data_args.group_weights = group_weights
                data_args.group_bs_factor = group_bs_factor
            else:
                data_args.group_names = None
                data_args.group_lengths = None
                data_args.group_weights = None
                data_args.group_bs_factor = None


            all_dataset_objs = [x[0] for x in all_dataset_objs]

            return ConcatDataset(all_dataset_objs)
    else:
        data_args.dataset_paths = [data_path]
        rank0_print(f"Loading {data_path}")
        with open(data_path, "r") as file:
            cur_data_dict = json.load(file)
            rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
            return LazySupervisedDataset(data_path, tokenizer, data_args, prepend_folder=prepend_folder, list_data=cur_data_dict)

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args,prepend_folder: bool = True,list_data=None,preprocess_fn=None,name=None):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = list_data
        self.preprocess_fn = preprocess_fn
        self.prepend_folder = prepend_folder
        self.name = name
        # Handle multiple JSON files specified in the data_path


        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        # self.image_cache = {}

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list
    
    def process_image_gen(self, image_file, overwrite_image_aspect_ratio=None,image_cache=None,pad=False,crop_parms=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor_gen
        image = self.load_image_path(image_file,image_cache)
        if crop_parms is not None:
            #print (f"Cropping: {crop_parms}")
            assert self.data_args.image_gen_size == 1024 # hack
            image_dict = resize_and_crop(image,crop_parms[0])
            # image_dict = {
            #     "image":image,
            #     "micro_conds": [],
            #     "micro_conds_text": [],
            # }
        elif pad:
            image = pad_to_square_and_resize(image,self.data_args.image_gen_size)
            image_dict = {
                "image":image,
                "micro_conds": [],
                "micro_conds_text": [],
            }
        else:
            image_dict = resize_and_random_crop(image,self.data_args.image_gen_size)
        # processed = processor.preprocess(image)
        image_dict['image'] = processor.preprocess(image_dict['image']) # 1 C H W
        return image_dict

    def load_image_path(self,image_file,cache=None):
        if cache is None:
            cache = {}
        if image_file in cache:
            return cache[image_file]
        try:
            if image_file.startswith("s3://"):
                image_data = piat.s3_image(image_file) 
                image = Image.fromarray(image_data).convert("RGB")
            elif image_file.startswith("piat://"):
                image = get_image_piat(image_file)
            elif self.prepend_folder and not os.path.isabs(image_file):
                image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            else:
                image = Image.open(image_file).convert("RGB")
            cache[image_file] = image
            return image
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

    def process_image(self, image_file, overwrite_image_aspect_ratio=None,image_cache=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        image = self.load_image_path(image_file,image_cache)

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300
        num_base_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        image_cache = {}
        if isinstance(self.list_data_dict, pd.DataFrame):
            # If the data is a DataFrame, we need to convert it to a dict
            # This is useful for lazy loading from parquet files
            sources = self.list_data_dict.iloc[i].to_dict()
        else:
            sources = self.list_data_dict[i]
        if self.preprocess_fn is not None:
            sources = self.preprocess_fn(data=sources, tokenizer=self.tokenizer, has_image=self.data_args.is_multimodal)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        image_gen = None
        image_gen_enc = None
        micro_conds_texts = None
        if "image_gen_enc" in sources[0] and sources[0]["image_gen_enc"] is not None:
            image_file_enc = sources[0]["image_gen_enc"]
            if type(image_file_enc) is list:
                image_gen_enc_dict = [self.process_image_gen(f,image_cache=image_cache,pad=True) for f in image_file_enc]
            else:
                image_gen_enc_dict = [self.process_image_gen(image_file_enc,image_cache=image_cache,pad=True)]
            image_gen_enc = [x['image'] for x in image_gen_enc_dict]
            #micro_conds_texts = [x['micro_conds_text'] for x in image_gen_dict]


        if "image_gen" in sources[0] and sources[0]["image_gen"] is not None:
            image_file = sources[0]["image_gen"]
            do_pad = sources[0].get('pad_image_gen',False)
            crop_parms = sources[0].get('crop_parms',None)
            if type(image_file) is list:
                image_gen_dict = [self.process_image_gen(f,image_cache=image_cache,pad=do_pad,crop_parms=crop_parms) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad 
                # if len(image_file) > 1:
                #     image = [self.process_image(f, "pad") for f in image_file]
                #     image = [[im[0], im[1], "image"] for im in image]
            else:
                image_gen_dict = [self.process_image_gen(image_file,image_cache=image_cache,pad=do_pad,crop_parms=crop_parms)]
            image_gen = [x['image'] for x in image_gen_dict]
            #[x['micro_condition_text'] for x in image_gen_dict]
            micro_conds_texts = [x['micro_conds_text'] for x in image_gen_dict]
            if 'gen_score' in sources[0]:
                gen_score = sources[0]['gen_score']
                for _idx,entry in enumerate(micro_conds_texts):
                    entry.append(f"SCORE : {gen_score[_idx]:.3f}")
            if 'hpsv2' in sources[0]:
                hpsv2 = sources[0]['hpsv2']
                for _idx,entry in enumerate(micro_conds_texts):
                    entry.append(f"HPS : {hpsv2[_idx] * 10:.3f}")    
            micro_conds_texts = ['; '.join(x) for x in micro_conds_texts]
            
        source_clone_raw = copy.deepcopy(sources)
        image = None
        if "image" in sources[0]:
            image_file = sources[0]["image"]
            if image_file is None:
                image_file = []
            if type(image_file) is list:
                image = [self.process_image(f,image_cache=image_cache) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad 
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad",image_cache=image_cache) for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file,image_cache=image_cache)]
        if sources[0].get('pixel_diff',False) and image_gen is not None and image_gen_enc is not None:
            if  len(image_gen) == 1 and len(image_gen_enc) == 1:
                # pass
                input_arr = image_gen_enc[0] # 1 3 H W
                target_arr = image_gen[0]
                pixel_diff = comput_pixel_diff(input_arr,target_arr,0.15) # [0, 1]
                sources[0]['conversations'][0]['value'] = sources[0]['conversations'][0]['value'] + f'; PIXEL_DIFF : {pixel_diff:.3f} ;'
                # breakpoint()
                # mask = mask * (self.data_args.mm_edit_area_weight-1) + 1 # [1, 5]
                # token_weight = [mask]
        
        
        if "image" in sources[0] or 'image_gen' in sources[0]:
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args,
                                            micro_conds_texts=micro_conds_texts)
        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.data_args.video_folder
            video_file = os.path.join(video_folder, video_file)
            suffix = video_file.split(".")[-1]
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))

            try:
                if "shareVideoGPTV" in video_file:
                    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

                    # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
                    if self.data_args.force_sample:
                        num_frames_to_sample = self.data_args.frames_upbound
                    else:
                        num_frames_to_sample = 10

                    avg_fps = 2
                    
                    total_frames = len(frame_files)
                    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)


                    frame_time = [i/2 for i in sampled_indices]
                    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

                    video_time = total_frames / avg_fps

                    # Read and store the sampled frames
                    video = []
                    for idx in sampled_indices:
                        frame_path = frame_files[idx]
                        try:
                            with Image.open(frame_path) as img:
                                frame = img.convert("RGB")
                                video.append(frame)
                        except IOError:
                            print(f"Failed to read frame at path: {frame_path}")
                else:
                    video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)

                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                if self.data_args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                    sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                image = [(image, video[0].size, "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                # print(sources)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_file}")
                return self._get_item(i + 1)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ("image" in source_clone_raw[0]) or ("video" in source_clone_raw[0])
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in source_clone_raw[0] and image is not None and len(image) > 0:
            data_dict["image"] = image
        elif "video" in source_clone_raw[0]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = source_clone_raw[0].get("id", i)

        token_weight = None
        if 'reweight' in source_clone_raw[0]:
            if source_clone_raw[0].get('reweight',False) and len(image_gen) == 1 and len(image_gen_enc) == 1:
                # pass
                input_arr = image_gen_enc[0] # 1 3 H W
                target_arr = image_gen[0]
                mask = diff_mask(input_arr,target_arr) # [0, 1]
                mask = mask * (self.data_args.mm_edit_area_weight-1) + 1 # [1, 5]
                token_weight = [mask]
            else:
                token_weight = []
                for img in image_gen:
                    #_h = int(np.sqrt(self.data_args.num_gen_image_tokens))
                    # hack
                    _h = img.shape[-1] // 16
                    null_mask = torch.ones(1,1,_h,_h).float()
                    token_weight.append(null_mask)
        do_not_mask_text = source_clone_raw[0].get('do_not_mask_text',False)
        if image_gen is not None and len(image_gen) == 0:
                image_gen = None
        if image_gen_enc is not None and  len(image_gen_enc) == 0:
                image_gen_enc = None
        if token_weight is not None and len(token_weight) == 0:
                token_weight = None
        data_dict['image_gen'] = image_gen
        data_dict['image_gen_enc'] = image_gen_enc
        data_dict['image_gen_weight'] = token_weight
        data_dict['name'] = self.name
        data_dict['do_not_mask_text'] = do_not_mask_text

        return data_dict

if __name__ == "__main__":
    from transformers import AutoTokenizer
    import sys
    from llava.model.language_model.llava_llada import *
    from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
    tokenizer = AutoTokenizer.from_pretrained('/sensei-fs-3/users/shufanl/LaViDa/lavida-llada-v1.0-instruct')
    vqvae_config = {
            "block_out_channels": [
            128,
            256,
            256,
            512,
            768
            ]
    }
    class ImP:
        crop_size = dict(height=384,width=384)
    
    data = 'scripts/train/t2i-debug.yaml'
    data = 'scripts/train/und_10m.yaml'
    data = sys.argv[1]
    class DataArgs:
        dataset_paths = data
        is_multimodal = True
        image_aspect_ratio = 'anyres'
        num_gen_image_tokens_enc_ds = 4
        image_grid_pinpoints = "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]"
        image_folder  = ''
        group_by_random_length = True
        mm_edit_area_weight= 4
        num_gen_image_tokens =  1024
        vae_scale_factor = 2 ** (len(vqvae_config['block_out_channels']) - 1)
        image_processor_gen = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False)
        mm_use_im_start_end = False
        image_processor = SigLipImageProcessor()
        image_gen_size = 1024
    data_args = DataArgs()
    dataset = build_dataset_lazy(data_args.dataset_paths,tokenizer,data_args,)
    print(dataset[1])
    print(dataset[-1])
    # dataset._get_item[122]
    # breakpoint()

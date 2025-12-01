import collections
import json
import math
import subprocess
import copy
from dataclasses import dataclass

from typing import Any, Dict, Literal, Union, Optional, List, Tuple, Callable, Sequence


from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.data.data_collator import DataCollator
from transformers.trainer_utils import (
    # EvalPrediction, 
    # EvalLoopOutput,
    has_length,
)
from PIL import Image
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
import torch.nn as nn
import torch.distributed as dist
import torch
import PIL
import pandas as pd
from tqdm import tqdm
import wandb
from llava.eval.predict_grounding import RECDataset,pairwise_iou,predict_grounding

from llava.train.llava_trainer import LLaVATrainer
# from llava.train.config import TrainingArguments
from llava.utils import rank0_print
from llava.mm_utils import (
    process_images,
    tokenizer_image_token
)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.conversation import conv_templates
from trl.models.utils import unwrap_model_for_generation
from llava.eval.predict_t2i_edit import text_to_image
import yaml
# from loguru import logger as eval_logger
import os
EVAL_CONV_TEMPLATE = os.environ.get('EVAL_CONV_TEMPLATE','llada')
DEBUG_PRINT_OUTPUT = os.environ.get('DEBUG_PRINT_OUTPUT',False)
try:
    from lmms_eval.tasks import (
        TaskManager,
        get_task_dict
    )
    task_manager = TaskManager()
except:
    rank0_print("Please install lmms_eval to use evaluation")
    # raise ModuleNotFoundError


VAL_PROMPTS_T2I = [
    "A cinematic wide shot of a lone astronaut standing on a desolate, alien planet, bathed in the glow of a binary sunset. Dust swirls around their boots. Highly detailed, 8K, sci-fi art, dramatic lighting.",
    "A mythical creature, a griffin with iridescent feathers, soaring majestically over a fantastical, mist-shrouded mountain range at dawn. Epic fantasy art, golden hour lighting, dynamic pose.",
    "Steampunk city skyline at night, with intricate clockwork buildings, glowing gears, and airships traversing the sky. Ornate, detailed, victorian, futuristic, volumetric lighting.",
    "A cyberpunk geisha with glowing neon tattoos and advanced augmented reality glasses, standing in a rain-slicked Tokyo alleyway at night. Blade Runner aesthetic, vibrant neons, detailed cybernetics, reflective surfaces.",
    "Highly stylized, whimsical illustration of a fox wearing a tiny monocle and top hat, reading a miniature book in a cozy, mushroom-filled forest glade. Children's book art, gentle colors, charming, adorable.",
    "An apple to the left of a tv."
]

VAL_PROMPTS_EDIT = [
    ('/sensei-fs-3/users/shufanl/LaViDa/images/temple.png', 'make the time midnight'),
    ('/sensei-fs-3/users/shufanl/LaViDa/images/temple.png', 'add a fox to the scene'),
    ('/sensei-fs-3/users/shufanl/LaViDa/images/dog.png', 'make the dog color red'),
    ('/sensei-fs-3/users/shufanl/LaViDa/images/dog.png', 'replace the dog with a bird'),
]
class LMMsEvalDataset(Dataset):
    def __init__(
        self, 
        hf_dataset, 
        task_obj, 
        model_config,
        image_processor,
        conv_template,
        task_type: Literal["loglikelihood", "generate_until"],
        tokenizer,
        limit=-1,
        ) -> None:
        super().__init__()
        self.hf_dataset = hf_dataset
        self.task_obj = task_obj
        self.model_config = model_config
        self.image_processor = image_processor
        self.conv_template = conv_template
        self.task_type = task_type
        self.generation_kwargs = task_obj.config.generation_kwargs
        self.tokenizer = tokenizer
        self.limit = limit
    
    def __getitem__(self, index):
        visual = self.task_obj.doc_to_visual(self.hf_dataset[index])
        context = self.task_obj.doc_to_text(self.hf_dataset[index])
        if visual is None or visual == []:
            visual = None
            task_type = "text"
            image_tensor = None
        else:
            if type(visual[0]) == PIL.Image.Image:
                image_tensor = process_images(visual, self.image_processor, self.model_config)
                if type(image_tensor) is list:
                    image_tensor = [_image for _image in image_tensor]
                else:
                    image_tensor = image_tensor

                task_type = "image"
        
        if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
            placeholder_count = len(visual) if isinstance(visual, list) else 1
            image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
            image_tokens = " ".join(image_tokens)
            prompts_input = image_tokens + "\n" + context
        else:
            prompts_input = context
        
        if "llama_3" in self.conv_template or 'llada' in self.conv_template:
            conv = copy.deepcopy(conv_templates[self.conv_template])
        else:
            conv = conv_templates[self.conv_template].copy()

        conv.append_message(conv.roles[0], prompts_input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        if type(self.task_obj.doc_to_target) == str:
            target = self.task_obj.doc_to_target
        else:
            target =self.task_obj.doc_to_target(self.hf_dataset[index])
        
        image_sizes = [visual[idx].size for idx in range(len(visual))]
        
        if self.task_type == "generate_until":
            return {
                "input_ids" : input_ids,
                "modalities" : ["image"] if task_type == "image" else ["text"],
                "images" : image_tensor,
                "image_sizes" : image_sizes,
                "index" : index,
                "prompt":prompt,
            }
        elif self.task_type == "loglikelihood":
            # Because caption tasts such as coco return a
            # list of answer, we pick the first one
            if isinstance(target, list):
                target = target[0]
            conv.messages[-1][1] = target
            full_prompt = conv.get_prompt()
            full_input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            labels = full_input_ids.clone()
            labels[ : input_ids.shape[0]] = -100
            return {
                "input_ids" : full_input_ids,
                "modalities" : ["image"] if task_type == "image" else ["text"],
                "images" : image_tensor,
                "image_sizes" : image_sizes,
                "index" : index,
                "labels" : labels,
            }
        else:
            raise ValueError(f"Task type : {self.task_type} is not Supported, please choose between generate_until or loglikelihood")

    
    def __len__(self) -> int:
        if self.limit > 0:
            return self.limit
        return len(self.hf_dataset)
    
@dataclass
class DataCollatorForEvaluationDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizerBase

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, modalities, images, image_sizes, index,prompt = tuple([instance[key] for instance in instances] for key in ("input_ids", "modalities", "images", "image_sizes", "index","prompt"))
            
        labels = []
        for instance in instances:
            if "labels" in instance:
                labels.append(instance["labels"])
        if len(labels) == 0:
            labels = None
        
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]

        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0  # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if labels is not None:
            labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
            labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids, 
            labels=labels.long() if labels is not None and labels.dtype == torch.int32 else labels, 
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
            modalities=modalities,
            image_sizes=image_sizes,
            index=index,
            prompt=prompt,
        )

def mean(arr):
    return sum(arr) / len(arr)

class LLaVAEvalTrainer(LLaVATrainer):
    def __init__(
        self, 
        *args,
        
        **kwargs
        # model: Union[PreTrainedModel, nn.Module] = None, 
        # args: TrainingArguments = None, 
        # data_collator: Optional[DataCollator] = None, 
        # train_dataset: Optional[Union[Dataset, IterableDataset]] = None, 
        # eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None, 
        # tokenizer: Optional[PreTrainedTokenizerBase] = None, 
        # model_init: Optional[Callable[[], PreTrainedModel]] = None, 
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None, 
        # callbacks: Optional[List[TrainerCallback]] = None, 
        # optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None), 
        # preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        # conv_template: str = "qwen_1_5",
        ):
        super().__init__(*args,**kwargs)
        # generate_tasks = self.args.lmms_eval_generate_tasks.split(",")
        if self.args.lmms_eval_generate_tasks:
            generate_tasks = self.args.lmms_eval_generate_tasks.split(",")
            self.generate_task_dict = get_task_dict(generate_tasks, task_manager)
            self.eval_data_collator = DataCollatorForEvaluationDataset(self.tokenizer)
        else:
            self.generate_task_dict = {}
        # ppl_tasks = self.args.lmms_eval_ppl_tasks.split(",")
        # self.ppl_task_dict = get_task_dict(ppl_tasks, task_manager)
        self.ppl_task_dict = {}
        #self.extra_tasks = getattr(self.args, "lmms_eval_extra_tasks", None)
        if hasattr(self.args, "lmms_eval_extra_tasks") and self.args.lmms_eval_extra_tasks:
            self.lmms_eval_extra_tasks = self.args.lmms_eval_extra_tasks.split(",")
        else:
            self.lmms_eval_extra_tasks = []
        self.model_config = self.model.config
        self.image_processor = self.model.get_vision_tower().image_processor
        self.conv_template = EVAL_CONV_TEMPLATE
        

    def evaluate(
        self, 
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None, 
        ignore_keys: Optional[List[str]] = None, 
        metric_key_prefix: str = "eval"
        ) -> Dict[str, float]:

        
        log_dict = {}

        for task in self.lmms_eval_extra_tasks:
            if task == 'refcoco':
                refcoco_results = self.refcoco_eval()
                if wandb.run is not None:
                    wandb.log(refcoco_results)
            if task == 'image_edit':
                images, config_str = self.text_to_image_edit(VAL_PROMPTS_EDIT)
                if wandb.run is not None:
                    payload = {}
                    payload['val/edit_images'] = [
                                    wandb.Image(image, caption=f"{i}:{VAL_PROMPTS_EDIT[i]} {config_str if i==0 else ''}") for i, image in enumerate(images)
                    ]
                    wandb.log(
                                payload
                    )
        torch.cuda.empty_cache()
        # t2i
        if self.args.t2i_eval:
            images, config_str = self.text_to_image(VAL_PROMPTS_T2I)
            if wandb.run is not None:
                payload = {}
                payload['val/images'] = [
                                wandb.Image(image, caption=f"{i}:{VAL_PROMPTS_T2I[i]} {config_str if i==0 else ''}") for i, image in enumerate(images)
                ]
                wandb.log(
                            payload
                )


            
        if self.state.global_step % (self.args.eval_steps * 2) != 0:
            return
        for task_name, task_obj in self.generate_task_dict.items():
            eval_dataset = LMMsEvalDataset(
                task_obj.test_docs(),
                task_obj,
                self.model_config,
                self.image_processor,
                self.conv_template,
                "generate_until",
                self.tokenizer,
            )

            eval_dataloader = self.get_lmms_eval_dataloader(
                eval_dataset,
                self.eval_data_collator
            )

            resps, correspond_index = self.generate_until_loop(
                eval_dataloader,
                description=task_obj.task_name,
            )
            # breakpoint()
            processed_results = self.process_results(
                resps,
                correspond_index,
                task_obj
            )

            # Because the resps are scattered in different ranks
            # We gather all the processed results and then merged
            all_processed_results = [None for _ in range(self.args.world_size)]
            dist.all_gather_object(all_processed_results, processed_results)

            
            merged_processed_results = collections.defaultdict(list)
            for processed_result in all_processed_results:
                for metric_name, data_dict in processed_result.items():
                    merged_processed_results[metric_name].extend(data_dict)


            if self.accelerator.is_main_process:
                for metric_name, processed_result in merged_processed_results.items():
                    aggregation_list = task_obj.aggregation()
                    # Okay, to be honest, other tasks might also suffer from this, 
                    # but mme strictly follows pair evaluation so I kind of hard code this handle logic in this way. 
                    # data loader might contain duplicate tasks when preparing. 
                    # I am just keep it this way for now, 
                    # since it is just an inofficial evaluation during middle training. 
                    # At last, recommend you to use lmms_eval for a wholistic evaluation after the training ! :D
                    if task_name == "mme":
                        processed_result = self.handle_mme_duplicate_result(processed_result)
                    score = self.aggregation(aggregation_list, metric_name, processed_result)
                    log_dict[f"{task_name}/{metric_name}"] = score
            self.accelerator.wait_for_everyone()

        for task_name, task_obj in self.ppl_task_dict.items():
            eval_dataset = LMMsEvalDataset(
                task_obj.test_docs(),
                task_obj,
                self.model_config,
                self.image_processor,
                self.conv_template,
                "loglikelihood",
                self.tokenizer,
            )

            eval_dataloader = self.get_lmms_eval_dataloader(
                eval_dataset,
                self.eval_data_collator
            )

            losses = self.loglikelihood_loop(
                eval_dataloader,
                description=task_obj.task_name
            )

            all_losses = [None for _ in range(self.args.world_size)]
            dist.all_gather_object(all_losses, losses)
            merged_losses = []
            for losses in all_losses:
                merged_losses.extend(losses)

            if self.accelerator.is_main_process:
                ppl = math.exp(-mean(merged_losses))
                log_dict[f"{task_name}/ppl"] = ppl
            
            self.accelerator.wait_for_everyone()


        self.log(log_dict)
        torch.cuda.empty_cache()

        return log_dict

    def get_lmms_eval_dataloader(
        self, 
        eval_dataset: Optional[Union[str, Dataset]] = None,
        data_collator = None,
        ) -> DataLoader:
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        return self.accelerator.prepare_data_loader(eval_dataloader)
    
    def text_to_image(
        self,
        prompts: List[str],
        overwrite = None,
    ):
        args = self.args
        batch_size =1
        guidance_scale = 3
        n_steps = 20
        shift=5
        schedule='shift'
        confidence_policy='mmada'
        alg_temp=1
        dynamic_temperature=False
        min_temperature=1
        image_resolution = self.args.image_gen_size
        n_tokens_map = {
            256: 256,
            512: 1024,
            1024: 4096
        }
        extra_kwargs = {}
        if image_resolution == 1024:
            guidance_scale = 4
            n_steps = 20
            shift = 5
            alg_temp = 5
            schedule = 'shift'
            dynamic_temperature = True
            min_temperature = 1
            try:
                with open('/sensei-fs-3/users/shufanl/LaViDa/llava/eval/1024_eval.yaml', "r") as file:
                    extra_kwargs = yaml.safe_load(file)['config']
                print(f"Loaded YAML config: {extra_kwargs}")
            except Exception as e:
                extra_kwargs = {}
                print("Failed to load YAML config, using default values.")
                print(e)
                print(e.__traceback__)
            # guidance_scale = yaml_data.get("guidance_scale", 4)
            # n_steps = yaml_data.get("n_steps", 20)
            # shift = yaml_data.get("shift", 5)
            # alg_temp = yaml_data.get("alg_temp", 5)
            # schedule = yaml_data.get("schedule", "shift")
            # dynamic_temperature = yaml_data.get("dynamic_temperature", True)
            # min_temperature = yaml_data.get("min_temperature", 1)
            # extra_kwargs = yaml_data.get("extra_kwargs", {})
        if image_resolution == 256:
            try:
                with open('/sensei-fs-3/users/shufanl/LaViDa/llava/eval/256_eval.yaml', "r") as file:
                    extra_kwargs = yaml.safe_load(file)['config']
                print(f"Loaded YAML config: {extra_kwargs}")
            except Exception as e:
                extra_kwargs = {}
                print("Failed to load YAML config, using default values.")
                print(e)
                print(e.__traceback__)
        config_str = f"guidance_scale={guidance_scale},n_steps={n_steps},shift={shift},alg_temp={alg_temp},dynamic_temperature={dynamic_temperature},min_temperature={min_temperature},schedule={schedule},extra_kwargs={extra_kwargs}"
        all_images = []
        self.model.eval()
        gen_dict = dict(tokenizer=self.tokenizer,
                        sample_policy='multinomial',
                        confidence_policy='halton',
                        guidance_scale=guidance_scale,
                        n_steps=n_steps,
                        batch_size=1,
                        image_resolution=image_resolution,
                        n_tokens=n_tokens_map[image_resolution],
                        shift=shift,
                        schedule=schedule,
                        alg_temp=alg_temp,
                        dynamic_temperature=dynamic_temperature,
                        min_temperature=min_temperature,
                        )
        gen_dict.update(extra_kwargs)
        if overwrite is not None:
            gen_dict.update(overwrite)
        with torch.inference_mode():
             with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model: 
                   for idx,prompt in enumerate(prompts):
                        img = text_to_image_edit(unwrapped_model,
                                            prompt,**gen_dict
                                                    )
                        all_images.append(img)
        return all_images,config_str
    

    def text_to_image_edit(
        self,
        prompts: List[str],
    ):
        torch.cuda.empty_cache()
        args = self.args
        batch_size =1
        guidance_scale = 3
        n_steps = 20
        shift=5
        schedule='shift'
        confidence_policy='mmada'
        alg_temp=1
        dynamic_temperature=False
        min_temperature=1
        image_resolution = self.args.image_gen_size
        n_tokens_map = {
            256: 256,
            512: 1024,
            1024: 4096
        }
        extra_kwargs = {}
        if image_resolution == 1024:
            guidance_scale = 4
            n_steps = 20
            shift = 5
            alg_temp = 5
            schedule = 'shift'
            dynamic_temperature = True
            min_temperature = 1
            try:
                with open('/sensei-fs-3/users/shufanl/LaViDa/llava/eval/1024_eval_edit.yaml', "r") as file:
                    extra_kwargs = yaml.safe_load(file)['config']
                print(f"Loaded YAML config: {extra_kwargs}")
            except Exception as e:
                extra_kwargs = {}
                print("Failed to load YAML config, using default values.")
                print(e)
                print(e.__traceback__)
            # guidance_scale = yaml_data.get("guidance_scale", 4)
            # n_steps = yaml_data.get("n_steps", 20)
            # shift = yaml_data.get("shift", 5)
            # alg_temp = yaml_data.get("alg_temp", 5)
            # schedule = yaml_data.get("schedule", "shift")
            # dynamic_temperature = yaml_data.get("dynamic_temperature", True)
            # min_temperature = yaml_data.get("min_temperature", 1)
            # extra_kwargs = yaml_data.get("extra_kwargs", {})
        if image_resolution == 256:
            try:
                with open('/sensei-fs-3/users/shufanl/LaViDa/llava/eval/256_eval.yaml', "r") as file:
                    extra_kwargs = yaml.safe_load(file)['config']
                print(f"Loaded YAML config: {extra_kwargs}")
            except Exception as e:
                extra_kwargs = {}
                print("Failed to load YAML config, using default values.")
                print(e)
                print(e.__traceback__)
        config_str = f"guidance_scale={guidance_scale},n_steps={n_steps},shift={shift},alg_temp={alg_temp},dynamic_temperature={dynamic_temperature},min_temperature={min_temperature},schedule={schedule},extra_kwargs={extra_kwargs}"
        all_images = []
        self.model.eval()
        gen_dict = dict(tokenizer=self.tokenizer,
                        sample_policy='multinomial',
                        confidence_policy='halton',
                        guidance_scale=guidance_scale,
                        n_steps=n_steps,
                        batch_size=1,
                        image_resolution=image_resolution,
                        n_tokens=n_tokens_map[image_resolution],
                        shift=shift,
                        schedule=schedule,
                        alg_temp=alg_temp,
                        dynamic_temperature=dynamic_temperature,
                        min_temperature=min_temperature,
                        )
        gen_dict.update(extra_kwargs)
        with torch.inference_mode():
             with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model: 
                   for idx,(src_img,prompt) in enumerate(prompts):
                        edit_image = Image.open(src_img)
                        img = text_to_image_edit(unwrapped_model,
                                            prompt,edit_image=edit_image,
                                            image_processor=self.image_processor,
                                            **gen_dict
                                                    )
                        all_images.append(img)
        return all_images,config_str

    def generate_until_loop(
        self,
        dataloader: DataLoader,
        description: str,
    ):
        self.model.eval()
        model = self.unwrap_model_for_inference(dataloader)
        args = self.args
        batch_size = self.args.eval_batch_size
        num_examples = self.num_examples(dataloader)
        rank0_print(f"\n***** Running {description} *****")
        rank0_print(f"  Num examples = {num_examples}")
        rank0_print(f"  Batch size = {batch_size}")

        world_size = max(1, args.world_size)
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        resps = []
        correspond_index = []

        pbar = tqdm(total=len(dataloader), desc=description, disable=not self.accelerator.is_local_main_process)
        gen_kwargs = {}
        gen_kwargs["max_new_tokens"] = 16
        gen_kwargs['block_length'] = min(128,gen_kwargs["max_new_tokens"])
        gen_kwargs['prefix_lm']=True
        gen_kwargs['step_per_block'] = gen_kwargs['block_length']
        
        if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = False
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
          
        with torch.inference_mode():
         with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:      
          for step, inputs in enumerate(dataloader):
            # Because batch size is 1, so we unwrap the list from inside
            modalities = inputs.pop("modalities")[0]
            image_sizes = inputs.pop("image_sizes")[0]
            inputs["images"] = inputs["images"][0]
            inputs["images"] = inputs["images"].to(model.dtype)
            index = inputs.pop("index")
            prompt = inputs.pop("prompt")
                # model.generate(input_ids=inputs["input_ids"],images=inputs["images"],attention_mask=inputs["attention_mask"],modalities=modalities, image_sizes=image_sizes, pad_token_id=pad_token_ids)
                
                    # with open('/data1/jacklishufan/trainer.pt', 'wb') as f:
                    #     torch.save(unwrapped_model.state_dict(), f)
                    #     print('saved')
                    #     print(1/0)  
            cont = unwrapped_model.generate(
                inputs=inputs["input_ids"],
                images=inputs["images"],
                #attention_mask=inputs["attention_mask"],
                attention_mask=None,
                modalities=modalities, 
                image_sizes=image_sizes, 
                pad_token_id=pad_token_ids,
                use_cache=True,
                # temperature=0.0,
                # do_sample=False,
                **gen_kwargs
            )
            if hasattr(cont,'sequences'):
                cont = cont.sequences
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            text_outputs = [response.strip().lstrip('!') for response in text_outputs]
            resps.extend(text_outputs)
            # breakpoint()
            if DEBUG_PRINT_OUTPUT:
                print(f'\n--------Start of Sample {index}---------')
                print("Question: ",prompt[0])
                print("Answer: ",text_outputs)
                log_kwargs = dict(
                    **gen_kwargs,
                    image_sizes=image_sizes,
                )
                print("Answer: ",log_kwargs)
                print('--------End---------')

            correspond_index.extend(index)
            pbar.update(1)
        pbar.close()

        
        return resps, correspond_index

    def refcoco_eval(
        self,
    ):
        # self.model.eval()
        description = "refcoco (n=500)"
        refcoco_data_path =  '/mnt/localssd/und_data/grounding_refcoco-unc_val.parquet'
        refcoco_data = pd.read_parquet(refcoco_data_path)
        seed = 42
        refcoco_data = refcoco_data.sample(n=500, random_state=seed).reset_index(drop=True)
        rec_dataset = RECDataset(refcoco_data, self.tokenizer, self.image_processor)
        dataloader = torch.utils.data.DataLoader(rec_dataset, batch_size=1, shuffle=False)
        dataloader = self.accelerator.prepare_data_loader(dataloader)
        model = self.unwrap_model_for_inference(dataloader)
        args = self.args
        batch_size = self.args.eval_batch_size
        num_examples = self.num_examples(dataloader)
        rank0_print(f"\n***** Running {description} *****")
        rank0_print(f"  Num examples = {num_examples}")
        rank0_print(f"  Batch size = {batch_size}")

        world_size = max(1, args.world_size)
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        resps = []
        correspond_index = []

        pbar = tqdm(total=len(dataloader), desc=description, disable=not self.accelerator.is_local_main_process)
       
        correct = 0
        total = 0
        with torch.inference_mode():
         with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:      
          for step, batch in enumerate(dataloader):
            try:
                image = batch['image'][0]
                expression = batch['expression'][0]
                image = Image.fromarray(image.cpu().numpy()).convert("RGB")
                boxes = predict_grounding(unwrapped_model, self.image_processor, image, [expression],tokenizer=self.tokenizer, mode='refcoco')
                gt = batch['transformed_box']
                boxes = torch.tensor(boxes).to(self.accelerator.device)
                box_iou = pairwise_iou(boxes, gt).item()
            except:
                box_iou = 0.0
            # all_ious.append(box_iou)
            if box_iou >= 0.5:
                correct += 1
            total += 1
            current_acc = correct / total if total > 0 else 0
            pbar.update(1)
        pbar.close()
        results = torch.tensor([correct,total], dtype=torch.float32).to(self.accelerator.device)
        results = self.accelerator.reduce(results, reduction = 'sum')
        correct, total = results[0].item(), results[1].item()
        acc = correct / (total + 1e-6)
        torch.cuda.empty_cache()
        print(f"Refcoco Acc: {acc:.4f} ({correct}/{total})")
        return {
            "val/refcoco_val_P@0.5": acc,
        }


    def loglikelihood_loop(
        self,
        dataloader: DataLoader,
        description: str,
    ) -> List[float]:
        model = self.unwrap_model_for_inference(dataloader)
        args = self.args
        batch_size = self.args.eval_batch_size
        num_examples = self.num_examples(dataloader)
        rank0_print(f"\n***** Running {description} *****")
        rank0_print(f"  Num examples = {num_examples}")
        rank0_print(f"  Batch size = {batch_size}")

        world_size = max(1, args.world_size)

        losses = []

        pbar = tqdm(total=len(dataloader), desc=description, disable=not self.accelerator.is_local_main_process)
        for step, inputs in enumerate(dataloader):
            # Because batch size is 1, so we unwrap the list from inside
            modalities = inputs.pop("modalities")[0]
            image_sizes = inputs.pop("image_sizes")[0]
            inputs["images"] = inputs["images"][0]
            inputs["images"] = inputs["images"].to(model.dtype)
            index = inputs.pop("index")
            with torch.no_grad():
                output = model(
                    input_ids=inputs["input_ids"],
                    images=inputs["images"],
                    attention_mask=inputs["attention_mask"],
                    modalities=modalities, 
                    image_sizes=image_sizes, 
                    labels=inputs["labels"],
                    )
                loss = output["loss"]
                losses.append(float(loss.item()))
            pbar.update(1)
        return losses

    def process_results(
        self,
        resps: List[str],
        correspond_index: List[int],
        task_obj,
        ) -> Dict[str, List[Dict[str, Any]]]:
        # We retrive our test docs first
        # Notice that here is no image, so probably you
        # can't evaluate llava_wilder etc. :D
        test_docs_no_image = task_obj.dataset_no_image[task_obj.config.test_split]
        processed_results = collections.defaultdict(list)
        pbar = tqdm(total=len(resps), desc="Processed eval results", disable= not self.accelerator.is_main_process)
        for resp, index in zip(resps, correspond_index):
            doc = test_docs_no_image[index]
            result = [resp]
            data_dict = task_obj.process_results(doc, result)
            for metric_name, data in data_dict.items():
                processed_results[metric_name].append(data)
            pbar.update(1)
        pbar.close()
        
        return processed_results

    def aggregation(
        self,
        aggregation_list: List[Dict[str, Callable]],
        metric_name: str,
        results: Dict[str, List[Dict[str, Any]]],
    ) -> float:
        if metric_name == 'submission':
            return -1
        aggregation_fn = aggregation_list[metric_name]
        score = aggregation_fn(results)
        return score
    
    def handle_mme_duplicate_result(
        self,
        data_dict: List[Dict[str, Any]]
    ):
        exist_question_id = collections.defaultdict(int)
        fixed_data_dict = []
        # Each question id may contains at most 2 images
        for res in data_dict:
            question_id = res["question_id"]
            if exist_question_id[question_id] >= 2:
                continue
            else:
                fixed_data_dict.append(res)
                exist_question_id[question_id] += 1
        
        return fixed_data_dict

    def unwrap_model_for_inference(
        self,
        dataloader : DataLoader,
    ):
        args = self.args

        if not has_length(dataloader):
            raise ValueError("dataloader must implement a working __len__")

        # if eval is called w/o train, handle model prep here
        hf_deepspeed_config = self.accelerator.state.deepspeed_plugin.hf_ds_config

        # resume config update - some bits like `model` and `num_training_steps` only become available during train
        if not hf_deepspeed_config.is_zero3():
            pass
        elif self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        model.eval()
        return model
    

class LLaVAEvalTrainerDistill(LLaVAEvalTrainer):
    def __init__(
        self, 
        reference_model=None,
        *args,
        **kwargs
        # model: Union[PreTrainedModel, nn.Module] = None, 
        # args: TrainingArguments = None, 
        # data_collator: Optional[DataCollator] = None, 
        # train_dataset: Optional[Union[Dataset, IterableDataset]] = None, 
        # eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None, 
        # tokenizer: Optional[PreTrainedTokenizerBase] = None, 
        # model_init: Optional[Callable[[], PreTrainedModel]] = None, 
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None, 
        # callbacks: Optional[List[TrainerCallback]] = None, 
        # optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None), 
        # preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        # conv_template: str = "qwen_1_5",
        ):
        super().__init__(*args,**kwargs)
        self.reference_model = reference_model


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.state.global_step % 1000 in [0,1]:
            torch.cuda.empty_cache()
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs,policy=self.args.policy,policy_args=self.args.policy_args)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        extra_logs =   dict(
                    und_loss = outputs["und_loss"].item() if "und_loss" in outputs else 0.0,
                    gen_loss = outputs["gen_loss"].item() if    "gen_loss" in outputs else 0.0,
                    p_mask = outputs["p_mask"].mean().item() if    "p_mask" in outputs else 0.0,
            )
        if wandb.run is not None:
            wandb.log(
                extra_logs
            )
        #if self.state.global_step % 20 == 0 and 'new_input_ids' in outputs:
        if 1:
            #self.log_data(outputs)
            with torch.no_grad():
                predictions =  outputs['logits'].argmax(-1) 
                if MASK_AR_LOGGING:
                    predictions = torch.cat([predictions[:,-1:], predictions[:,1:]], dim=1)
                new_input_ids = outputs['new_input_ids'].clone()#.repeat(2,1)
                do_inv = outputs['do_inv']
                
                labels = inputs['labels'].clone()#.repeat(2,1)
                if do_inv:
                    labels = labels.repeat(2,1)
                final_masked_indices = outputs['final_masked_indices']
                acc_pred_input = predictions == new_input_ids
                non_padding = new_input_ids != self.tokenizer.pad_token_id
                acc =  acc_pred_input[final_masked_indices].float().mean().item()
                acc_non_padding =  acc_pred_input[(final_masked_indices & non_padding)].float().mean().item()
                # if self.state.global_step % 20 == 0:
                new_input_ids[new_input_ids<0] = self.tokenizer.pad_token_id
                x_t = new_input_ids.clone()
                x_t[final_masked_indices] = self.tokenizer.mask_token_id or 126336
                # dream has it in tokenizer
                # llada does not
                # llada mask_token_id is 126336
                x_t = self.tokenizer.batch_decode(x_t)
                
                x_t = [x.replace('<|endoftext|>','') for x in x_t]
                x_t = [x.replace('<|mdm_mask|>','[*]') for x in x_t]
                x_t = [x.replace('<|mask|>','[*]') for x in x_t]
                x_t = [x.replace('<|reserved_token_5|>','') for x in x_t]
                x_0 = new_input_ids.clone()
                x_0[final_masked_indices] = predictions[final_masked_indices]
                x_0 = self.tokenizer.batch_decode(x_0)
                x_0 = [x.replace('<|endoftext|>','') for x in x_0]

                
                labels[labels<0] = self.tokenizer.pad_token_id
                x_0_gt = self.tokenizer.batch_decode(labels)
                x_0_gt = [x.replace('<|endoftext|>','') for x in x_0_gt]
                if 'new_token_mask_dup' in outputs:
                    with torch.no_grad():
                        gen_x_0_pred = outputs['gen_x_0_pred'].clone()
                        gen_x_mask = outputs['gen_x_mask']
                        gen_x0_gt = outputs['gen_x0_gt']
                        gen_x_0_pred[~gen_x_mask] = gen_x0_gt[~gen_x_mask]
                        gen_x0_gt_masked = gen_x0_gt.clone()
                        gen_x0_gt_masked[gen_x_mask] = 0
                        images_to_decode = torch.stack([gen_x_0_pred[0],gen_x0_gt[0],gen_x0_gt_masked[0]])
                        decoded_images = self.model.decode_image_gen(images_to_decode,self.args.image_gen_size,self.args.image_gen_size)

                html_table = """
                <table border="1">
                    <tr><th>x_t</th><th>x_0</th><th>label</th></tr>
                    {}
                </table>
                """.format("\n".join(f"<tr><td>{html.escape(t)}</td><td>{html.escape(o)}</td> <td>{html.escape(g)}</td></tr>" for t, o,g in zip(x_t, x_0,x_0_gt)))
                if wandb.run is not None:
                    payload = {
                            "train/acc_mask":acc,
                            "train/acc_mask_non_padding":acc_non_padding,
                            "html_table": wandb.Html(html_table)
                    }
                    if 'new_token_mask_dup' in outputs:
                        anno = ['x_0_pred','x_0','x_t']
                        payload['gen_images'] = [
                            wandb.Image(image, caption=f"{anno[i]}") for i, image in enumerate(decoded_images)
                        ]
                    if 'skip_batch' in outputs:
                        payload['skip_batch'] = outputs['skip_batch']
                    wandb.log(
                        payload
                    )
        if self.state.global_step % 500 == 0:
            torch.cuda.empty_cache() # cleanup memory
        #self.tokenizer.batch_decode(outputs.logits.argmax(-1))[0]
        # print()
        # sefl.tokenizer.batch_decode(outputs["logits"], skip_special_tokens=True)

        return (loss, outputs) if return_outputs else loss
  
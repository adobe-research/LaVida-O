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


print("Importing necessary libraries...")
import os
if __name__ == "__main__":
    os.environ['DEBUG_FIX_PADDING'] = '1'
    os.environ['NOT_ALWASY_DO_2DPOOL'] = '1'
from llava.model.builder import load_pretrained_model

from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX,SKIP_DOWN_SAMPLE
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
from llava.model.language_model.llada.generate import generate as llada_generate,cosine_schedule_2,logit_normal_schedule,exp_schedule
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
from llava.mm_utils import pad_to_square_and_resize
from llava.model.utils import maybe_truncate_last_dim,pad_along_last_dim
from llava.eval.pmj import stratified_random
from llava.mm_utils import resize_and_center_crop
def build_model(
    pretrained = "",
    model_name = "llava_llada",
    device = "cuda",
    ):
    print("Loading model...")
    vision_kwargs = dict(
        mm_vision_tower="google/siglip-so400m-patch14-384",
        mm_resampler_type=None,
        mm_projector_type='mlp2x_gelu',
        mm_hidden_size=1152,
        use_mm_proj=True
    )
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device,vision_kwargs=vision_kwargs,torch_dtype='bfloat16') # Add any other thing you want to pass in llava_model_args
    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)
    model.requires_grad_(False)
    return tokenizer, model, image_processor

def gumbel_noise(t, generator=None):
    noise = torch.zeros_like(t).uniform_(0, 1, generator=generator)
    return -torch.log(-torch.log(noise))



t2i_prompts = [
    "A cinematic wide shot of a lone astronaut standing on a desolate, alien planet, bathed in the glow of a binary sunset. Dust swirls around their boots. Highly detailed, 8K, sci-fi art, dramatic lighting.",
    "Macro shot of a dewdrop clinging to a spiderweb, reflecting an entire miniature forest scene within its surface. Ultra-realistic, extreme detail, natural light, bokeh background.",
    "An impressionistic painting of a bustling Parisian street cafe in the rain, seen through a slightly blurred window. Warm lights glow inside, contrasting with the cool, wet tones outside. Soft brushstrokes, vibrant colors, atmospheric.",
    "A mythical creature, a griffin with iridescent feathers, soaring majestically over a fantastical, mist-shrouded mountain range at dawn. Epic fantasy art, golden hour lighting, dynamic pose.",
    "Steampunk city skyline at night, with intricate clockwork buildings, glowing gears, and airships traversing the sky. Ornate, detailed, victorian, futuristic, volumetric lighting.",
    "A surreal landscape where a giant, floating teapot pours a waterfall into a teacup-shaped lake, surrounded by lollipop trees. Dreamlike, vibrant colors, whimsical, Salvador Dali inspired.",
    "Photorealistic portrait of an elderly Japanese fisherman with a weathered face, smoking a pipe, against a backdrop of stormy seas and traditional wooden boats. Deep wrinkles, intense eyes, chiaroscuro lighting, gritty textures.",
    "A cyberpunk geisha with glowing neon tattoos and advanced augmented reality glasses, standing in a rain-slicked Tokyo alleyway at night. Blade Runner aesthetic, vibrant neons, detailed cybernetics, reflective surfaces.",
    "Baroque style painting of a majestic lion wearing a regal crown and a velvet cape, sitting on a throne in a grand, opulent hall. Rich textures, dramatic shadows, golden accents, classical art.",
    "Underwater photography of a bioluminescent deep-sea creature, like an anglerfish or jellyfish, glowing against the absolute darkness of the abyss. Ethereal, mysterious, high contrast, focus on light emission.",
    "A futuristic cityscape seen from above, with flying vehicles, towering skyscrapers, and interconnected sky bridges, under a clear, bright blue sky. Clean lines, minimalist, utopian, architectural visualization.",
    "Gothic horror scene of a vampire noblewoman standing in a decaying, moonlit castle ballroom, mist swirling around her feet. Dark tones, intricate lace, Victorian clothing, eerie atmosphere.",
    "Highly stylized, whimsical illustration of a fox wearing a tiny monocle and top hat, reading a miniature book in a cozy, mushroom-filled forest glade. Children's book art, gentle colors, charming, adorable.",
    "Still life composition of a single, perfectly ripe peach on a dark, rustic wooden table, illuminated by a single shaft of natural light from a window. Hyper-realistic, Vermeer inspired, textures, subtle shadows.",
    "A vibrant, chaotic street art mural covering an entire brick wall, depicting fantastical creatures and abstract patterns in a burst of colors. Graffiti style, urban, dynamic composition, spray paint texture.",
    "Ancient Egyptian pharaoh's tomb, dimly lit by flickering torches, revealing hieroglyphs, sarcophagi, and glittering gold treasures. Archaeological discovery, detailed, dusty, mysterious.",
    "A renaissance portrait of a modern individual, perhaps wearing a hoodie or holding a smartphone, but rendered in the style of a 16th-century master. Oil painting, soft lighting, anachronistic, intriguing.",
    "Nordic mythology inspired landscape with towering, snow-capped fjords, a longship sailing through icy waters, and the aurora borealis shimmering in the sky. Vast, majestic, cold tones, dramatic lighting.",
    "Abstract expressionist painting composed entirely of swirling, vibrant colors that evoke a sense of intense motion and emotion, without discernible figures. Jackson Pollock style, bold strokes, energetic, non-representational.",
    "A close-up of a fantastical flower, with petals made of glass and leaves of shimmering metal, blooming in an enchanted, bioluminescent garden. Magical realism, intricate details, glowing elements, dreamlike."
]


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        #print(sorted_indices_to_remove.shape[-1]-sorted_indices_to_remove.sum(-1))
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

@torch.no_grad()
def text_to_image(model, 
    prompt,
    sample_policy = 'multinomial',
    confidence_policy = 'mmada',
    guidance_scale = 5,
    n_steps = 20,
    batch_size = 1,
    tokenizer=None,
    image_resolution=512,
    n_tokens=1024,
    shift=3,
    alg_temp=1,
    schedule='shift',
    min_temperature=0.01,
    dynamic_temperature=False,
    micro_cond = 'ORIGINAL WIDTH : 1024; ORIGINAL HEIGHT : 1024; TOP : 0; LEFT : 0; SCORE : 6.5',
    temperature=1,
    schedule_temp='linear',
    shift_alg=None,
    is_4k=False,
    is_4k_2=False,
    top_p=None,
    top_k=None,
    unmask_order=None,
    init_image=None,
    use_3d=False,
    schedule_temp_samp='linear',
    dynamic_temperature_samp=False,
    min_temperature_samp=1,
    cfg_interval=[0,1],
    order_cutoff = 100,
    edit_image=None,
    image_processor=None,
    guidance_scale_image=5,              
    edit_mode=0,
    n_refinement=0,
    local_temp_samp_refinement=0.00001,
    remask_ratio = 0.01,
    template='Generate an image with the caption:\n <prompt>',
    guidance_scale_refinement=1,
    enable_stratified=False,
    plan=None,
    feedback_imgs=None,
    feedback_texts=None,
    *args,
    **kwargs
):
    if shift_alg is None:
        shift_alg = shift
    device = model.get_model().device
    conv_template = "llada" 
    reserve_token = '<|reserved_token_5|>'
    reserve_id = 126089
    reserve_id2 = 126090
    #OFFSET = 150000
    INT_MAX = 1_000_000
    img_mask_id =8193
    txt_mask_id = 126336
    plan_range = 126349
    #img_mask_offset = OFFSET+img_mask_id

    gen_shape_map = {
        1024: (64,64),
        512: (32,32),
        256: (16,16),
    }
    gen_shape = gen_shape_map[image_resolution]
        
    micro_cond = micro_cond
    # if len(prompt) > 800:
    #     prompt = prompt[:800]
    if edit_image is not None:
        micro_cond = '' # disable
    prompt = f"{prompt} {micro_cond}"
    if feedback_texts is not None:
        feedback_str = [f'Generation {i+1}: <image>\n Feedback {i+1}: {feedback}' for i,feedback in enumerate(feedback_texts)]
        feedback_str = '\n'.join(feedback_str)
        feedback_str = f' Please also consider past generations and their feedbacks. Do not repeat these errors {feedback_str}'
        prompt = f"{prompt} {feedback_str}"
    question = template.replace('<prompt>',prompt)
    n_tokens_txt = n_tokens
    if image_resolution == 1024:
        n_tokens_txt = 1024
    if plan is None:
            plan = ""
    if edit_image is not None:
        image = edit_image
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.bfloat16, device=model.device) for _image in image_tensor]
        modalities = ['image']
        image_sizes = [image.size]
        conv_template = 'llada'
        n_tokens_txt = 1024
        conv = copy.deepcopy(conv_templates[conv_template])
        reserve_token_2 = '<|reserved_token_6|>'
        # breakpoint()
        image = image.convert('RGB')
        image_1024 = pad_to_square_and_resize(image,image_resolution)
        enc_latents,_gen_shape = model.encode_image_gen(model.model.image_processor_gen.preprocess(image_1024).to(model.device,model.dtype),enc=True)
        enc_embeddings = model.model.call_gen_embedding(enc_latents,_gen_shape,enc=True)
        conv.append_message(conv.roles[0], f"<image> {reserve_token_2*enc_embeddings.shape[1]}\n {prompt} ")
        reserve_token = '<|reserved_token_5|>'
        conv.append_message(conv.roles[1], f"{plan}{reserve_token*n_tokens_txt}")
        prompt_question = conv.get_prompt()
        prompt_question.removesuffix('<|start_header_id|>assistant<|end_header_id|>\n\n')
        print(prompt_question.replace('<|reserved_token_5|>','*').replace('<|reserved_token_6|>','*'))
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

        attention_mask=None
        position_ids=None
        image_sizes = [image.size]
        modalities=["image"]
        
        (inputs, position_ids, attention_mask, _, inputs_embeds, _,raw_input_ids) = model.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, None, None, image_tensor, modalities, image_sizes=image_sizes,return_inputs=True)
        # print(raw_input_ids)
        
        inputs_embeds[raw_input_ids == reserve_id2] = 0
        _enc = pad_along_last_dim(enc_embeddings,inputs_embeds.shape[-1])
        # _enc = torch.zeros_like(inputs_embeds[raw_input_ids == reserve_id2])
        # assert _enc[:,:enc_embeddings.shape[-1]].shape ==enc_embeddings.flatten(0,1).shape
        # _enc[:,:enc_embeddings.shape[-1]] = enc_embeddings.flatten(0,1)
        inputs_embeds[raw_input_ids == reserve_id2] = _enc.flatten(0,1)
        is_eot = torch.where(raw_input_ids==126348)[1]
        assert len(is_eot) == 3
        prompt_cutoff = is_eot[1]
        is_prompt = torch.zeros_like(raw_input_ids,dtype=torch.bool)
        is_prompt[:,:prompt_cutoff+1] = True
        is_gen = raw_input_ids == reserve_id
        is_gen_enc = raw_input_ids == reserve_id2
    else:
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], f"{plan}{reserve_token*n_tokens_txt}")
        prompt_question = conv.get_prompt()
        prompt_question.removesuffix('<|start_header_id|>assistant<|end_header_id|>\n\n')
        print(prompt_question.replace('<|reserved_token_5|>','*'))
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        is_gen = input_ids == reserve_id
        is_gen_enc = input_ids == reserve_id2
        is_eot = torch.where(input_ids==126348)[1]
        assert len(is_eot) == 3
        prompt_cutoff = is_eot[1]
        is_prompt = torch.zeros_like(input_ids,dtype=torch.bool)
        is_prompt[:,:prompt_cutoff+1] = True
        raw_input_ids = input_ids
        if feedback_texts is None:
            inputs_embeds,_ = wte(model.model,raw_input_ids)
        else:
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
    
            def processs_image_feedback(image):
                
                image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
                image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                return image
            def load_image(image):
                if isinstance(image,str):
                    image = Image.open(image).convert('RGB')
                return image.convert('RGB')
            attention_mask=None
            position_ids=None
            feedback_imgs = [load_image(x) for x in feedback_imgs]
            image_tensor = [processs_image_feedback(x) for x in feedback_imgs ]
            image_tensor = [_image.to(dtype=torch.bfloat16, device=model.device) for _image in image_tensor]
            image_sizes = [x.size for x in feedback_imgs]
            modalities=["image"]       
            (inputs, position_ids, attention_mask, _, inputs_embeds, _,raw_input_ids) = model.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, None, None, image_tensor, modalities, image_sizes=image_sizes,return_inputs=True)
            is_eot = torch.where(raw_input_ids==126348)[1]
            assert len(is_eot) == 3
            prompt_cutoff = is_eot[1]
            is_prompt = torch.zeros_like(raw_input_ids,dtype=torch.bool)
            is_prompt[:,:prompt_cutoff+1] = True
            is_gen = raw_input_ids == reserve_id
            is_gen_enc = raw_input_ids == reserve_id2  
    if plan:
        is_gen_min = torch.where(raw_input_ids==reserve_id)[1][0]
        is_prompt[:,:is_gen_min] = True
    inputs_embeds_uncond = inputs_embeds.clone()
    noise_embed = model.model.transformer.wte(torch.tensor([txt_mask_id]).to(model.device)) # 1, d
    inputs_embeds_uncond[is_prompt] =  noise_embed
    if edit_image is not None:
        inputs_embeds_uncond_enc = inputs_embeds.clone()
        if edit_mode == 0:
            inputs_embeds_uncond_enc[~is_gen_enc] = noise_embed
            is_gen_enc_ccc = is_gen_enc
        elif edit_mode == 1:
            inputs_embeds_uncond_enc[is_gen_enc] = noise_embed
            is_gen_enc_ccc =  torch.zeros_like(is_gen_enc,dtype=torch.bool)
        elif edit_mode == 2:
            inputs_embeds_uncond_enc[is_gen_enc|(raw_input_ids<0)] = noise_embed
            is_gen_enc_ccc =  torch.zeros_like(is_gen_enc,dtype=torch.bool)
        elif edit_mode == 3:
            inputs_embeds_uncond_enc[(~is_gen_enc) & (raw_input_ids>0)] = noise_embed
            is_gen_enc_ccc = is_gen_enc
        else:
            raise ValueError("Not Supported edit_mode")
    
    image_gen_latents_offset = torch.zeros(batch_size,n_tokens, dtype=torch.long, device=input_ids.device) #+ img_mask_offset
    is_unitok = 'unitok' in model.get_model().config.mm_vqvae
    if is_unitok:
        image_gen_latents_offset = image_gen_latents_offset.unsqueeze(1).repeat(1,8,1) # N, Num codebooks, Length1
    image_gen_latents_offset[:] =  img_mask_id

        #x_txt_raw = all_input_ids.clone()

    
    xt = image_gen_latents_offset.clone()
    if init_image is not None:
        init_latents,_gen_shape = model.encode_image_gen(model.model.image_processor_gen.preprocess(img).to(model.device,model.dtype))
        n_mask_remask = max(int(n_tokens * remask_ratio),1)
        indices = np.arange(n_tokens)
        np.random.shuffle(indices)
        init_mask_indices = indices[:n_mask_remask]
        xt[:,init_mask_indices] = init_latents[:,init_mask_indices]
        print("INIT INDICES",init_mask_indices)
    mask_idx = xt == img_mask_id
    if is_unitok: 
        mask_idx = mask_idx[:,0,:]
    n_mask = mask_idx.sum()
    if is_4k:
        mask_idx = mask_idx[:, :1024]
    mask_idx_sch = mask_idx
    if use_3d:
        mask_idx_sch = mask_idx_sch.repeat(1,8)
    if schedule == 'shift':
        num_transfer_tokens = get_num_transfer_tokens_sch(mask_idx_sch, n_steps,schedule='shift',schedule_kwargs=dict(shift=shift))
    else:
        num_transfer_tokens = get_num_transfer_tokens_sch(mask_idx_sch, n_steps,schedule=schedule,schedule_kwargs=dict(shift=shift))
    print(num_transfer_tokens)
    #sch_temperatures = #torch.linspace(1, min_temperature, n_steps, device=device)
    sch_temperatures = torch.linspace(0,1, n_steps, device='cpu').numpy()
    if schedule_temp == 'linear':
        sch_temperatures = (1- sch_temperatures) * (1 - min_temperature) + min_temperature
    elif schedule_temp == 'cosine2':
        sch_temperatures = cosine_schedule_2(1-sch_temperatures) * (1 - min_temperature) + min_temperature
    elif schedule_temp == 'shift':
        sch_temperatures = logit_normal_schedule(shift_alg,1-sch_temperatures) * (1 - min_temperature) + min_temperature
    elif schedule_temp_samp == 'exp':
        sch_temperatures = exp_schedule(1-sch_temperatures) * (1 - min_temperature) + min_temperature
    else:
        raise NotImplementedError(f"Unknown schedule_temp: {schedule_temp}")

    sch_temperatures_samp = torch.linspace(0,1, n_steps, device='cpu').numpy()
    if schedule_temp_samp == 'linear':
        sch_temperatures_samp = (1- sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
    elif schedule_temp_samp == 'cosine2':
        sch_temperatures_samp = cosine_schedule_2(1-sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
    elif schedule_temp_samp == 'shift':
        sch_temperatures_samp = logit_normal_schedule(shift_alg,1-sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
    elif schedule_temp_samp == 'exp':
        sch_temperatures_samp = exp_schedule(1-sch_temperatures_samp) * (1 - min_temperature_samp) + min_temperature_samp
    else:
        raise NotImplementedError(f"Unknown schedule_temp_samp: {schedule_temp_samp}")
        
    sch_temperatures = torch.tensor(sch_temperatures, device=device, dtype=torch.float32)
    sch_temperatures_samp = torch.tensor(sch_temperatures_samp, device=device, dtype=torch.float32)
    temp_idx = 0
    cfg_start = int(cfg_interval[0] * n_steps)
    cfg_end = int(cfg_interval[1] * n_steps)
    if isinstance(unmask_order, str):
        unmask_order =  torch.load(unmask_order).tolist()
    if confidence_policy == 'stratified' and unmask_order is None:
        _dim = int(np.sqrt(n_tokens))
        seq = stratified_random(n=_dim, seed=42, shuffle_blocks=True)
        unmask_order = seq
    if confidence_policy == 'stratified':
        assert unmask_order is not  None
    n_c = 0
    cat_cfg_time = 0
    post_cfg_time = 0
    wte_time = 0
    fwd_time = 0
    sample_time = 0
    all_time = 0
    for num_transfer in tqdm(num_transfer_tokens[0]):
        t0 = time.time()
        local_temp = sch_temperatures[temp_idx]
        local_temp_samp = sch_temperatures_samp[temp_idx]
        temp_idx += 1
        if temp_idx / n_steps > order_cutoff:
            confidence_policy = 'mmada'
        # if temp_idx == n_steps:
        #     sample_policy = 'argmax'
        mask_idx = xt == img_mask_id
        if is_unitok and not use_3d:
            mask_idx = mask_idx[:,0,:] 
        n_mask = mask_idx.sum()

        timesteps = n_mask / mask_idx.numel()
        timesteps = timesteps.view(1)
        # if n_mask==0:
        #     break
        
        # Get Logits
        do_cfg = guidance_scale > 0 and cfg_start <= temp_idx and  cfg_end >= temp_idx
        if do_cfg:
            if edit_image is not None:
                input_embeddings_input = torch.cat([inputs_embeds_uncond,inputs_embeds_uncond_enc,inputs_embeds]).clone()
                # nothing, masking vq tokens, final inputs
                xt_input = torch.cat([xt,xt,xt])
                new_token_mask = is_gen.repeat(3,1)
                is_gen_enc_null = torch.zeros_like(is_gen_enc,dtype=torch.bool)
                is_gen_enc_mask = torch.cat([is_gen_enc_null,is_gen_enc_ccc,is_gen_enc])
                timesteps = timesteps.repeat(3)
            else:
                input_embeddings_input = torch.cat([inputs_embeds_uncond,inputs_embeds]).clone()
                xt_input = torch.cat([xt,xt])
                new_token_mask = is_gen.repeat(2,1)
                is_gen_enc_mask = is_gen_enc.repeat(2,1)
                is_gen_enc_mask[0,:] = False # for uncond, no need to have different branch
                timesteps = timesteps.repeat(2)
        else:
            input_embeddings_input = inputs_embeds.clone()
            # x_txt_raw_input = x_txt_raw
            new_token_mask = is_gen
            xt_input = xt
            is_gen_enc_mask = is_gen_enc
        # breakpoint()
        enc_use_image_branch = getattr(model.get_model().config,'enc_use_image_branch',False)
        if enc_use_image_branch:
            modality_indices = new_token_mask| is_gen_enc_mask
        else:
            modality_indices = new_token_mask
        t1 = time.time()
        all_input_embeddings,new_token_mask = wte(model.get_model(),None,True,x_gen=xt_input,gen_shape=gen_shape,inputs_embeds_curr=input_embeddings_input,new_token_mask=new_token_mask)
        #torch.cuda.synchronize()
        t2 = time.time()
        logits = get_logits(model.get_model(),all_input_embeddings,new_token_mask,True,gen_shape=gen_shape,input_modality_indices=modality_indices,timesteps=timesteps)
        t3 = time.time()
        if do_cfg:
            if edit_image is not None:
                new_token_mask,_,_ = new_token_mask.chunk(3)
                logits_un,logits_un_enc,logits = logits.chunk(3)
                logits_is_ninf = logits == -np.inf
                #logits_cond = (logits + guidance_scale_image * logits_un_enc ) / (1 + guidance_scale_image)
                if edit_mode in [0, 3]:
                    logits_cond = (1 + guidance_scale_image) * logits - guidance_scale_image * logits_un_enc
                elif edit_mode in [1,2]:
                    logits_cond = (logits + guidance_scale_image * logits_un_enc ) / (1 + guidance_scale_image)                    
                #logits_cond = logits_cond
                logits = (1 + guidance_scale) * logits_cond - guidance_scale * logits_un
                logits[logits_is_ninf] = -np.inf
            else:
                new_token_mask,_ = new_token_mask.chunk(2)
                logits_un,logits = logits.chunk(2)
                logits_is_ninf = logits == -np.inf
                logits = (1 + guidance_scale) * logits - guidance_scale * logits_un
                logits[logits_is_ninf] = -np.inf
        # 4096 8 2064
        #torch.cuda.synchronize()
        t4 = time.time()
        if top_p is not None or top_k is not None:
            _b,_l,_v = logits.shape
            logits = logits.view(_b*_l,_v)
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, filter_value=-np.inf, min_tokens_to_keep=1)
            logits = logits.view(_b,_l,_v)
        if is_unitok:
            logits[...,4096:] = float('-inf')
        probs = logits.softmax(dim=-1) # N L 8 D
        
            
        if sample_policy == 'multinomial':
            _temperature = temperature
            if dynamic_temperature_samp:
                _temperature = _temperature * local_temp_samp
            x0 = dists.Categorical(logits=logits/_temperature).sample()
            x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
        elif sample_policy == 'argmax':
            x0 = logits.argmax(-1)
            x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
        else:
            raise NotImplementedError
        if is_unitok:
            # x0 is N L 8
            x0 = x0.permute(0,2,1) #  N 8 L
            # x0 = x0.unsqueeze(0) 
            if use_3d:
                x0 = torch.where(mask_idx, x0, xt)
                x0_p = x0_p.permute(0,2,1)[0] # N 8 L
            else:
                x0 = torch.where(mask_idx.unsqueeze(1).repeat(1,8,1), x0, xt)
                x0_p = x0_p.permute(0,2,1).max(dim=1)[0] # N L
        else:
            x0 = torch.where(mask_idx, x0, xt)
        # get confidence
        if confidence_policy == 'mask_git':
            _alg_temp = alg_temp
            if dynamic_temperature:
                _alg_temp = _alg_temp * local_temp
            confidence = x0_p / _alg_temp
            confidence = torch.where(mask_idx, confidence, -np.inf)
            if use_3d:
                confidence = confidence.flatten(1,2)
            confidence = torch.softmax(confidence, dim=-1)
            select_index = torch.multinomial(confidence, num_samples=num_transfer)
        elif confidence_policy == 'mmada':
            _alg_temp = alg_temp
            if dynamic_temperature:
                _alg_temp = _alg_temp * local_temp
            if is_4k:
                # x0_p = rearrange(x0_p,'b (h w) -> b h w', h=64, w=64)
                x0_p = rearrange(x0_p,'b (h p w q) -> b h w (p q)', h=32, w=32, p=2, q=2)
                x0_p = x0_p.max(dim=-1)[0]
                #print("b h w",x0_p.shape)
                x0_p = rearrange(x0_p,'b h w -> b (h w)')
                #print(mask_idx.shape)
                mask_idx = rearrange(mask_idx,'b (h p w q) -> b (h w) (p q)', h=32, w=32, p=2, q=2) 
                mask_idx = mask_idx.any(dim=-1)
                #print(mask_idx.shape)
            confidence = torch.log(x0_p.clamp(1e-20)) + _alg_temp * gumbel_noise(x0_p, generator=None)
            confidence = torch.where(mask_idx, confidence, -np.inf)
            if is_4k_2:
                confidence = rearrange(confidence,'b (h p w q) -> b h w (p q)', h=32, w=32, p=2, q=2)
                confidence_max = confidence.max(dim=-1,keepdims=True)[0]
                confidence[(confidence<confidence_max )&( confidence > -np.inf) ] = -9999 
                confidence = rearrange(confidence,'b h w (p q)-> b (h p w q)', h=32, w=32, p=2, q=2)
            if use_3d:
                confidence = confidence.flatten(1,2)
            # print(confidence.shape,num_transfer)
            _, select_index = torch.topk(confidence[0], k=num_transfer)
            if is_4k:
                z = torch.arange(4096,device=select_index.device, dtype=torch.long)
                z = rearrange(z,'(h w p q) -> (h w) (p q)', h=32, w=32, p=2, q=2)
                select_index = z[select_index].reshape(-1)
            # return select_index
        elif confidence_policy == 'stratified':
            # unmask_order
            start = n_tokens - n_mask
            if use_3d:
                start = n_tokens * 8 - n_mask
            
            select_index = torch.tensor(unmask_order[start:start+num_transfer], device=x0.device, dtype=torch.long)
            # print(select_index.shape)
            #print(select_index,n_mask,len(unmask_order))
        else:
            raise NotImplementedError
        
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        j = 0
        # _, select_index = torch.topk(confidence[j], k=num_transfer)
        # print(transfer_index.shape,select_index)
        # return
 
        if is_unitok:
            if use_3d:
                _b,_d,_l =  transfer_index.shape
                transfer_index = transfer_index.view(_b,-1)
                transfer_index[j, select_index] = True
                transfer_index = transfer_index.view(_b,_d,_l)
            else:
                transfer_index[j,:, select_index] = True
        else:
            transfer_index[j, select_index] = True
        xt[transfer_index] = x0[transfer_index]
        xt_is_mask = xt == img_mask_id
        t5 = time.time()
        cat_cfg_time += t1-t0
        post_cfg_time += t2-t1
        wte_time  += t3-t2
        fwd_time += t4-t3
        sample_time += t5-t4
        all_time += t5-t0
        n_c += 1
        # xt = x0
        # xt[xt_is_mask] = img_mask_id
    xt = x0.clone()
    # print('cat_cfg_time',cat_cfg_time/n_c)
    # print('post_cfg_time',post_cfg_time/n_c)
    # print('wte_time',wte_time/n_c)
    # print('fwd_time',fwd_time/n_c)
    # print('sample_time',sample_time/n_c)
    # print('all_time',all_time/n_c)
    guidance_scale = guidance_scale_refinement
    for _ in tqdm(range(n_refinement)):
        mask = torch.rand_like(xt,dtype=torch.float) < remask_ratio
        # print(mask)
        xt[mask] = img_mask_id
        n_refinement -= 1
        local_temp = 1 #sch_temperatures[temp_idx]
        local_temp_samp = 0.8 #sch_temperatures_samp[temp_idx]
        # temp_idx += 1
       
        mask_idx = xt == img_mask_id
        # print(mask_idx.sum())
        if is_unitok and not use_3d:
            mask_idx = mask_idx[:,0,:] 
        n_mask = mask_idx.sum()
        do_cfg = guidance_scale > 0
        if do_cfg:
            if edit_image is not None:
                input_embeddings_input = torch.cat([inputs_embeds_uncond,inputs_embeds_uncond_enc,inputs_embeds]).clone()
                # nothing, masking vq tokens, final inputs
                xt_input = torch.cat([xt,xt,xt])
                new_token_mask = is_gen.repeat(3,1)
                is_gen_enc_null = torch.zeros_like(is_gen_enc,dtype=torch.bool)
                is_gen_enc_mask = torch.cat([is_gen_enc_null,is_gen_enc_ccc,is_gen_enc])
            else:
                input_embeddings_input = torch.cat([inputs_embeds_uncond,inputs_embeds]).clone()
                xt_input = torch.cat([xt,xt])
                new_token_mask = is_gen.repeat(2,1)
                is_gen_enc_mask = is_gen_enc.repeat(2,1)
                is_gen_enc_mask[0,:] = False # for uncond, no need to have different branch
        else:
            input_embeddings_input = inputs_embeds.clone()
            # x_txt_raw_input = x_txt_raw
            new_token_mask = is_gen
            xt_input = xt
            is_gen_enc_mask = is_gen_enc
        # breakpoint()
        enc_use_image_branch = getattr(model.get_model().config,'enc_use_image_branch',False)
        if enc_use_image_branch:
            modality_indices = new_token_mask| is_gen_enc_mask
        else:
            modality_indices = new_token_mask
        all_input_embeddings,new_token_mask = wte(model.get_model(),None,True,x_gen=xt_input,gen_shape=gen_shape,inputs_embeds_curr=input_embeddings_input,new_token_mask=new_token_mask)
        logits = get_logits(model.get_model(),all_input_embeddings,new_token_mask,True,gen_shape=gen_shape,input_modality_indices=modality_indices)
        if do_cfg:
            if edit_image is not None:
                new_token_mask,_,_ = new_token_mask.chunk(3)
                logits_un,logits_un_enc,logits = logits.chunk(3)
                logits_is_ninf = logits == -np.inf
                #logits_cond = (logits + guidance_scale_image * logits_un_enc ) / (1 + guidance_scale_image)
                if edit_mode in [0, 3]:
                    logits_cond = (1 + guidance_scale_image) * logits - guidance_scale_image * logits_un_enc
                elif edit_mode in [1,2]:
                    logits_cond = (logits + guidance_scale_image * logits_un_enc ) / (1 + guidance_scale_image)                    
                #logits_cond = logits_cond
                logits = (1 + guidance_scale) * logits_cond - guidance_scale * logits_un
                logits[logits_is_ninf] = -np.inf
            else:
                new_token_mask,_ = new_token_mask.chunk(2)
                logits_un,logits = logits.chunk(2)
                logits_is_ninf = logits == -np.inf
                logits = (1 + guidance_scale) * logits - guidance_scale * logits_un
                logits[logits_is_ninf] = -np.inf
        probs = logits.softmax(dim=-1) # N L 8 D
        if sample_policy == 'multinomial':
            _temperature = temperature
            _temperature = _temperature * local_temp_samp_refinement
            x0 = dists.Categorical(logits=logits/_temperature).sample()
            x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
        elif sample_policy == 'argmax':
            x0 = logits.argmax(-1)
            x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
        else:
            raise NotImplementedError
        xt[mask] = x0[mask]
    
    # print(( xt == img_mask_id).sum()) # hack
    xt[xt==img_mask_id] = x0[xt==img_mask_id]
    if is_unitok:
        x0_img = x0 #-OFFSET
    else:
        x0_img = xt[None]
    
    # print(x0_img.shape)
    # # breakpoint()
    return Image.fromarray(model.decode_image_gen(x0_img,image_resolution,image_resolution)[0])

def create_plan(model,tokenizer,caption, micro_cond):
    conv_template = "llada" 
    #micro_cond = 'ORIGINAL WIDTH : 1024; ORIGINAL HEIGHT : 1024; TOP : 0; LEFT : 0; SCORE : 7.000; HPS : 3.222;'
    question = DEFAULT_IMAGE_TOKEN + "\nDescribe the image in detail."
    #caption = "an image of a warrior holding an apple, with birds flying in the sky"
    feedback_str = ''
    question = f"Generate an image with the caption: {caption} {micro_cond}. Please first think and plan the layout in LOC format.{feedback_str}"

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    image_sizes = []
    #warmup
    res = model.generate(
        input_ids,
        images=None,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0.5,
        max_new_tokens=64,
        block_length=16,
        step_ratio=1.0, # 32 steps
        tokenizer=tokenizer,
        prefix_lm=True,
        verbose=True,
    )
    plan = tokenizer.batch_decode(res[0])[0].replace('<|eot_id|>','').replace('<|endoftext|>','').replace('<image_gen_fake>','')
    return plan


def create_plan_edit(model,tokenizer,caption,edit_image,image_processor):
    device = model.device
    if isinstance(edit_image,str):
        edit_image = Image.open(edit_image).convert('RGB')
    else:
        edit_image = edit_image.convert('RGB')
    edit_image = resize_and_center_crop(edit_image,1024)
    edit_prompt = caption
    image_tensor = process_images([edit_image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
    conv_template = "llada" 
    question = f"<image>\n{edit_prompt}. Please first think and plan the layout in LOC format."

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    image_sizes = [edit_image.size]
    res = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=64,
        block_length=16,
        step_ratio=1.0, # 32 steps
        tokenizer=tokenizer,
        prefix_lm=True,
        verbose=True,
    )
    plan = tokenizer.batch_decode(res[0])[0].replace('<|eot_id|>','').replace('<|endoftext|>','').replace('<image_gen_fake>','')
    return plan

def get_feedback(model,tokenizer,image_processor,caption,image):
    if isinstance(image,str):
        edit_image = Image.open(image)
    else:
        edit_image = image
    device = model.device
    edit_image = resize_and_center_crop(edit_image,1024)
    image_tensor = process_images([edit_image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
    conv_template = "llada" 
    question = f"<image>\nPlease evaluate this generated image based on the following prompt: {caption} Focus on text alignment and compositionality."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    image_sizes = [edit_image.size]
    res = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=32,
        block_length=16,
        step_ratio=1.0, # 32 steps
        tokenizer=tokenizer,
        prefix_lm=True,
        verbose=True,
    )
    plan = tokenizer.batch_decode(res[0])[0].replace('<|eot_id|>','').replace('<|endoftext|>','').replace('<image_gen_fake>','')
    return plan



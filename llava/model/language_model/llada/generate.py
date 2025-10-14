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

import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from llava.model.utils import maybe_truncate_last_dim,pad_along_last_dim

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def get_num_transfer_tokens_sch(mask_index, steps,schedule=None,schedule_kwargs=None):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    if schedule is None:
        return get_num_transfer_tokens(mask_index,steps)
    if schedule_kwargs is None:
        schedule_kwargs = {}
   
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = int(min(steps,mask_num[0]))
    t = torch.linspace(0, 1, steps+1)
    # at least one sample per step
    if schedule =='logit_normal':
      sigmas = sigmoid_normal_cdf(t)
    elif schedule =='shift':
      sigmas = logit_normal_schedule(schedule_kwargs.get('shift',3),t)
    elif schedule == 'cosine':
        sigmas = cosine_schedule(t)
    elif schedule == 'cosine2':
        sigmas = cosine_schedule_2(t)
    elif schedule == 'exp':
        sigmas = exp_schedule(t)
    else:
      sigmas = t
    sigmas = sigmas.to(mask_num.device)
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
    
    for i in range(mask_num.size(0)):
      # print(sigmas.shape)
      sigmas_sample = (sigmas*mask_num[i]).to(torch.int64)
      # print(sigmas_sample)
      sigmas_sample = sigmas_sample[1:]-sigmas_sample[:-1]
      # print(sigmas_sample)
      # fix detal
      sigmas_sample = torch.clamp(sigmas_sample,1,None) # should only increase
      delta = sigmas_sample.sum() - mask_num[i]
    #   breakpoint()
      assert delta>=0
      j = 0
      
      while delta > 0:
        j = j % len(sigmas_sample) 
        if sigmas_sample[j] == 1:
          j += 1
          continue
        
        delta -= 1
        sigmas_sample[j] -= 1
        j += 1
    #   breakpoint()
      assert sigmas_sample.sum()==mask_num[i]
      num_transfer_tokens[i] = sigmas_sample#.to(torch.int64)
    return num_transfer_tokens.flip(-1)

def linear(y):
    return y

def cosine_schedule(x):
    """
    Cosine schedule mapping [0, 1] -> [1, 0]
    """
    x = np.clip(x, 0, 1)
    return 1-0.5 * (1 + np.cos(np.pi * x))


def cosine_schedule_2(x):
    """
    Cosine schedule mapping [0, 1] -> [1, 0]
    """
    x = 1- np.clip(x, 0, 1)
    return np.cos(np.pi * x / 2)

def exp_schedule(x):
    """
    Cosine schedule mapping [0, 1] -> [1, 0]
    """
    # x = 1- np.clip(x, 0, 1)
    z = (1-np.exp(-5 * x) )/ (1-np.exp(-5))
    return np.clip(z,0.0001,1)

def sigmoid_normal_cdf(y):
    # y must be in (0, 1)
    logit_y = torch.log(y / (1 - y))
    return 0.5 * (1 + torch.erf(logit_y / torch.sqrt(torch.tensor(2.0))))
def logit_normal_schedule(shift,sigmas):
    # shift = 1 / shift
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return sigmas

INT_MAX = 1_000_000
def get_logits(model,input_emnbeddings,modality_indices=None,t2i_inference=False,past_key_values=None,gen_shape=None,timesteps=None,input_modality_indices=None):
    if t2i_inference:
        if input_modality_indices is None:
            input_modality_indices =modality_indices
        output = model(None,input_embeddings=input_emnbeddings,modality_indices=input_modality_indices,output_hidden_states=True,past_key_values=past_key_values)
        hidden_states = output.hidden_states[-1]
        gen_hidden_states = hidden_states[modality_indices]
        gen_hidden_states = maybe_truncate_last_dim(gen_hidden_states,model.config.d_model_gen)
        gen_logits = model.call_gen_predictor(gen_hidden_states,gen_shape,timesteps=timesteps) # * 8 D
        seq_len_per_img = np.product(gen_shape)
        if len(gen_logits.shape) == 2:
            gen_logits = gen_logits.view(-1,seq_len_per_img,gen_logits.shape[-1])
        else:
            gen_logits = gen_logits.view(-1,seq_len_per_img,*gen_logits.shape[-2:])
            # N L 8 D
        return gen_logits
        
        
        final_logits = torch.zeros(*gen_logits.shape[:-1],OFFSET+gen_logits.shape[-1],dtype=output.logits.dtype,device=output.logits.device)
        final_logits[:] = float('-inf')
        final_logits[...,OFFSET:] = gen_logits
        # breakpoint()
        # inal_logits = torch.zeros(*hidden_states.shape[:-1],OFFSET+gen_logits.shape[-1],dtype=output.logits.dtype,device=output.logits.device)
        
        # final_logits = final_logits + float('-inf')
        # final_logits[...,:output.logits.shape[-1]] = output.logits
        # final_logits[modality_indices] = float('-inf')
        # local = final_logits[modality_indices]

        # local[...,OFFSET:] = gen_logits
        # final_logits[modality_indices] = local

        logits = final_logits
        return logits
    else:
        modality_indices = torch.zeros(input_emnbeddings.shape[:-1],device=input_emnbeddings.device,dtype=torch.bool)
        logits = model(None,input_embeddings=input_emnbeddings,modality_indices=modality_indices,past_key_values=past_key_values).logits
    return logits

def wte(model,x,t2i_inference=False,gen_shape=None,x_gen=None,inputs_embeds_curr=None,new_token_mask=None):

    if t2i_inference:
        assert x_gen is not None
        if new_token_mask is None:
            new_token_mask = x >= INT_MAX
        # if x_gen is  None:
        #     x_gen = x[new_token_mask] - OFFSET
        # else:
        #     x_gen = x_gen - OFFSET

        gen_latents_comp_embeds = model.call_gen_embedding(x_gen,gen_shape)
        if inputs_embeds_curr is None:
            x_txt_only = x.clone()
            
            # replace consequtent [1] * 4096 to [1] * 1024

            x_txt_only[new_token_mask] = 0
            inputs_embeds_curr = model.transformer.wte(x_txt_only)
        inputs_embeds_curr[new_token_mask] = pad_along_last_dim(gen_latents_comp_embeds,inputs_embeds_curr.shape[-1]).view(-1,inputs_embeds_curr.shape[-1])
    else:
        inputs_embeds_curr = model.transformer.wte(x)
        new_token_mask = None
    return inputs_embeds_curr,new_token_mask
    
    
import os
DEBUG_PRINT_OUTPUT = os.environ.get('DEBUG_PRINT_OUTPUT',False)
@ torch.no_grad()
def generate(model, prompt=None, steps=None, max_new_tokens=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,inputs_embeds=None, position_ids=None,attention_mask=None,
              tokenizer=None,
                verbose=False,
                step_per_block=None,
                prefix_lm=False,
                schedule=None,
                schedule_kwargs=None,
                draft_tokens=None,
                step_ratio=None,
                prompt_index=None,
                t2i_inference=False,
                input_modality_indices=None,
             **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    # breakpoint()
    # remasking = 
    # step_ratio = 0.5
    # block_length = 1024
    # steps = 1024
    # OFFSET = 150000 # used in inference
    steps = max_new_tokens # min(steps,max_new_tokens)
    # if step_ratio:
    #     steps = int(max_new_tokens*step_ratio)
    gen_length = max_new_tokens
    assert position_ids is None
    if prompt is None:
        assert inputs_embeds is not None
        bsz, seq_len = inputs_embeds.shape[:2]
        prompt = torch.full((bsz, seq_len), 0, dtype=torch.long).to(model.device)
    past_key_values = None
    if prefix_lm:
        print("HH2")
        if input_modality_indices is None and t2i_inference:
            input_modality_indices = torch.zeros(inputs_embeds.shape[:-1], dtype=torch.bool, device=inputs_embeds.device)
        past_key_values = model(None,input_embeddings=inputs_embeds,use_cache=True,modality_indices=input_modality_indices).attn_key_values
        # breakpoint()
        # print(past_key_values is None)
        x = torch.full((bsz, gen_length), mask_id, dtype=torch.long).to(model.device)
        prompt = torch.full((bsz, 0), 0, dtype=torch.long).to(model.device)
        # x[:, :prompt.shape[1]] = prompt.clone()
    else:
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

    if prompt_index is None:
        prompt_index = (x != mask_id)
    # assert prompt.shape[0] == 1
    if draft_tokens is not None:
        assert draft_tokens.shape[1] <= gen_length
        x[:, prompt.shape[1]:prompt.shape[1]+draft_tokens.shape[1]] = draft_tokens.clone()

    # if block_length < gen_length:
    #    block_length = gen_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    # breakpoint()
    assert ( steps % num_blocks == 0) or step_per_block is not None
    steps = steps // num_blocks
    if step_per_block:
        steps = min(step_per_block,block_length)
        assert step_ratio is None, 'Please do not pass both step_ratio and step_per_block'
    # step_ratio = 0.5
    # schedule = 'shift'
    # schedule_kwargs = dict(shift=3)
    # breakpoint()
    if step_ratio:
        steps = int(steps*step_ratio)

    # print(steps,step_per_block,block_length,draft_tokens.shape[-1])
    # NFE = 0
    if verbose:
        history = []
    noise_embeddings = model.transformer.wte(torch.tensor([126336]).to(model.device)) # 1, 4096
    for num_block in range(num_blocks):

        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_sch(block_mask_index, steps,schedule=schedule,schedule_kwargs=schedule_kwargs)
        if DEBUG_PRINT_OUTPUT:
            print(f"Block: {num_block + 1}/{num_blocks}, Steps per Block: {steps}, Block Length: {block_length}")
            print(f"Tokens generated per step {num_transfer_tokens[0]}")
        for i in range(steps):
            # print(i)
            mask_index = (x == mask_id)
            block_mask_index = mask_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:]
            # print(mask_index.sum())
            if block_mask_index.sum() == 0:
                continue
            # NFE += 2

            inputs_embeds_curr,new_token_mask = wte(model,x,t2i_inference)
                

            if cfg_scale > 0.:
                #assert NotImplementedError('cfg_scale > 0. is not supported.')
                un_inputs_embeds_curr = inputs_embeds_curr.clone()
                if prefix_lm:
                    raise NotImplementedError()
                else:
                    inputs_embeds_curr[:,:inputs_embeds.shape[1]] = inputs_embeds
                    un_inputs_embeds_curr[:,:inputs_embeds.shape[1]] = inputs_embeds
                    un_inputs_embeds_curr[prompt_index] = noise_embeddings
                    inputs_embeds_curr_cat = torch.cat([un_inputs_embeds_curr, inputs_embeds_curr], dim=0)
                    logits = get_logits(model,inputs_embeds_curr_cat,new_token_mask.repeat(2,1),t2i_inference)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                if prefix_lm:
                    # print("HH")
                    # breakpoint()
                    #logits = model(None,input_embeddings=inputs_embeds_curr,past_key_values=past_key_values).logits
                    logits = get_logits(model,inputs_embeds_curr,new_token_mask,t2i_inference,past_key_values=past_key_values)
                else:
                    if inputs_embeds is not None:
                        inputs_embeds_curr[:,:inputs_embeds.shape[1]] = inputs_embeds
                    logits = get_logits(model,inputs_embeds_curr,new_token_mask,t2i_inference)
            # logits = logits.cpu()
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            # breakpoint()
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'entrophy':
                epsilon = 1e-10
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                log_probs = torch.log(probs + epsilon)
                x0_p = torch.sum(probs * log_probs, dim=-1)
            elif remasking == 'margin':
                ## similar to margin algo in Dream
                p = F.softmax(logits.to(torch.float64), dim=-1)
                sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                top1_probs = sorted_probs[:, :, 0] 
                top2_probs = sorted_probs[:, :, 1] 
                x0_p = top1_probs - top2_probs 
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                try:
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                except:
                    breakpoint()
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            if verbose:
                history.append(x.clone().cpu())
    # breakpoint()
    # print(f"NFE: {NFE} Num Blocks: {num_blocks}")
    if verbose:
        return x,history
    return x

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index
def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


@ torch.no_grad()
def generate_with_dual_cache(model, prompt=None,steps=128,max_new_tokens=128, block_length=128, temperature=0.,
            cfg_scale=0.,remasking='low_confidence', mask_id=126336, inputs_embeds=None,
            threshold=None, 
            factor=1.0,
            step_per_block=None,
            prefix_lm=False,
            schedule=None,
            schedule_kwargs=None,
            draft_tokens=None,
            step_ratio=None, 
            prompt_index=None,
            t2i_inference=False,
            input_modality_indices=None,
            **kwargs):
    # model, prompt=None, steps=128, max_new_tokens=128, block_length=128, temperature=0.,
            #  cfg_scale=0., remasking='low_confidence', mask_id=126336,inputs_embeds=None, position_ids=None,attention_mask=None,
            #   tokenizer=None,
            #     verbose=False,
            #     step_per_block=None,
            #     prefix_lm=False,
            #     schedule=None,
            #     schedule_kwargs=None,
            #     draft_tokens=None,
            #     step_ratio=None,          
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    gen_length = max_new_tokens
    assert draft_tokens is None, "Not supported in fast dllm"
    assert t2i_inference is False, "Not supported in fast dllm"
    assert input_modality_indices is None,"Not supported in fast dllm"
    # breakpoint()
    if prompt is None:
        assert inputs_embeds is not None
        bsz, seq_len = inputs_embeds.shape[:2]
        prompt = torch.full((bsz, seq_len), 0, dtype=torch.long).to(model.device)
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    if step_per_block:
        steps = min(step_per_block,block_length)
        assert step_ratio is None, 'Please do not pass both step_ratio and step_per_block'
    if step_ratio:
        steps = int(steps*step_ratio)

    nfe = 0  
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        inputs_embeds_curr = model.transformer.wte(x)
        inputs_embeds_curr[:,:inputs_embeds.shape[1]] = inputs_embeds
        output = model(x,input_embeddings=inputs_embeds_curr, use_cache=True)
        past_key_values = output.attn_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], None, factor)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
   

if __name__ == '__main__':
    main()

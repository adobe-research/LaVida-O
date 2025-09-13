
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from .llada import *
from transformers import AutoConfig, AutoModelForCausalLM

from torch.nn import CrossEntropyLoss

from .llada.modeling_llada import LLaDAModel,LLaDAModelLM,LLaDAConfig,create_model_config_from_pretrained_config
from .llada.generate import generate as llada_generate,generate_with_dual_cache
from llava.model.language_model.llada.log_likelyhood import get_log_likelihood as get_log_likelihood
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
import os
DO_DEBUG = os.environ.get('DO_DEBUG',False)
from accelerate.utils import reduce
from llava.model.utils import maybe_truncate_last_dim,pad_along_last_dim
from llava.constants import IGNORE_TEXT_LOSS,SKIP_DOWN_SAMPLE
import math
ENFORCE_NUM_ITEMIN_BATCH = os.environ.get("ENFORCE_NUM_ITEMIN_BATCH", False)
class LlavaLladaConfig(LLaDAConfig):
    model_type = "llava_llada"
    # temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
    # max_new_tokens: int = 1024
    # do_sample: bool = False
    # top_p: Optional[float] = None
    # rope_scaling: Optional[dict] = {}
    
    
class LlavaLladaModel(LlavaMetaModel,LLaDAModel):
    config_class = LlavaLladaConfig
    dtype = torch.bfloat16 # hack

    def __init__(self, pretrained_config,llada_config,init_params=None,vision_kwargs=None):
        # breakpoint()
        
        LLaDAModel.__init__(self, llada_config)
        LlavaMetaModel.__init__(self, pretrained_config,vision_kwargs=vision_kwargs,skip_init=True)
        
    def embed_tokens(self, x):
        return self.transformer.wte(x)

def sample_t(b,device,policy='uniform',policy_args=None):
    if policy == 'uniform':
        return torch.rand(b, device=device)
    elif policy == 'logit_normal':
        if policy_args is None:
            policy_args = dict(logit_mean=0.0,logit_std=1.0)
        u = torch.normal(mean=policy_args['logit_mean'], std=policy_args['logit_std'], size=(b,), device="cpu")
        u = torch.nn.functional.sigmoid(u).to(device=device)
        return u
    elif policy == 'cosine':
        timesteps = torch.rand(b, device=device)
        mask_prob = torch.cos(timesteps * math.pi * 0.5)
        mask_prob = mask_prob.clip(0,1)
        return mask_prob
    elif policy == "mode":
        u = torch.rand(size=(b,), device="cpu")
        u = 1 - u - policy_args['mode_scale'] * (torch.cos(torch.pi * u / 2) ** 2 - 1 + u)
        return u
        
def forward_process(bsz,seq_len,device, eps=1e-3,policy='uniform',policy_args=None):
    b, l = bsz,seq_len
    t = sample_t(b,device,policy=policy,policy_args=policy_args)
    # t = torch.sigmoid(t)
    p_mask = (1 - eps) * t + eps
    
    p_mask = p_mask[:, None]#.repeat(1, l)
    
    masked_indices = torch.rand((b, l), device=device)
    mask_cutoff =  torch.max(p_mask,masked_indices.min(-1,keepdim=True).values)
    masked_indices = masked_indices <= mask_cutoff
    # mask at least one token
    # 126336 is used for [MASK] token
    #noisy_batch = torch.where(masked_indices, 126336, input_ids)
    
    return masked_indices, p_mask
import os
LOG_BATCH_LENGTH = os.environ.get('LOG_BATCH_LENGTH', False)
DEBUG_PRINT_IMAGE_RES = os.environ.get("DEBUG_PRINT_IMAGE_RES", False)

SKIP_COMPLEMENTARY_MASKING = os.environ.get("SKIP_COMPLEMENTARY_MASKING", False)

class LlavaLladaForMaskedDiffusion(LLaDAModelLM,LlavaMetaForCausalLM):
    
    config_class = LlavaLladaConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config: LLaDAConfig, model: Optional[LLaDAModel] = None, init_params: bool = False,vision_kwargs=None,prefix_lm=False,**kwargs):
        if not hasattr(config,'d_model_gen') or config.d_model_gen < 0 :
            config.d_model_gen = config.d_model
        if not hasattr(config,'mlp_hidden_size_gen') or config.mlp_hidden_size_gen < 0:
            config.mlp_hidden_size_gen = config.mlp_hidden_size
        if not hasattr(config,'downsample'):
            config.downsample = False
        LLaDAModelLM.__init__(self, config,model,init_params)
        # hack

        # configure default generation settings
        config.model_type = "llava_llada"
        # config.rope_scaling = None
        self.prefix_lm = prefix_lm

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            self.model = LlavaLladaModel(config,model_config, init_params=init_params,vision_kwargs=vision_kwargs)
        else:
            self.model = model
        self.model.set_activation_checkpointing('whole_layer')
        
        self.post_init() # TODO
        # self.eos_id = 126081 # hack
        # self.mask_id = 126336 # hack
        
    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        cache_position=None,
        policy='uniform',
        policy_args=None,
        images_gen = None,
        gen_latents = None,
        t2i_inference=False,
        gen_shape=None,
        images_gen_enc = None,
        gen_latents_enc = None,
        image_gen_weight =None,
        dataset_name=None,
        do_not_mask_text=None,
        # **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print(dataset_name)
        # for now, lets just concat evrything
        reserve_id = 126089 # reserve , used in training
        reserve_id_enc = 126090 # reserved_token_6 reserve , used in training
        OFFSET = 150000 # used in inference

        if images_gen is not None or gen_latents is not None:
            if gen_latents is None:
                images_gen = torch.cat([torch.cat(x) for x in images_gen])
                gen_latents,gen_shape = self.encode_image_gen(images_gen)
            
            token_mask = (input_ids==-300)


        input_ids[input_ids==-300] =reserve_id

        eos_id = 126081 # hack
        mask_id = 126336
        img_mask_id =8193
        fim_id = 126085
        raw_inputs_ids = input_ids
        attention_mask_raw = attention_mask.clone()
        non_padding = ~(raw_inputs_ids==eos_id)
        attention_mask[raw_inputs_ids==eos_id] = True # no sequence attention mask per Sec B.1
        labels[raw_inputs_ids==eos_id] = eos_id # revert target
        # fix attention mask
        input_ids == input_ids
        # pad_len = torch.randint(0,pad_len_max,(1,)).item()
        # padding = torch.full((bsz,pad_len),eos_id,dtype=labels.dtype,device=labels.device) 
        skip_batch = 0
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels,new_input_ids) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes,return_inputs=True)
        assert input_ids is None
        print(dataset_name,new_input_ids.shape[-1],new_input_ids.shape[0])
        new_token_mask:torch.Tensor = (new_input_ids == reserve_id)
        new_token_mask_enc: torch.Tensor = (new_input_ids ==  reserve_id_enc)
        if images_gen_enc is not None or gen_latents_enc is not None:
            if gen_latents_enc is None:
                images_gen_enc = torch.cat([torch.cat(x) for x in images_gen_enc])
                gen_latents_enc,gen_shape_enc = self.encode_image_gen(images_gen_enc,enc=True)
        
            gen_latents_enc_embeds = self.get_model().call_gen_embedding(gen_latents_enc,gen_shape=gen_shape_enc,enc=True)  
            gen_latents_enc_embeds = pad_along_last_dim(gen_latents_enc_embeds,self.config.d_model)
            # if DO_DEBUG:
            #     breakpoint()
            if not new_token_mask_enc.sum() == gen_latents_enc_embeds.shape[0]*gen_latents_enc_embeds.shape[1]:
                skip_batch = 1
                print(f"SKIP BATCH!!! {dataset_name}")
            inputs_embeds_replaced = inputs_embeds.masked_scatter(new_token_mask_enc.unsqueeze(-1), gen_latents_enc_embeds.view(-1,4096))
            inputs_embeds = inputs_embeds_replaced
        do_inv = not SKIP_COMPLEMENTARY_MASKING
        if image_gen_weight is not None:
            image_gen_weight =torch.cat([torch.cat(x) for x in image_gen_weight])
        image_gen_weight_dup = image_gen_weight
        if do_inv:
            new_token_mask_dup = new_token_mask.repeat(2,1)
            new_token_mask_enc_dup = new_token_mask_enc.repeat(2,1)
            if image_gen_weight_dup is not None:
                image_gen_weight_dup = image_gen_weight_dup.repeat(2,1,1,1)
        else:
            new_token_mask_dup = new_token_mask
            new_token_mask_enc_dup = new_token_mask_enc
        modality_indices = new_token_mask_dup
        enc_use_image_branch = getattr(self.get_model().config,'enc_use_image_branch',False)
        if enc_use_image_branch:
            modality_indices = modality_indices| new_token_mask_enc_dup
        
        # breakpoint()
        # if t2i_inference:
        #     new_token_mask_dup = new_token_mask = new_input_ids >= OFFSET
        #     gen_latents_inference = new_input_ids[new_token_mask_dup] - OFFSET
        #     gen_latents_comp_embeds = self.get_model().gen_embedding(gen_latents_inference)
        #     inputs_embeds = inputs_embeds.masked_scatter(new_token_mask_dup.unsqueeze(-1), gen_latents_comp_embeds.view(-1,4096))
        #labels[new_token_mask].unique() = [-100]
        #prompt_lengths = 
        #breakpoint()
        # hack starts here
        # 1. Get the mask of trget tokens 
        # if we have labels, run forward process
        # prefix_length = 
        # 
        prompt_len = None
        
        if labels is not None:
            assert labels.min() == -100
            labels_mask = ~(labels == -100) # targets mask
            infill_token_pos = labels==fim_id
            # find index of the first non zero mask
            # labels_mask = labels_mask.cumsum(-1).eq(1)
            if self.prefix_lm:
                # breakpoint()
                prompt_len = labels_mask.float().argmax(dim=1)
                # print(prompt_len)
            noise_embeddings = self.get_model().transformer.wte(torch.tensor([mask_id]).to(raw_inputs_ids))
            # noise_embeddings is 1, 4096
            bsz,seq_len = labels_mask.shape
            noise_embeddings = noise_embeddings.view(1,1,-1)#.repeat(bsz,seq_len,1)
            # t = torch.rand(b, device=input_ids.device)
            masked_indices, p_mask = forward_process(bsz,seq_len,raw_inputs_ids.device,policy=policy,policy_args=policy_args)
            # torch.where()
            #breakpoint()
            prompt_drop_rate = getattr(self.config,'prompt_drop_rate',0)
            rand_drop = (torch.rand(labels_mask.shape[0], device=labels_mask.device) <prompt_drop_rate).view(-1,1)
            
            is_prompt = (~labels_mask) & (~new_token_mask)  # not text answer and not imag answer
            prompt_to_mask = is_prompt & rand_drop
            if images_gen_enc is not None:
                prompt_drop_rate_enc = getattr(self.config,'image_enc_drop_rate',0.5)
                rand_drop_enc = (torch.rand(labels_mask.shape[0], device=labels_mask.device) <prompt_drop_rate_enc).view(-1,1)
                image_enc_to_mask = rand_drop_enc & new_token_mask_enc_dup
                prompt_to_mask = prompt_to_mask & (~new_token_mask_enc_dup)
                prompt_to_mask = prompt_to_mask | image_enc_to_mask
                modality_indices = modality_indices & (~prompt_to_mask)
                # breakpoint()
                # N x 1
            #breakpoint()
            if do_not_mask_text is not None and sum(do_not_mask_text) > 0:
                do_not_mask = torch.tensor(do_not_mask_text)
                do_not_mask = do_not_mask & (torch.rand_like(do_not_mask,dtype=torch.float) < 0.8)
                do_not_mask = do_not_mask.unsqueeze(-1).to(masked_indices)
            else:
                do_not_mask = torch.zeros((masked_indices.shape[0],1),dtype=torch.bool).to(masked_indices)
            final_masked_indices = masked_indices & (~do_not_mask) &labels_mask & (~infill_token_pos) 
            final_masked_indices = final_masked_indices | prompt_to_mask
            
            if do_inv:
                final_masked_indices_inv = (~masked_indices) & (~do_not_mask) &labels_mask & (~infill_token_pos)
                final_masked_indices_inv = final_masked_indices_inv | prompt_to_mask
                inputs_embeds_inv = torch.where(final_masked_indices_inv.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)

            
            inputs_embeds = torch.where(final_masked_indices.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)
            # inputs_embeds_inv = torch.where(final_masked_indices_inv.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)
            # print(final_masked_indices.float().mean(-1).cpu())
            # new_input_ids
            # breakpoint()
            if do_inv:
                labels_inv = labels.clone()
                labels_inv[~final_masked_indices_inv] = -100
            labels[~final_masked_indices] = -100
            labels[labels==fim_id] = -100 # kill infill token so we don't predict it
            is_unitok = 'unitok' in getattr(self.get_model().config,'mm_vqvae','')
            is_unitok_submask = getattr(self.get_model().config,'mm_submask',False)
            
            if do_inv:
                inputs_embeds = torch.cat([inputs_embeds,inputs_embeds_inv])
            if images_gen is not None:
                gen_latents_masked = gen_latents.clone()
                if is_unitok_submask:
                    # gen_latents is N 8 L 
                    # _latents_b, _, _latents_l = gen_latents.shape
                    # gen_latents = gen_latents,view()
                    masked_indices_gen, p_mask = forward_process(gen_latents_masked.shape[0],gen_latents.shape[-1]*8,raw_inputs_ids.device,policy=policy,policy_args=policy_args)
                    masked_indices_gen = masked_indices_gen.view(-1,8,gen_latents.shape[-1])
                else:
                    masked_indices_gen, p_mask = forward_process(gen_latents_masked.shape[0],gen_latents.shape[-1],raw_inputs_ids.device,policy=policy,policy_args=policy_args)
                    if is_unitok:
                        masked_indices_gen = masked_indices_gen.unsqueeze(1).repeat(1,8,1) # N 8
                        
                gen_latents_masked[masked_indices_gen] = img_mask_id
                if do_inv:
                    gen_latents_masked_inv = gen_latents.clone()
                    gen_latents_masked_inv[~masked_indices_gen] = img_mask_id
                    gen_latents_comp = torch.cat([gen_latents_masked,gen_latents_masked_inv])
                else:
                    gen_latents_comp = gen_latents_masked
                gen_latents_comp_embeds = self.get_model().call_gen_embedding(gen_latents_comp,gen_shape=gen_shape)  
                gen_latents_comp_embeds = pad_along_last_dim(gen_latents_comp_embeds,self.config.d_model)
                if do_inv:
                    gen_latents_comp_labels = torch.cat([gen_latents,gen_latents])
                else:
                    gen_latents_comp_labels = gen_latents.clone()
                    
                gen_latents_comp_labels_raw = gen_latents_comp_labels.clone().detach()
                gen_latents_comp_labels[~(gen_latents_comp==img_mask_id)] = -100
                gen_latents_comp_labels_is_mask = gen_latents_comp==img_mask_id
                # if DO_DEBUG:
                #     breakpoint()
                if not new_token_mask_dup.sum() == gen_latents_comp_embeds.shape[0]*gen_latents_comp_embeds.shape[1]:
                    skip_batch = 1
                    print(f"SKIP BATCH!!! {dataset_name}")
                inputs_embeds_replaced = inputs_embeds.masked_scatter(new_token_mask_dup.unsqueeze(-1), gen_latents_comp_embeds.view(-1,4096))
                # inputs_embeds_replaced_2 = inputs_embeds.clone()
                # inputs_embeds_replaced_2[new_token_mask_dup] = gen_latents_comp_embeds.view(-1,4096)
                # assert torch.all(inputs_embeds_replaced == inputs_embeds_replaced_2)
                inputs_embeds = inputs_embeds_replaced
            
            if do_inv:
                labels =  torch.cat([labels,labels_inv])
                if self.prefix_lm:
                    prompt_len = prompt_len.repeat(2,1)
                final_masked_indices = torch.cat([final_masked_indices,final_masked_indices_inv])
            seq_len = labels.shape[-1]
            # print(seq_len)
            if LOG_BATCH_LENGTH:
                print("Batch Length",seq_len)
            CUFOFF=30720
            if seq_len > CUFOFF:
                print(seq_len,labels.shape)
                labels = labels[:,:CUFOFF]
                inputs_embeds = inputs_embeds[:,:CUFOFF]
                attention_mask = attention_mask[:,:CUFOFF]
                if position_ids is not None:
                    position_ids = position_ids[:,:CUFOFF]
                assert input_ids is None
                assert past_key_values is None
            elif seq_len < CUFOFF:
                pass
                # raise ValueError("Out of Length")
                # pad_len_max = 128 #torch.randint(0, 128, (1,)).item()
                # if pad_len_max > 0:
                #     pad_len = torch.randint(0,pad_len_max,(1,)).item()
                #     padding = torch.full((bsz,pad_len),eos_id,dtype=labels.dtype,device=labels.device) 
                #     labels = torch.cat([labels,padding],dim=-1)
                #     new_input_ids  = torch.cat([new_input_ids,padding],dim=-1)
                #     padding = torch.full((bsz,pad_len,inputs_embeds.shape[-1]),0,dtype=inputs_embeds.dtype,device=inputs_embeds.device)
                #     inputs_embeds = torch.cat([inputs_embeds,padding],dim=-2)
                #     padding = torch.full((bsz,pad_len),1,dtype=attention_mask.dtype,device=attention_mask.device)
                #     attention_mask = torch.cat([attention_mask,padding],dim=-1)
                #     if position_ids is not None:
                #         padding = torch.full((bsz,padding),0,dtype=position_ids.dtype,device=position_ids.device)
                #         position_ids = torch.cat([position_ids,padding],dim=-1)
        if dpo_forward:
            raise NotImplementedError() 
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            #assert attention_mask is None or torch.all(attention_mask)
            attention_mask = None
            num_items_in_batch = None
            if ENFORCE_NUM_ITEMIN_BATCH:
                num_items_in_batch = labels.ne(-100).float().sum()
                num_items_in_batch = reduce(num_items_in_batch)
                num_items_in_batch = num_items_in_batch.long()
            output_hidden_states = True
            output =  super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                prompt_len=prompt_len,
                num_items_in_batch=num_items_in_batch,
                modality_indices= modality_indices
            )
            if skip_batch:
                print(f"SKIP BATCH!!! {dataset_name}")
            if images_gen is not None:
                hidden_states = output.hidden_states[-1]
                gen_hidden_states = hidden_states[new_token_mask_dup]
                gen_hidden_states = maybe_truncate_last_dim(gen_hidden_states,self.config.d_model_gen)
                timesteps = gen_latents_comp_labels_is_mask.sum(-1) / gen_latents_comp_labels_is_mask.shape[-1]
                gen_logits = self.get_model().call_gen_predictor(gen_hidden_states,gen_shape,timesteps)
                # (B L, 8, V)
                #breakpoint()
                gen_targets = gen_latents_comp_labels

                if is_unitok:
                    _b,_d,_l = gen_latents_comp_labels_is_mask.shape
                    gen_logits = gen_logits.view(_b,_l,*gen_logits.shape[-2:]) # B L 8 2064
                    gen_logits = gen_logits.permute(0,2,1,3) # B 8 4096 2064
                    _loss_mask = gen_latents_comp_labels_is_mask.flatten()
                    gen_loss = torch.nn.functional.cross_entropy(gen_logits.flatten(0,2)[_loss_mask],gen_latents_comp_labels.flatten()[_loss_mask])
                else:
                    if image_gen_weight_dup is not None and image_gen_weight_dup.max() > 1:
                        _loss_mask = gen_latents_comp_labels_is_mask.flatten()
                        gen_loss = torch.nn.functional.cross_entropy(gen_logits[_loss_mask].float(),gen_latents_comp_labels.flatten()[_loss_mask],reduction='none') # 5170
                        image_gen_weight_dup_flt = image_gen_weight_dup.flatten(1,) # B L
                        image_gen_weight_dup_flt = image_gen_weight_dup_flt  * gen_latents_comp_labels_is_mask
                        factor_per_sample = gen_latents_comp_labels_is_mask.sum(-1,keepdims=True) #/ (gen_latents_comp_labels_is_mask.sum()+1e-7)
                        loss_weight = image_gen_weight_dup_flt.float()  * (factor_per_sample / image_gen_weight_dup_flt.float().sum(-1,keepdims=True) )
                        loss_weight = loss_weight.flatten()[_loss_mask].to(gen_loss.dtype)
                        gen_loss = (gen_loss * loss_weight).mean()
                    else:
                        _loss_mask = gen_latents_comp_labels_is_mask.flatten()
                        gen_loss = torch.nn.functional.cross_entropy(gen_logits[_loss_mask].float(),gen_targets.flatten()[_loss_mask])
                und_loss = output.loss
                if torch.isnan(und_loss):
                    und_loss = output.logits.mean() * 0
                if IGNORE_TEXT_LOSS:
                    und_loss = und_loss * 0
                    output.loss = gen_loss 
                else:
                    output.loss = und_loss + gen_loss 
                
                output['und_loss'] = und_loss.detach()
                output['gen_loss'] = gen_loss.detach()
                x_0 = gen_logits.argmax(-1)
                x_0 = x_0.view(gen_targets.shape)
                output['gen_x0_gt'] = gen_latents_comp_labels_raw.detach()
                output['gen_x_0_pred'] = x_0.detach()
                output['gen_x_mask'] = gen_latents_comp_labels_is_mask.detach()
                output['new_token_mask_dup'] = new_token_mask_dup.detach()
            elif t2i_inference:
                hidden_states = output.hidden_states[-1]
                gen_hidden_states = hidden_states[new_token_mask_dup]
                gen_logits = self.get_model().gen_predictor(gen_hidden_states)
                #output['gen_logits'] = gen_logits
                final_logits = torch.zeros(*hidden_states.shape[:-1],OFFSET+gen_logits.shape[-1])
                final_logits = final_logits + float('-inf')
                final_logits[...,:output.logits.shape[-1]] = output.logits
                final_logits[...,OFFSET:] = gen_logits
                # breakpoint()
            
            if do_inv:
                new_input_ids = new_input_ids.repeat(2,1)
            output['new_input_ids']=new_input_ids
            output['labels'] = labels
            output['final_masked_indices']=final_masked_indices
            output['p_mask'] = p_mask
            output['do_inv'] = do_inv
            output['skip_batch'] = skip_batch
            
            return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        use_fast_dlm: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            # breakpoint()
            inputs_embeds = self.get_model().embed_tokens(inputs)
        if use_fast_dlm:
            return generate_with_dual_cache(self.get_model(),inputs_embeds=inputs_embeds,position_ids=position_ids,attention_mask=attention_mask,**kwargs)
        return llada_generate(self.get_model(),inputs_embeds=inputs_embeds,position_ids=position_ids,attention_mask=attention_mask,**kwargs)
    
    
    @torch.no_grad()
    def log_likelyhood_inference(
        self,
        inputs: Optional[torch.Tensor] = None,
        answer: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        mc_num=128,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        max_seq_len = 5000
        #if inputs_embeds.shape[1] > max_seq_len:
        max_seq_len = max_seq_len[:,-max_seq_len:]
        answer = answer[:300]
        return get_log_likelihood(self.get_model(), None,inputs_embeds=inputs_embeds, answer=answer, mc_num=mc_num,**kwargs)
        #return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        return llada_generate(self.get_model(),inputs_embeds=inputs_embeds,position_ids=position_ids,attention_mask=attention_mask,**kwargs)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_llada", LlavaLladaConfig)
AutoModelForCausalLM.register(LlavaLladaConfig, LlavaLladaForMaskedDiffusion)

    
    
            
    

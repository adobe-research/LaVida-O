
import os
from llava.constants import MICRO_CONDITION_LABLEL,MAX_CAP_LENGTH

def process_metaquery(data,s3_prefix='',*args,**kwargs):
    source_images_paths = data['source_images_paths']
    tgt_path =  data['tgt_path']
    dir_name = data['dir_name']
    prompt = data['prompt']
    source_images_paths = [os.path.join(s3_prefix,dir_name,x) for x in source_images_paths]
    tgt_path =  os.path.join(s3_prefix,dir_name,tgt_path)
    payload = {
        "id": "000951660",
        "image_gen_enc": source_images_paths,
        "image": source_images_paths,
        "image_gen": tgt_path,
        "pad_image_gen": True,
        "conversations": [
        {
            "from": "human",
            "value": f"<image> <image_gen_enc> {prompt}"
        },
        {
            "from": "gpt",
            "value": "Sure <image_gen>"
        }
        ]
    }
    return payload


def process_uniworld(data,s3_prefix='',pixel_diff=False,do_reweight=False,*args,**kwargs):
    # payload = dict(data)
    total_images = len(data['image']) + len(data['image_gen']) #  + len(data['image_gen_enc'])
    should_pad = data['is_edit'] or total_images >= 2 
    if total_images == 2 and data['is_edit']:
        reweight = True
    else:
        reweight = False
    reweight = reweight and do_reweight
    pixel_diff = pixel_diff
    data['image_gen_enc'] = [os.path.join(data['folder'], img_path) for img_path in data['image_gen_enc']] 
    data['image'] = [os.path.join(data['folder'], img_path) for img_path in data['image']]
    data['image_gen'] = [os.path.join(data['folder'], img_path) for img_path in data['image_gen']]
    payload = {
        "id": "000951660",
        "image_gen_enc": data['image_gen_enc'] if len(data['image_gen_enc'])>0 else None,
        "image": data['image'] if len(data['image'])>0 else None,
        "image_gen": data['image_gen'] if len(data['image_gen'])>0 else None,
        "pad_image_gen": should_pad,
        "reweight": reweight,
        "conversations": data['conversations'],
        'pixel_diff':pixel_diff
    }
    return payload


def process_sharegpt_4o_edit(data,s3_prefix='',no_template=False,reweight=False,t2i=False,*args,**kwargs):
    source_images_paths = data['input_image']
    tgt_path =  data['output_image']
    prompt = data['input_prompt']
    template = '<image> <image_gen_enc> '
    if isinstance(source_images_paths,str):
        source_images_paths = [source_images_paths]
    source_images_paths = [os.path.join(s3_prefix,x) for x in source_images_paths]
    final_template = template * len(source_images_paths)
    tgt_path =  os.path.join(s3_prefix,tgt_path)
    if no_template:
        final_template = ''
    micro = ''
    if t2i:
        micro = f' {MICRO_CONDITION_LABLEL}'
    payload = {
        "id": "000951660",
        "image_gen_enc": source_images_paths,
        "image": source_images_paths,
        "image_gen": tgt_path,
        "pad_image_gen": True,
        "conversations": [
        {
            "from": "human",
            "value": f"{final_template} {prompt}" + micro
        },
        {
            "from": "gpt",
            "value": "Sure <image_gen>"
        }
        ]
    }
    if reweight:
        payload['reweight'] = True
    else:
        payload['reweight'] = False
    return payload


def process_gpt_edit(data,s3_prefix='',no_template=False,reweight=False,t2i=False,pixel_diff=False,*args,**kwargs):
    source_images_paths = data['input_paths']
    tgt_path =  data['output_paths']
    prompt = data['instruction']
    template = '<image> <image_gen_enc> '
    if isinstance(source_images_paths,str):
        source_images_paths = [source_images_paths]
    source_images_paths = [os.path.join(s3_prefix,x) for x in source_images_paths]
    final_template = template * len(source_images_paths)
    tgt_path =  os.path.join(s3_prefix,tgt_path)
    if no_template:
        final_template = ''
    micro = ''
    if t2i:
        micro = f' {MICRO_CONDITION_LABLEL}'
    payload = {
        "id": "000951660",
        "image_gen_enc": source_images_paths,
        "image": source_images_paths,
        "image_gen": tgt_path,
        "pad_image_gen": True,
        "pixel_diff": pixel_diff,
        "conversations": [
        {
            "from": "human",
            "value": f"{final_template} {prompt}" + micro
        },
        {
            "from": "gpt",
            "value": "Sure <image_gen>"
        }
        ]
    }
    if reweight:
        payload['reweight'] = True
    else:
        payload['reweight'] = False
    if 'hpsv2' in data:
        payload['hpsv2'] = [data['hpsv2']] # assert 1 output
    return payload


def process_pica_banana(data,s3_prefix='',no_template=False,reweight=False,t2i=False,pixel_diff=False,*args,**kwargs):
    source_images_paths = data['local_input_image']
    tgt_path =  data['local_output_image']
    prompt = data['text']
    template = '<image> <image_gen_enc> '
    if isinstance(source_images_paths,str):
        source_images_paths = [source_images_paths]
    source_images_paths = [os.path.join(s3_prefix,x) for x in source_images_paths]
    final_template = template * len(source_images_paths)
    tgt_path =  os.path.join(s3_prefix,tgt_path)
    if no_template:
        final_template = ''
    micro = ''
    if t2i:
        micro = f' {MICRO_CONDITION_LABLEL}'
    payload = {
        "id": "000951660",
        "image_gen_enc": source_images_paths,
        "image": source_images_paths,
        "image_gen": tgt_path,
        "pad_image_gen": True,
        "pixel_diff": pixel_diff,
        "conversations": [
        {
            "from": "human",
            "value": f"{final_template} {prompt}" + micro
        },
        {
            "from": "gpt",
            "value": "Sure <image_gen>"
        }
        ]
    }
    if reweight:
        payload['reweight'] = True
    else:
        payload['reweight'] = False
    if 'hpsv2' in data:
        payload['hpsv2'] = [data['hpsv2']] # assert 1 output
    return payload

# def process_gpt_edit_loc(data,s3_prefix='',no_template=False,reweight=False,t2i=False,pixel_diff=False,*args,**kwargs):
#     source_images_paths = data['input_paths']
#     tgt_path =  data['output_paths']
#     prompt = data['instruction']
#     template = '<image> <image_gen_enc> '
#     if isinstance(source_images_paths,str):
#         source_images_paths = [source_images_paths]
#     source_images_paths = [os.path.join(s3_prefix,x) for x in source_images_paths]
#     final_template = template * len(source_images_paths)
#     tgt_path =  os.path.join(s3_prefix,tgt_path)
#     if no_template:
#         final_template = ''
#     micro = ''
#     if t2i:
#         micro = f' {MICRO_CONDITION_LABLEL}'
#     payload = {
#         "id": "000951660",
#         "image_gen_enc": source_images_paths,
#         "image": source_images_paths,
#         "image_gen": tgt_path,
#         "pad_image_gen": True,
#         "pixel_diff": pixel_diff,
#         "conversations": [
#         {
#             "from": "human",
#             "value": f"{final_template} {prompt}" + micro
#         },
#         {
#             "from": "gpt",
#             "value": "Sure <image_gen>"
#         }
#         ]
#     }
#     if reweight:
#         payload['reweight'] = True
#     else:
#         payload['reweight'] = False
#     if 'hpsv2' in data:
#         payload['hpsv2'] = [data['hpsv2']] # assert 1 output
#     return payload




def process_gpt_edit_loc_gnd(data,s3_prefix='',
                             no_template=False,
                             reweight=False,t2i=False,
                             task='add',
                             plan_only=False,
                             pixel_diff=False,
                             *args,**kwargs):
    source_images_paths = data['input_paths']
    tgt_path =  data['output_paths']
    prompt = data['instruction']
    template = '<image> <image_gen_enc> '
    if isinstance(source_images_paths,str):
        source_images_paths = [source_images_paths]
    source_images_paths = [os.path.join(s3_prefix,x) for x in source_images_paths]
    final_template = template * len(source_images_paths)
    tgt_path =  os.path.join(s3_prefix,tgt_path)
    if no_template:
        final_template = ''
    micro = ''
    if t2i:
        micro = f' {MICRO_CONDITION_LABLEL}'
    if 'edit_type' in data:
        task = data['edit_type'] or task
    if task == 'add':
        reason = "I should consider adding the objects in these locations : "
    elif task == 'remove':
        reason = "I should consider removing the objects in these locations : "
    elif task in ['swap','replace']:
        reason = "I should consider replacing the objects in these locations : "
    elif task in ['adjust']:
        reason = "I should consider adjusting the objects in these locations : "
    else:
        reason = "I should consider edit the objects in these locations : "
    loc_str = data['grounding_text']
    img_token = '<image_gen_fake>' if plan_only else '<image_gen>' 
    payload = {
        "id": "000951660",
        "image_gen_enc": source_images_paths,
        "image": source_images_paths,
        "pad_image_gen": True,
        "pixel_diff": pixel_diff,
        "conversations": [
        {
            "from": "human",
            "value": f"{final_template} {prompt}. Please first think and plan the layout in LOC format. " + micro
        },
        {
            "from": "gpt",
            "value": f"Sure {reason} {loc_str} {img_token}"
        }
        ]
    }
    if not plan_only:
        payload.update({ "image_gen": tgt_path,"do_not_mask_text":True})
    # if reweight:
    #     payload['reweight'] = True
    # else:
    #     payload['reweight'] = False
    if 'hpsv2' in data:
        payload['hpsv2'] = [data['hpsv2']] # assert 1 output
    return payload
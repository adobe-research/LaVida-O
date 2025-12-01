import os
# MAX_CAP_LENGTH = 500
from llava.constants import MICRO_CONDITION_LABLEL,MAX_CAP_LENGTH
from llava.train.data.utils import randomly_select_sentences
import random
import numpy as np
def preproces_text_to_image_generation_s3(
    data,random_truncate=True,caption_key='caption',no_drop_keys=None,random_caption_keys=None,*args,**kwargs
):
    img_path = data['s3_path']
    if random_caption_keys is not None:
        caption_key = np.random.choice(random_caption_keys)
    if caption_key not in data:
        print(f"Warning: Caption key '{caption_key}' not found in data. Available keys: {list(data.keys())}")
        raise ValueError(f"Caption key '{caption_key}' not found in data.")
    caption = data[caption_key]
    do_random_truncate = random_truncate
    if no_drop_keys is not None: # higher priority
        do_random_truncate = caption_key not in no_drop_keys
    if random.random() < 0.2 and len(caption) > 150 and do_random_truncate: 
        caption = randomly_select_sentences(caption)
    if len(caption) > MAX_CAP_LENGTH:
        caption = caption[:MAX_CAP_LENGTH]
    payload = {
        "id": "000951660",
        "image_gen": img_path,
        "conversations": [
        {
            "from": "human",
            "value": f"Generate an image with the caption: {caption} {MICRO_CONDITION_LABLEL}"
        },
        {
            "from": "gpt",
            "value": "Sure <image_gen>"
        }
        ]
    }
    if 'fltLaionAesthScore' in data:
        payload['gen_score'] = [data['fltLaionAesthScore']]
    if 'hpsv2' in data:
        payload['hpsv2'] = [data['hpsv2']]
    return payload



def preproces_text_to_image_generation_s3_loc(
    data,random_truncate=True,caption_key='caption',no_drop_keys=None,random_caption_keys=None,plan_only=False,
    add_feedback=False,s3_prefix_enc=False,max_reflection=5,do_print=False,no_plan=False,*args,**kwargs
):
    img_path = data['s3_path']
    if random_caption_keys is not None:
        caption_key = np.random.choice(random_caption_keys)
    if caption_key not in data:
        print(f"Warning: Caption key '{caption_key}' not found in data. Available keys: {list(data.keys())}")
        raise ValueError(f"Caption key '{caption_key}' not found in data.")
    caption = data[caption_key]
    do_random_truncate = random_truncate
    if no_drop_keys is not None: # higher priority
        do_random_truncate = caption_key not in no_drop_keys
    if random.random() < 0.2 and len(caption) > 150 and do_random_truncate: 
        caption = randomly_select_sentences(caption)
    if len(caption) > MAX_CAP_LENGTH:
        caption = caption[:MAX_CAP_LENGTH]
    loc_str = data['grounding_text']
    if add_feedback:
        img_path_enc = data['negatives'].split(';')
        negative_reasons = data['negative_reasons'].split(';')
        max_choices = min(len(img_path_enc),max_reflection)
        n_choices = np.random.randint(1,max_choices+1)
        choices = np.random.choice(range(len(img_path_enc)),n_choices)

        img_path_enc = np.array(img_path_enc)[choices].tolist()
        negative_reasons = np.array(negative_reasons)[choices].tolist()

        if s3_prefix_enc:
            img_path_enc = [os.path.join(s3_prefix_enc,x) for x in img_path_enc]
        feedback_str = [f'Generation {i+1}: <image>\n Feedback {i+1}: {feedback}' for i,feedback in enumerate(negative_reasons)]
        feedback_str = '\n'.join(feedback_str)
        feedback_str = f' Please also consider past generations and their feedbacks. Do not repeat these errors {feedback_str}'

    else:
        img_path_enc = None
        feedback_str = ''
    if plan_only:
        payload = {
            "id": "000951660",
            "conversations": [
            {
                "from": "human",
                "value": f"Generate an image with the caption: {caption} {MICRO_CONDITION_LABLEL}. Please first think and plan the layout in LOC format.{feedback_str}"
            },
            {
                "from": "gpt",
                "value": f"Sure I should place the objects in the following manner {loc_str} <image_gen_fake>"
            }
            ]
        }
    elif no_plan:
        payload = {
            "id": "000951660",
            "image_gen": img_path,
            "conversations": [
            {
                "from": "human",
                "value": f"Generate an image with the caption: {caption} {MICRO_CONDITION_LABLEL}. {feedback_str}"
            },
            {
                "from": "gpt",
                "value": f"<image_gen>"
            },
            ]
        }
    else:
        payload = {
            "id": "000951660",
            "image_gen": img_path,
            "do_not_mask_text": True,
            "conversations": [
            {
                "from": "human",
                "value": f"Generate an image with the caption: {caption} {MICRO_CONDITION_LABLEL}. Please first think and plan the layout in LOC format.{feedback_str}"
            },
            {
                "from": "gpt",
                "value": f"Sure I should place the objects in the following manner {loc_str} <image_gen>"
            },
            ]
        }
    if img_path_enc is not None:
        payload['image'] = img_path_enc
    if 'fltLaionAesthScore' in data:
        payload['gen_score'] = [data['fltLaionAesthScore']]
    if 'hpsv2' in data:
        payload['hpsv2'] = [data['hpsv2']]
    if do_print:
        print(payload)
    return payload


def preproces_text_to_image_generation_piat_hash(
    data,caption_key='longLLA_captions',random_truncate=True,*args,**kwargs
):
    img_path = data['strImagehash']
    if caption_key not in data:
        print(f"Warning: Caption key '{caption_key}' not found in data. Available keys: {list(data.keys())}")
        raise ValueError(f"Caption key '{caption_key}' not found in data.")
    caption = data[caption_key]
    if random.random() < 0.2 and len(caption) > 150 and random_truncate:
        caption = randomly_select_sentences(caption)
    if len(caption) > MAX_CAP_LENGTH:
        caption = caption[:MAX_CAP_LENGTH]
    payload = {
        "id": "000951660",
        "image_gen": "piat://"+img_path,
        "conversations": [
        {
            "from": "human",
            "value": f"Generate an image with the caption: {caption} {MICRO_CONDITION_LABLEL}"
        },
        {
            "from": "gpt",
            "value": "Sure <image_gen>"
        }
        ]
    }
    if 'fltLaionAesthScore' in data:
        payload['gen_score'] = [data['fltLaionAesthScore']]
    if 'hpsv2' in data:
        payload['hpsv2'] = [data['hpsv2']]
    return payload



def preproces_text_to_image_generation_layout_sam(
    data,random_truncate=False,caption_key='caption',caption_drop=0.1,plan_only=False,
    s3_prefix=None,do_print=False,no_plan=False,*args,**kwargs
):
    global_caption = data['caption']
    img_path = data['s3_path']
    if s3_prefix:
        img_path = os.path.join(s3_prefix, img_path)

    caption = data[caption_key]
    if len(caption) > MAX_CAP_LENGTH:
        caption = caption[:MAX_CAP_LENGTH]
    loc_str = data['cap_short']
    detail_caption =  np.random.rand() < 0.4
    feedback_str = ''
    if detail_caption:
        feedback_str = f"Please plan each region in detail by providing individual descriptions. "
        loc_str = data['cap_detail']
    if np.random.rand() < caption_drop:
        caption = 'an image with region captions.' 

    img_path_enc = None

    if plan_only:
        payload = {
            "id": "000951660",
            "conversations": [
            {
                "from": "human",
                "value": f"Generate an image with the caption: {caption} {MICRO_CONDITION_LABLEL}. Please first think and plan the layout in LOC format.{feedback_str}"
            },
            {
                "from": "gpt",
                "value": f"Sure I should place the objects in the following manner {loc_str} <image_gen_fake>"
            }
            ]
        }
    elif no_plan:
        payload = {
            "id": "000951660",
            "image_gen": img_path,
            "conversations": [
            {
                "from": "human",
                "value": f"Generate an image with the caption: {caption} {MICRO_CONDITION_LABLEL}. {feedback_str}"
            },
            {
                "from": "gpt",
                "value": f"<image_gen>"
            },
            ]
        }
    else:
        payload = {
            "id": "000951660",
            "image_gen": img_path,
            "do_not_mask_text": True,
            "conversations": [
            {
                "from": "human",
                "value": f"Generate an image with the caption: {caption} {MICRO_CONDITION_LABLEL}. Please first think and plan the layout in LOC format.{feedback_str}"
            },
            {
                "from": "gpt",
                "value": f"Sure I should place the objects in the following manner {loc_str} <image_gen>"
            },
            ]
        }
    if img_path_enc is not None:
        payload['image'] = img_path_enc
    if 'fltLaionAesthScore' in data:
        payload['gen_score'] = [data['fltLaionAesthScore']]
    if 'hpsv2' in data:
        payload['hpsv2'] = [data['hpsv2']]
    payload['crop_parms'] = [data['crop_parms']]
    if do_print:
        print(payload)
    return payload

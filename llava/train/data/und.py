import os
# MAX_CAP_LENGTH = 500
from llava.constants import MICRO_CONDITION_LABLEL,MAX_CAP_LENGTH
from llava.train.data.utils import randomly_select_sentences
import random
import json
import copy
def preproces_und_mammoth_si_10M(
    data,s3_prefix='',*args,**kwargs
):
    img_path = data['image']
    conversations = json.loads(data['conversations'])
    payload = {
        "id": "000951660",
        "conversations": conversations
    }
    if img_path is not None:
        s3_path = os.path.join(s3_prefix, img_path)
        payload["image"] = s3_path
    return payload


def preproces_vw_instruct(
    data,s3_prefix='',*args,**kwargs
):
    images = data['image']
    conversations =  copy.deepcopy(data['conversations'])
    if len(conversations[1]['value'].split()) > 150:
        conversations[0]['value'] = conversations[0]['value'] + "\n Please also perform detailed reasoning for this problem and provide detailed reasoning traces."
    payload = {
        "id": "000951660",
        "conversations": conversations
    }
    if images is not None:
        payload["image"] = [ os.path.join(s3_prefix, img_path) for img_path in images ]
    return payload

def preproces_grandf(data,s3_prefix='',*args,**kwargs):
    img_path = data['s3_path']
    caption = data['loc_caption']

    payload = {
        "id": "000951660",
        "image": img_path,
        "conversations": [
        {
            "from": "human",
            "value": f"<image>\n Generate a dense image caption with bounding boxes in LOC format."
        },
        {
            "from": "gpt",
            "value": f"{caption}"
        }
        ]
    }
    return payload



def preproces_refcoco_rec(data,s3_prefix='',*args,**kwargs):
    img_path = data['s3_path']
    expression = data['label']
    loc_string = data['loc_string']
    assert isinstance(expression,str)
    assert isinstance(loc_string,str)

    payload = {
        "id": "000951660",
        "image": img_path,
        "conversations": [
        {
            "from": "human",
            "value": f"<image>\n Please locate {expression} in this image. Give bounding boxes in LOC format."
        },
        {
            "from": "gpt",
            "value": f"{loc_string}"
        }
        ]
    }
    return payload





def preprocess_reflection_und(
    data,s3_prefix='',*args,**kwargs
):
    img_path = data['image']
    result = data['results']
    prompt = data['prompts']
    payload = {
        "id": "000951660",
        "conversations": [
        {
            "from": "human",
            "value": f"<image>\nPlease evaluate this generated image based on the following prompt: {prompt} Focus on text alignment and compositionality."
        },
        {
            "from": "gpt",
            "value": f"{result}"
        }
        ]
    }
    if img_path is not None:
        s3_path = os.path.join(s3_prefix, img_path)
        payload["image"] = s3_path
    return payload

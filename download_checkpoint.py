import os
os.environ['DEBUG_FIX_PADDING'] = '1'
os.environ['NOT_ALWASY_DO_2DPOOL'] = '1'
from llava.eval.predict_t2i_edit import text_to_image, build_model,t2i_prompts
from llava.eval.predict_t2i_edit import create_plan,get_feedback,create_plan_edit
import os
from llava.model.utils import maybe_truncate_last_dim,pad_along_last_dim
from PIL import Image
from llava.mm_utils import resize_and_center_crop
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
import yaml
import torch
from llava.eval.predict_grounding import predict_grounding
from llava.eval.demo_utils import  visualize_boxes
import copy
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX,SKIP_DOWN_SAMPLE
from llava.conversation import conv_templates, SeparatorStyle
from matplotlib import pyplot as plt

pretrained = 'jacklishufan/LaViDa-O-v1.0'
tokenizer, model, image_processor = build_model(pretrained=pretrained)
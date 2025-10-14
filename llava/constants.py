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

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN_INDEX_GEN = -300
IMAGE_TOKEN_INDEX_GEN_ENC = 126090 # <|reserved_token_6|>
IMAGE_TOKEN_INDEX_GEN_XTD_RESERVE = 126092 # <|reserved_token_8|>
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_GEN_TOKEN = "<image_gen>"
DEFAULT_IMAGE_GEN_TOKEN_XTD = "<image_gen_xtd>"
DEFAULT_IMAGE_GEN_TOKEN_FAKE= "<image_gen_fake>" # reserved_token_7
DEFAULT_IMAGE_GEN_TOKEN_FAKE_REPL = '<|reserved_token_7|>'
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

import os

IGNORE_TEXT_LOSS = os.environ.get('IGNORE_TEXT_LOSS',None)
SKIP_DOWN_SAMPLE = os.environ.get('SKIP_DOWN_SAMPLE',None)
MICRO_CONDITION_LABLEL = '[[micro_conds_texts]]'

MAX_CAP_LENGTH = os.environ.get('MAX_CAP_LENGTH',800)
MAX_CAP_LENGTH = int(MAX_CAP_LENGTH) # num characters

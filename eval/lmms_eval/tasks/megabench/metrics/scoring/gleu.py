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

from numbers import Number

import jieba
from nltk.translate.gleu_score import sentence_gleu


class GLEUChinese:
    """Compute GLEU score for Chinese text."""

    @staticmethod
    def match(response, correct_answer) -> Number:
        """Compute the BLEU scores between two strings."""
        if isinstance(response, str) and isinstance(correct_answer, str):
            reference_tokens = list(jieba.cut_for_search(response))
            translation_tokens = list(jieba.cut_for_search(correct_answer))
        else:
            return 0
        return sentence_gleu([reference_tokens], translation_tokens)

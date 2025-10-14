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

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge


def get_requery_score(prediction, gt):
    score_dict = dict()

    # 计算BLEUBLEU分数
    smoothing_function = SmoothingFunction().method1  # * used to deal with non-overlap n-gram

    # calculate BLEU-1 score with smoothing function
    bleu_score = sentence_bleu([gt.split()], prediction.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing_function)

    # ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(prediction, gt)[0]
    rouge_l_f1 = rouge_scores["rouge-l"]["f"]

    score_dict["bleu"] = bleu_score
    score_dict["rouge_l"] = rouge_l_f1
    score_dict["score"] = (bleu_score + rouge_l_f1) / 2

    return score_dict

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

import json

from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.synthdog.donut_evaluator import JSONParseEvaluator

evaluator = JSONParseEvaluator()


def synthdog_doc_to_visual(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [doc["image"].convert("RGB")]


def synthdog_doc_to_target(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [json.loads(doc["ground_truth"])["gt_parse"]["text_sequence"]]


def synthdog_process_results(doc, results):
    pred = {"output": results[0].lower().strip()}
    gt_ans = json.loads(doc["ground_truth"])["gt_parse"]

    predictions = []
    ground_truths = []
    accs = []

    score = evaluator.cal_acc(pred, gt_ans)

    accs.append(score)

    predictions.append(pred)
    ground_truths.append(gt_ans)

    return {
        "tree_edit_distance": {"score": score, "prediction": pred, "ground_truth": gt_ans},
    }


def synthdog_aggregate_ted(results, args):
    final_score = 0
    for result in results:
        final_score += result["score"]
    return final_score

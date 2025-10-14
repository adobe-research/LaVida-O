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
import os


def save_result_to_cache(doc, round_res, previous_round_info, save_dir):
    save_dict = dict(
        sample_id=doc["sample_id"],
        query=doc["query"],
        round_res=round_res,
    )
    save_dict.update(previous_round_info)
    json.dump(save_dict, open(os.path.join(save_dir, f"{save_dict['sample_id']}.json"), "w"), indent=4)

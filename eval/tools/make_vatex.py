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

from datasets import Dataset, load_dataset

with open("data/vatex_public_test_english_v1.1.json", "r") as f:
    data = json.load(f)

for da in data:
    da["url"] = "https://www.youtube.com/watch?v=" + da["videoID"]

vatex_dataset = Dataset.from_list(data)
# vatex_dataset.rename_columns({
#     'videoID': 'video_name',
#     'enCap': 'caption'
# }) #if change name is needed
hub_dataset_path = "lmms-lab/vatex_from_url"

vatex_dataset.push_to_hub(repo_id=hub_dataset_path, split="test", config_name="vatex_test", token=True)

with open("data/vatex_validation_v1.0.json", "r") as f:
    data = json.load(f)
for da in data:
    da["url"] = "https://www.youtube.com/watch?v=" + da["videoID"]

vatex_dataset = Dataset.from_list(data)
# vatex_dataset.rename_columns({
#     'videoID': 'video_name',
#     'enCap': 'caption'
# }) #if change name is needed
hub_dataset_path = "lmms-lab/vatex_from_url"

vatex_dataset.push_to_hub(repo_id=hub_dataset_path, split="validation", config_name="vatex_val_zh", token=True)

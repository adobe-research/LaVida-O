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

import os

import yaml

# splits = ["train", "test", "val", "testA", "testB"]
splits = ["test", "val", "testA", "testB"]
tasks = ["seg", "bbox"]

if __name__ == "__main__":
    dump_tasks = []
    for task in tasks:
        for split in splits:
            yaml_dict = {"group": f"refcoco_{task}", "task": f"refcoco_{task}_{split}", "test_split": split, "include": f"_default_template_{task}_yaml"}
            if split == "train":
                yaml_dict.pop("group")
            else:
                dump_tasks.append(f"refcoco_{task}_{split}")

            save_path = f"./refcoco_{task}_{split}.yaml"
            print(f"Saving to {save_path}")
            with open(save_path, "w") as f:
                yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    group_dict = {"group": "refcoco", "task": dump_tasks}

    with open("./_refcoco.yaml", "w") as f:
        yaml.dump(group_dict, f, default_flow_style=False, indent=4)

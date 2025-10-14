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

if __name__ == "__main__":
    path = "/data/pufanyi/project/lmms-eval/tools/temp/2024-09"
    new_path = "/data/pufanyi/project/lmms-eval/tools/temp/processed_images_2"
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    subjects = os.listdir(path)
    for subject in subjects:
        subject_folder = os.path.join(path, subject)
        if not os.path.isdir(subject_folder):
            continue
        print(f"Processing {subject_folder}")
        images = os.listdir(subject_folder)
        for id, image in enumerate(images):
            image_ext = image.split(".")[-1]
            new_image_name = f"{subject}_{id}.{image_ext}"
            os.rename(os.path.join(subject_folder, image), os.path.join(new_path, new_image_name))

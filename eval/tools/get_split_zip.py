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
import zipfile


def split_zip(input_zip, output_dir, max_size=5 * 1024**3):  # 5GB
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    part = 1
    current_size = 0
    prefix_name = input_zip.split(".")[0]
    output_zip = zipfile.ZipFile(os.path.join(output_dir, f"{prefix_name}_part_{part}.zip"), "w", zipfile.ZIP_DEFLATED)

    with zipfile.ZipFile(input_zip, "r") as zip_ref:
        for file in zip_ref.namelist():
            file_data = zip_ref.read(file)
            file_size = len(file_data)

            if current_size + file_size > max_size:
                output_zip.close()
                part += 1
                current_size = 0
                output_zip = zipfile.ZipFile(os.path.join(output_dir, f"{prefix_name}_part_{part}.zip"), "w", zipfile.ZIP_DEFLATED)

            output_zip.writestr(file, file_data)
            current_size += file_size

    output_zip.close()


# Usage
split_zip("Charades_v1_480.zip", "split_zips")

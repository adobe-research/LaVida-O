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

from live_bench import LiveBench
from live_bench.websites import load_websites, load_websites_from_file

if __name__ == "__main__":
    website = load_websites()
    dataset = LiveBench(name="2024-09")

    website = load_websites_from_file("/data/pufanyi/project/lmms-eval/tools/temp/processed_images/selected")
    dataset.capture(websites=website, screen_shoter="human", qa_generator="claude", scorer="claude", checker="gpt4v", driver_kwargs={}, shoter_kwargs={}, generator_kwargs={})
    dataset.upload()

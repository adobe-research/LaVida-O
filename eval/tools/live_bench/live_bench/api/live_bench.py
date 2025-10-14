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

from live_bench.data_generator import LiveBench
from live_bench.websites import load_websites, load_websites_from_file


def generate_live_bench(*, force_clear=False, screen_shoter="single_screen", qa_generator="gpt4v", scorer="gpt4v", checker="gemini", driver_kwargs={}, shoter_kwargs={}, generator_kwargs={}):
    website = load_websites()
    dataset = LiveBench(force_clear=force_clear)
    dataset.capture(websites=website, screen_shoter=screen_shoter, qa_generator=qa_generator, scorer=scorer, checker=checker, driver_kwargs=driver_kwargs, shoter_kwargs=shoter_kwargs, generator_kwargs=generator_kwargs)
    dataset.upload()


def generate_live_bench_from_path(path, *, force_clear=False, qa_generator="gpt4v", scorer="gpt4v", checker="gemini", driver_kwargs={}, shoter_kwargs={}, generator_kwargs={}):
    website = load_websites_from_file(path)
    dataset: LiveBench = LiveBench(force_clear=force_clear)
    dataset.capture(websites=website, screen_shoter="human", qa_generator=qa_generator, scorer=scorer, checker=checker, driver_kwargs=driver_kwargs, shoter_kwargs=shoter_kwargs, generator_kwargs=generator_kwargs)
    dataset.upload()


if __name__ == "__main__":
    generate_live_bench()

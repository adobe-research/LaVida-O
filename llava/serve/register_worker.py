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

"""
Manually register workers.

Usage:
python3 -m fastchat.serve.register_worker --controller http://localhost:21001 --worker-name http://localhost:21002
"""

import argparse

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str)
    parser.add_argument("--worker-name", type=str)
    parser.add_argument("--check-heart-beat", action="store_true")
    args = parser.parse_args()

    url = args.controller_address + "/register_worker"
    data = {
        "worker_name": args.worker_name,
        "check_heart_beat": args.check_heart_beat,
        "worker_status": None,
    }
    r = requests.post(url, json=data)
    assert r.status_code == 200

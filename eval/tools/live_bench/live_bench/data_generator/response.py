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

class Response(object):
    def __init__(self, success: bool, content: str, full_log: dict):
        self.success = success
        self.content = content
        self.full_log = full_log

    def to_dict(self):
        return {
            "success": self.success,
            "content": self.content,
            "full_log": self.full_log,
        }

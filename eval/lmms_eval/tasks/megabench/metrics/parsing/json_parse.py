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

from metrics.parsing.common.parsers import parse_json
from metrics.parsing.common.utils import evaluate_as_string


class JsonParse:
    """Load the response as a JSON object."""

    @staticmethod
    def parse(response: str):
        """Parse the JSON object, including nested JSON strings."""
        parsed_res = parse_json(response)
        # Drop the potentially duplicated string quotes
        if isinstance(parsed_res, dict):
            for key, val in parsed_res.items():
                parsed_res[key] = evaluate_as_string(val)

        return parsed_res

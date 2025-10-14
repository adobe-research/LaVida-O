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

from metrics.scoring.exact_str_match import ExactStrMatch


class SimpleStrMatch:
    """Basic string matching, without spaces or hyphens."""

    @staticmethod
    def match(response, correct_answer: str) -> int:
        """Simple string match between response and correct_answer."""
        if not isinstance(response, str):
            response = str(response)  # If it is JSON-like
        response = response.replace(" ", "").replace("-", "").replace("\n", "").replace("\t", "").replace(".", "").lower()
        correct_answer = correct_answer.replace(" ", "").replace("-", "").replace("\n", "").replace("\t", "").replace(".", "").lower()

        return ExactStrMatch.match(response, correct_answer)

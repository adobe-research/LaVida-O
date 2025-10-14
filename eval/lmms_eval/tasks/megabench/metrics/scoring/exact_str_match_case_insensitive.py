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


class ExactStrMatchCaseInsensitive:
    """Case-insensitive exact string matching."""

    @staticmethod
    def match(response, correct_answer) -> int:
        """Case-insensitive exact match between targets and responses."""
        if not isinstance(response, str) and isinstance(correct_answer, str):
            return 0
        return ExactStrMatch.match(response.lower(), correct_answer.lower())

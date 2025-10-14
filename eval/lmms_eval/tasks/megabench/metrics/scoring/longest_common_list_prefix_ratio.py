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

from metrics.scoring.common.conversions import str_to_list
from metrics.scoring.common.metrics import longest_common_prefix


class LongestCommonListPrefixRatio:
    """Determines how much of the first part of the list
    was predicted correctly.
    """

    @classmethod
    def match(cls, responses, targets) -> int:
        """Exact match between targets and responses."""
        responses = str_to_list(responses)
        targets = str_to_list(targets)
        return len(longest_common_prefix(responses, targets)) / len(targets)

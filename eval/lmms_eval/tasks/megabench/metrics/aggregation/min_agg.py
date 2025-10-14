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

from numbers import Number
from typing import Dict


class MinAggregation:
    """Take the minimum of all valid scores."""

    @staticmethod
    def aggregate(scores: Dict[str, Number], weights: Dict[str, Number]) -> Number:
        """Exact match between targets and responses."""
        filtered_scores = [s for s in scores.values() if s >= 0]
        if not filtered_scores:
            return -1
        return min(filtered_scores)

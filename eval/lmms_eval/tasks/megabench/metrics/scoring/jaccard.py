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

from metrics.scoring.common.conversions import cast_to_set
from metrics.scoring.common.metrics import jaccard_index


class Jaccard:
    """Calculates the Jacard index for iterables."""

    @classmethod
    def match(cls, responses, targets) -> float:
        """Exact match between targets and responses."""
        if responses is None:
            return 0
        responses = cast_to_set(responses)
        targets = cast_to_set(targets)

        return jaccard_index(responses, targets)


class JaccardCaseInsensitive:
    """Calculates the Jacard index for iterables of strings,
    Do not consider the case
    """

    @classmethod
    def match(cls, responses, targets) -> float:
        """Exact match between targets and responses."""
        if responses is None:
            return 0
        responses = cast_to_set(responses)
        targets = cast_to_set(targets)

        if isinstance(list(targets)[0], str):
            new_responses = {item.lower() if isinstance(item, str) else str(item).lower() for item in responses}
            new_targets = {item.lower() for item in targets}
        elif isinstance(list(targets)[0], tuple):
            new_responses = set()
            new_targets = set()
            try:
                for res in responses:
                    new_res = tuple([item.lower().replace(" ", "").replace("-", "").replace("\n", "").replace("\t", "").replace("_", "").replace(".", "") for item in res])
                    new_responses.add(new_res)
            except:  # the data type of the response might be wrong, return 0 in this case
                return 0
            for tgt in targets:
                new_tgt = tuple([item.lower().replace(" ", "").replace("-", "").replace("\n", "").replace("\t", "").replace("_", "").replace(".", "") for item in tgt])
                new_targets.add(new_tgt)
        else:
            return 0

        return jaccard_index(new_responses, new_targets)

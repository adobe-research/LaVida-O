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

import logging
from typing import Any, Dict

from metrics.scoring.common.conversions import str_to_set
from metrics.scoring.common.metrics import jaccard_index


def chess_transform(move_sequence: str) -> set:
    """Transform a sequence of chess moves encoded in SAN into a set."""
    move_sequence = str_to_set(move_sequence)
    return {move_san.removesuffix("!").removesuffix("#") for move_san in move_sequence}


class ChessMoveJaccard:
    """Calculates the Jacard index for chess moves."""

    @classmethod
    def match(cls, responses: str | None, targets: str) -> float:
        """Exact match between targets and responses."""
        if responses is None:
            return 0
        responses = chess_transform(responses)
        targets = chess_transform(targets)

        return jaccard_index(responses, targets)

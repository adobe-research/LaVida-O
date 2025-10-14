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

import rapidfuzz
import unidecode
from metrics.scoring.common.transformations import remove_def_indef_articles


def approximate(text: str) -> str:
    """Return an approximation of the original string."""
    return unidecode.unidecode(remove_def_indef_articles(text)).lower()


class NearStrMatch:
    """Near string matching."""

    @staticmethod
    def match(response, correct_answer: str, threshold=0.9) -> int:
        """Simple string match between response and correct_answer."""
        if not isinstance(response, str) or not isinstance(correct_answer, str):
            return 0
        response = approximate(response)
        correct_answer = approximate(correct_answer)
        return rapidfuzz.distance.DamerauLevenshtein.normalized_similarity(response, correct_answer, score_cutoff=threshold)

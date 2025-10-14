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


class NormalizedSimilarityDamerauLevenshtein:
    """Normalized Damerau-Levenshtein Similarity."""

    @staticmethod
    def match(response, correct_answer) -> int:
        """Normalized indel similarityuiio do between targets and responses."""
        if not isinstance(response, str) and isinstance(correct_answer, str):
            return 0
        return rapidfuzz.distance.DamerauLevenshtein.normalized_similarity(response, correct_answer)

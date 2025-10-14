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

import sacrebleu


class Bleu:
    """Compute BLEU score, using SacreBLEU."""

    @staticmethod
    def match(response, correct_answer) -> Number:
        """Compute the BLEU scores between two strings."""
        if isinstance(response, str) and isinstance(correct_answer, str):
            resp = [response]
            corr = [correct_answer]
        elif isinstance(response, (list, tuple)) and isinstance(correct_answer, (list, tuple)):
            resp = tuple(response)
            corr = tuple(correct_answer)
        else:
            return 0
        result = sacrebleu.corpus_bleu(corr, [resp]).score / 100
        return result

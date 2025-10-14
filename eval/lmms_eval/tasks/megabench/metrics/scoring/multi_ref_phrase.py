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

from metrics.scoring.common.conversions import str_to_iterable
from metrics.scoring.simple_str_match import SimpleStrMatch


def replace_potential_chinese_comma(input_string):
    return input_string.replace("，", ",")


class MultipleReferencePhraseEval:
    """
    Check the response with multiple correct references
    As long as one is matched, the score is 1, otherwise the score is 0
    """

    @staticmethod
    def match(response, targets) -> Number:
        targets = replace_potential_chinese_comma(targets)
        refs = str_to_iterable(list, targets)
        matched = False
        for ref in refs:
            str_ref = ref if isinstance(ref, str) else str(ref)
            if SimpleStrMatch.match(response, str_ref):
                matched = True
                break
        return 1 if matched else 0

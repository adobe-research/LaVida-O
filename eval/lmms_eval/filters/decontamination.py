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

from lmms_eval.api.filter import Filter


class DecontaminationFilter(Filter):
    """
    A filter which evaluates
    """

    name = "track_decontamination"

    def __init__(self, path) -> None:
        """

        TODO: make sure only ever run one time on the train set (should this be cached as a class var? keyed by value for "path").
        should further cache result on a given (task_name, doc_id)
        """
        self._decontam_results = None

    def apply(self, resps, docs) -> None:
        """
        Return {"no_contamination", "only_contamination"} keys for the 2 different subsets
        """
        pass

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

import json
import pickle


class PickleWriteable:
    """Mixin for persisting an instance with pickle."""

    def save(self, path):
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        except (pickle.PickleError, OSError) as e:
            raise IOError("Unable to save {} to path: {}".format(self.__class__.__name__, path)) from e

    @classmethod
    def load(cls, path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, OSError) as e:
            raise IOError("Unable to load {} from path: {}".format(cls.__name__, path)) from e

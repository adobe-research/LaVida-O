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

from lmms_eval.filters.extraction import MultiChoiceRegexFilter


def parse_multi_choice_answer(answer):
    # Example responses and documents
    model_responses = [["The answer is (B)", "I believe it is (A)", "(C) seems correct"], ["Answer is: B!", "Answer: B", "Answer: B"]]  # Model response set 1  # Model response set 2

    documents = [{"choices": ["A. Apple", "B. Banana", "C. Cherry"]}, {"choices": ["A. Alpha", "B. Beta", "C. Gamma"]}]  # Multiple choice options for question 1  # Multiple choice options for question 2

    # Instantiate the filter
    multi_choice_filter = MultiChoiceRegexFilter(regex_pattern=r"\(([A-D])\)", group_select=0, ignore_case=False, ignore_punctuation=True)

    filtered_responses = multi_choice_filter.apply(model_responses, documents)

    # Print the filtered answers
    for i, filtered in enumerate(filtered_responses):
        print(f"Question {i+1} filtered responses: {filtered}")


parse_multi_choice_answer("a")

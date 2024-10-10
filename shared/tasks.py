#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Configurable task list."""

import lm.task as task


def get_task(task_name):
    assert task_name == "lm"
    return task

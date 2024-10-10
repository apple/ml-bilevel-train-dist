#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import random
import string

import tensorflow as tf

# global writer / task_id initialized once
_tf_default_writer = None
_uniq_task_id = None


def _generate_random_id(length=10):
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def get_artifacts_dir():
    task_id = get_task_id()
    return os.path.join("artifacts", task_id)


def get_task_id():
    global _uniq_task_id
    if _uniq_task_id is None:
        _uniq_task_id = _generate_random_id()
        os.makedirs(os.path.join("artifacts", _uniq_task_id))
    return _uniq_task_id


def init_tf_default_writer(log_dir=None):
    global _tf_default_writer
    if _tf_default_writer is None:
        if log_dir is None:
            log_dir = get_artifacts_dir()
        _tf_default_writer = tf.summary.create_file_writer(log_dir)


def tf_send_metrics(metric_dict, iteration):
    global _tf_default_writer
    # Initialize writer if not already initialized
    if _tf_default_writer is None:
        init_tf_default_writer()

    # Use the writer to log the metrics
    with _tf_default_writer.as_default():
        for metric_name, metric_value in metric_dict.items():
            tf.summary.scalar(metric_name, metric_value, step=iteration)
        _tf_default_writer.flush()


send_metrics = tf_send_metrics

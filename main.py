#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import shutil

import jax
import tensorflow as tf
from absl import app, flags
from ml_collections import config_flags

from shared import evaluation, sysutils, train

FLAGS = flags.FLAGS

flags.DEFINE_string("datadir", ".", "Directory to load data from.")
flags.DEFINE_boolean("evaluate", False, "Predict loss per point.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)
flags.mark_flags_as_required(["config"])


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # prevent tf from allocating accelerator mem.
    tf.config.experimental.set_visible_devices([], "GPU")

    # save hyperparameter config.
    config_file = flags.FLAGS["config"].config_filename
    work_dir = sysutils.get_artifacts_dir()
    try:
        shutil.copy(config_file, os.path.join(work_dir, "cfg.py"))
    except shutil.SameFileError:
        pass

    config = FLAGS.config

    # copy init checkpoint
    if config.init_checkpoint:
        shutil.copy(
            config.init_checkpoint,
            os.path.join(work_dir, os.path.basename(config.init_checkpoint)),
        )

    if FLAGS.evaluate:
        evaluation.evaluate(config, work_dir, FLAGS.datadir)
    else:
        train.train_and_evaluate(config, work_dir, FLAGS.datadir)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)

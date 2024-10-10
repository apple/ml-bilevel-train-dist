#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import builtins
import hashlib
import json

import numpy as np
from jax._src.lib import xla_client

# Hack to enable pickling of bfloat16
# https://github.com/google/jax/issues/8505
builtins.bfloat16 = xla_client.bfloat16


class FIELDS:
    INPUTS = "inputs"
    POS_INPUTS = "pos_inputs"
    SEG_INPUTS = "seg_inputs"
    TARGETS = "targets"
    POS_TARGETS = "pos_targets"
    SEG_TARGETS = "seg_targets"
    LABELS = "labels"
    LATENTS = "latents"
    IDENTIFIER = "identifier"
    LANG = "language"
    POS_LANG = "pos_language"
    EMBEDDING = "embedding"
    DISTRACTORS = "distractors"
    CLASSES = "classes"
    RNG = "rng"


NON_SELECTABLE_FIELDS = [FIELDS.DISTRACTORS, FIELDS.RNG, FIELDS.CLASSES]


def dataset_size(
    dataset,
):
    return len(dataset[FIELDS.INPUTS])


def random_split(dataset, split_sizes, seed=0):
    keys = list(dataset.keys())
    data_size = len(dataset[keys[0]])

    # random shuffle
    rng = np.random.RandomState(seed=seed)
    shuf = np.arange(data_size)
    rng.shuffle(shuf)

    # sizes to splits:
    # We start with the last split as it allows to have the same examples
    # in the last subset (valid) when we vary the first size of the first one
    # (train).
    output, start = [], 0
    for split_size in reversed(split_sizes):
        idx = shuf[start : start + split_size]
        output.append({k: dataset[k][idx] for k in keys})
        start += split_size

    return list(reversed(output))


def hash_py_structure(inputs):
    enc_str = json.dumps(inputs, sort_keys=True).encode()
    return hashlib.md5(enc_str).hexdigest()


def save_dataset(filename, dataset):
    keys = np.array(list(dataset.keys()))
    with open(filename, "wb") as fout:
        np.save(fout, keys)
        for k in keys:
            np.save(fout, np.array(dataset[k]))


def load_dataset(filename):
    with open(filename, "rb") as fin:
        keys = np.load(fin)
        return {k: np.load(fin) for k in keys}


def save_batched_dataset(filename, dataset):
    """
    Save a batched dataset
    """
    with open(filename, "wb") as fout:
        for batch in dataset:
            for k, v in batch.items():
                batch[k] = np.array(v)
            np.save(fout, batch)


def load_batched_dataset(filename, dataset):
    """
    Load a batched dataset
    """
    ds = []
    with open(filename, "rb") as fin:
        while fin.peek(1):
            ds.append(np.load(fin, allow_pickle=True).item())
        return ds


def train_batches(dataset, batch_size, seed=0, shuffle_once=False):
    """Provide a function returning a random training batch."""
    num_examples = dataset_size(dataset)
    num_batches = num_examples // batch_size
    state = []

    def set_permutation(epoch):
        rng = np.random.RandomState(seed=seed + epoch)
        p = rng.permutation(num_examples)
        p = p[: num_batches * batch_size]  # drop incomplete last batch
        perm = np.reshape(p, (num_batches, batch_size))
        state.clear()
        state.extend((epoch, perm))

    set_permutation(0)

    def get_batch_fn(batch_index):
        prev_epoch, perm = state
        epoch = batch_index // num_batches
        batch_index = batch_index % num_batches
        if (epoch != prev_epoch) and not shuffle_once:
            set_permutation(epoch)
            _, perm = state
        index = perm[batch_index]
        return {k: v[index] for k, v in dataset.items()}

    return get_batch_fn


def eval_batches(dataset, batch_size, drop_remainder=False):
    """Iterator over non-shuffled batches."""

    def fn():
        num_examples = dataset_size(dataset)
        index = 0
        while index < num_examples:
            next_index = index + batch_size
            batch = {k: v[index:next_index] for k, v in dataset.items()}
            if drop_remainder and dataset_size(batch) < batch_size:
                break
            index = next_index
            yield batch

    return fn

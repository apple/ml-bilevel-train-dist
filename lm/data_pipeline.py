#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import csv
import itertools
import os
import sys
from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np
from absl import logging

import lm.sp_tokenization as sp_tokenization
import shared.data_pipeline as dp
import shared.sparse_data_pipeline as sdp

csv.field_size_limit(sys.maxsize)

INPUT_DEFAULT_LENGTH = 64
PAD_INT = 0
FIELDS = dp.FIELDS
TOKENIZER_BATCH_SIZE = 1_000_000


def _read_tsv(filename, quotechar=None):
    """Reads a tab separated value file."""
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        for line in reader:
            yield line


def to_positions(inputs: np.ndarray, dtype=np.uint32):
    positions = np.arange(1, inputs.shape[-1] + 1, dtype=dtype)
    not_padding = (inputs > 0).astype(dtype)
    positions = positions[np.newaxis] * not_padding
    return positions


def input_placeholder(batch_size=1, length=1, int_type=jnp.int32):
    """Placeholder batch for model initialization."""

    input_ph = jnp.ones([batch_size, length], dtype=int_type)
    return {
        FIELDS.INPUTS: input_ph,
        FIELDS.TARGETS: input_ph,
        FIELDS.LABELS: input_ph,
        FIELDS.POS_INPUTS: input_ph,
        FIELDS.POS_TARGETS: input_ph,
        FIELDS.LANG: input_ph,
        FIELDS.POS_LANG: input_ph,
    }


def train_sentence_piece_with_caching(
    data_dir: str,
    input_tsv: str,
    vocabulary_size: int,
):
    args = (
        input_tsv,
        vocabulary_size,
    )
    h = dp.hash_py_structure(args)
    dirname = os.path.join(data_dir, "cache")
    sp_model_path = os.path.join(dirname, os.path.basename(input_tsv) + "_" + h)
    sp_model_path += ".sp_model"

    if not os.path.exists(sp_model_path):
        logging.info("Cached sentence piece model not found: training.")
        os.makedirs(dirname, exist_ok=True)
        sp_tokenization.train_tokenizer(
            itertools.chain.from_iterable(_read_tsv(input_tsv)),
            vocabulary_size,
            sp_model_path,
        )
    logging.info("Sentence piece model available.")

    return sp_model_path


def tokenize_as_bytes(texts):
    tokenize = lambda x: np.frombuffer(x.encode("utf-8"), np.uint8)
    return [tokenize(x) for x in texts]


def detokenize_from_bytes(arr):
    arr = np.array(arr).astype(np.uint8)
    if arr.ndim > 1:
        return [detokenize_from_bytes(x) for x in arr]
    else:
        l = (arr > 0).sum()
        return arr[:l].tobytes().decode("utf-8")


def tokenize_as_sentence_pieces(
    texts,
    sp_tokenizer,
):
    return sp_tokenizer.encode(list(texts), out_type=int)


def detokenize_from_sentence_pieces(arr, sp_tokenizer):
    arr = np.array(arr).astype(np.uint32)
    assert arr.ndim <= 2, "Expect single sequence or a batch of sequences."
    if arr.ndim == 1:
        return detokenize_from_sentence_pieces(arr[np.newaxis], sp_tokenizer)[0]
    arr = [[t for t in l if t > 0] for l in arr.tolist()]  # discard padding
    return sp_tokenizer.decode(arr)


def length_stat_logging(tokenized, field_name):
    max_length = tokenized.shape[-1]
    num_pad = (tokenized == PAD_INT).sum(axis=-1)
    is_max_length = num_pad == 0
    ql99 = np.quantile(max_length - num_pad, 0.99)
    logging.info(
        "  field %s: frac_pad = %.3f "
        + "frac_truncated = %.6f, quantile_length_0.99 = %5.2f",
        field_name,
        num_pad.mean() / max_length,
        is_max_length.mean(),
        ql99,
    )


def load_dataset(
    input_filename: str,
    vocabulary_filename: Optional[str] = None,
    field_names: Sequence[str] = (FIELDS.INPUTS,),
    padded_lengths: Sequence[int] = (INPUT_DEFAULT_LENGTH,),
) -> sdp.SparseDataset:
    if vocabulary_filename:
        # Sentence piece tokenizer
        tokenizer = sp_tokenization.load_tokenizer(vocabulary_filename)
        tokenize = lambda t: tokenize_as_sentence_pieces(
            t, sp_tokenizer=tokenizer
        )
        vocab_size = tokenizer.vocab_size()
        dtype = np.uint16 if vocab_size < 2**16 - 1 else np.uint32
    else:
        # byte tokenizer
        tokenize = tokenize_as_bytes
        dtype = np.uint8

    logging.info("Load tsv data from %s", input_filename)
    builder = sdp.SparseDatasetBuilder(
        field_names=list(field_names) + [dp.FIELDS.IDENTIFIER],
        dtypes=[dtype] * len(field_names) + [np.uint32],
        max_lengths=list(padded_lengths) + [1],
        shapes=[tuple()] * (len(field_names) + 1),
    )

    queue = []  # queue to batch tokenizer request, for speed.
    num_examples = 0
    num_fields = len(field_names)
    for t in itertools.chain(_read_tsv(input_filename), [None]):
        if t is not None:  # still some example left?
            assert len(t) == num_fields
            queue.extend(t)
        # tsv exhausted or full queue.
        if t is None or len(queue) >= TOKENIZER_BATCH_SIZE:
            tokenized = tokenize(queue)
            queue.clear()
            for i in range(len(tokenized) // num_fields):
                t = tokenized[i * num_fields : (i + 1) * num_fields]
                builder.append(t + [[num_examples]])
                num_examples = num_examples + 1
            logging.info("Tokenized %d examples.", num_examples)

    dataset = builder.to_sparse_dataset()
    logging.info("Tokenized data available for %s", input_filename)
    return dataset


def load_dataset_with_caching(
    workdir: str,
    input_filename: str,
    vocabulary_filename: Optional[str] = None,
    field_names: Sequence[str] = (FIELDS.INPUTS,),
    padded_lengths: Sequence[int] = (INPUT_DEFAULT_LENGTH,),
) -> sdp.SparseDataset:
    args = (
        input_filename,
        vocabulary_filename,
        field_names,
        padded_lengths,
    )
    h = dp.hash_py_structure(args)
    dirname = os.path.join(workdir, "cache")
    filename = os.path.join(dirname, os.path.basename(input_filename) + "_" + h)

    if os.path.exists(filename):
        logging.info("Cached data found; loading %s", filename)
        return sdp.SparseDataset.from_file(filename)
    else:
        logging.info("Cached data not found; caching %s", filename)
        os.makedirs(dirname, exist_ok=True)
        dataset = load_dataset(
            input_filename,
            vocabulary_filename,
            field_names=field_names,
            padded_lengths=padded_lengths,
        )
        dataset.to_file(filename)
        logging.info("Cached data at %s", filename)
        return dataset


def insert_and_shift_at_start(x, sos=PAD_INT):
    """Insert padding at the beginning of a sequence."""
    return np.pad(x, ((0, 0), (1, 0)), constant_values=sos)[:, :-1]


def to_language_modeling_batch(inputs, to_accelarator=True):
    """Prepare batch for LM where inputs are the past version of the labels."""
    outputs = dict(inputs)
    outputs[FIELDS.POS_INPUTS] = to_positions(outputs[FIELDS.INPUTS])

    if FIELDS.TARGETS in outputs:  # conditional lm: (inputs, targets) -> labels
        outputs[FIELDS.POS_TARGETS] = to_positions(outputs[FIELDS.TARGETS])
        text = outputs.pop(FIELDS.TARGETS)
        outputs[FIELDS.TARGETS] = insert_and_shift_at_start(text)
    else:  # regular lm: inputs -> labels
        text = outputs.pop(FIELDS.INPUTS)
        outputs[FIELDS.INPUTS] = insert_and_shift_at_start(text)
    outputs[FIELDS.LABELS] = text

    if to_accelarator:
        outputs = {k: jnp.array(v) for k, v in outputs.items()}

    return outputs


def train_batches(
    dataset: sdp.SparseDataset,
    batch_size: int,
    seed: int = 0,
    shuffle_once: bool = False,
):
    fn = dataset.train_batches(batch_size, seed=seed, shuffle_once=shuffle_once)

    def lm_fn(batch_index):
        return to_language_modeling_batch(fn(batch_index))

    return lm_fn


def eval_batches(
    dataset: sdp.SparseDataset,
    batch_size: int,
    drop_remainder: bool = False,
):
    fn = dataset.eval_batches(
        batch_size,
        drop_remainder=drop_remainder,
    )

    def lm_fn():
        for x in fn():
            yield to_language_modeling_batch(x)

    return lm_fn

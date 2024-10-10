#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import functools
import os

import jax.numpy as jnp
from absl import logging
from lm import data_pipeline as lm_dp
from lm import model as mlib
from lm import sp_tokenization
from shared import data_pipeline as dp
from shared import parallel
from shared import sparse_data_pipeline as sdp


class TokenizerType:
    BYTE = "byte"
    SENTENCE_PIECE = "sentence_piece"


def uniform_weights(
    batch,
    dtype=jnp.float32,
    epsilon=1e-6,
    pos_field=dp.FIELDS.POS_INPUTS,
):
    positions = batch[pos_field]
    mask = (positions > 0).astype(dtype)
    return mask / (epsilon + mask.sum(axis=(-2, -1), keepdims=True))


def _sharded_fn(fn):
    """Function with sharded result."""

    def sfn(*args, **kwargs):
        return parallel.shard(fn(*args, **kwargs))

    return sfn


def _sharded_generator(gen_fn):
    """Generator with sharded items."""

    def s_gen(*args, **kwargs):
        for output in gen_fn(*args, **kwargs):
            yield parallel.shard(output)

    return s_gen


def init_data_from_tsv(config, data_dir):
    generic_file = os.path.join(data_dir, config.generic_tsv)
    specific_file = os.path.join(data_dir, config.specific_tsv)
    assert len(config.padded_lengths) == len(config.field_names)

    # Train tokenizer if necessary.
    if config.tokenizer.type == TokenizerType.BYTE:
        vocab_file = ""
        detokenizer = lm_dp.detokenize_from_bytes
    elif config.tokenizer.type == TokenizerType.SENTENCE_PIECE:
        if config.tokenizer.vocabulary:
            # Explitictely provided vocab file.
            vocab_file = os.path.join(data_dir, config.tokenizer.vocabulary)
            assert os.exists(vocab_file), "Vocabulary file missing."
        else:
            # Train from generic data.
            vocab_size = config.num_hidden_units[0]
            vocab_file = lm_dp.train_sentence_piece_with_caching(
                data_dir,
                generic_file,
                vocab_size,
            )
        detokenizer = functools.partial(
            lm_dp.detokenize_from_sentence_pieces,
            sp_tokenizer=sp_tokenization.load_tokenizer(vocab_file),
        )
    else:
        raise ValueError("Unkown tokenizer.")

    logging.info("Loading generic data...")
    g_data = lm_dp.load_dataset_with_caching(
        data_dir,
        generic_file,
        vocab_file,
        field_names=config.field_names,
        padded_lengths=config.padded_lengths,
    )
    g_train, g_eval = sdp.random_split(g_data, config.generic_split)

    logging.info("Loading specific data...")
    s_data = lm_dp.load_dataset_with_caching(
        data_dir,
        specific_file,
        vocab_file,
        field_names=config.field_names,
        padded_lengths=config.padded_lengths,
    )

    s_data = sdp.random_split(s_data, config.specific_split)
    s_meta_train, s_train, _, s_eval = s_data
    if config.specific_split[0] == 0:
        s_meta_train = s_train
    datasets = [g_train, g_eval, s_train, s_meta_train, s_eval]
    dataset_sizes = [d.num_examples for d in datasets]
    logging.info(
        "Loaded generic (train_size=%d, eval_size=%d) "
        + "and specific (train_size=%d, meta_train_size=%d, eval_size=%d).",
        *dataset_sizes,
    )

    generic_train_batch_fn = lm_dp.train_batches(
        g_train,
        config.generic_batch_size,
        seed=config.seed,
        shuffle_once=config.shuffle_once,
    )
    generic_train_batch_fn = _sharded_fn(generic_train_batch_fn)

    specific_train_batch_fn = lm_dp.train_batches(
        s_train,
        config.batch_size,
        seed=config.seed,
        shuffle_once=config.shuffle_once,
    )
    specific_train_batch_fn = _sharded_fn(specific_train_batch_fn)

    meta_batch_size = config.get("meta_batch_size", config.batch_size)
    meta_train_batch_fn = lm_dp.train_batches(
        s_meta_train,
        meta_batch_size,
        seed=config.seed + 1,
        shuffle_once=config.shuffle_once,
    )
    meta_train_batch_fn = _sharded_fn(meta_train_batch_fn)

    # eval sets are triples (is_generic, name, batch_generator)
    eval_sets = []
    if g_eval.num_examples > 0:
        generic_eval_batch_fn = lm_dp.eval_batches(
            g_eval,
            config.batch_size,
            drop_remainder=True,
        )
        generic_eval_batch_fn = _sharded_generator(generic_eval_batch_fn)
        eval_sets.append((True, "generic", generic_eval_batch_fn))

    if s_eval.num_examples > 0:
        specific_eval_batch_fn = lm_dp.eval_batches(
            s_eval,
            config.batch_size,
            drop_remainder=True,
        )
        specific_eval_batch_fn = _sharded_generator(specific_eval_batch_fn)
        eval_sets.append((True, "specific", specific_eval_batch_fn))

    # extra datasets.
    for name, filename, is_generic, size in config.eval_sets:
        logging.info("Loading additional eval data (%s)...", name)
        dset = lm_dp.load_dataset_with_caching(
            data_dir,
            os.path.join(data_dir, filename),
            vocab_file,
            field_names=config.field_names,
            padded_lengths=config.padded_lengths,
        )
        dset = sdp.random_split(dset, (size,))[0]
        logging.info("Loaded %s, %d examples", name, dset.num_examples)
        batch_fn = lm_dp.eval_batches(
            dset, config.batch_size, drop_remainder=True
        )
        batch_fn = _sharded_generator(batch_fn)
        eval_sets.append((is_generic, name, batch_fn))

    return (
        generic_train_batch_fn,
        specific_train_batch_fn,
        meta_train_batch_fn,
        eval_sets,
        detokenizer,
    )


def init_model(rng, config):
    model_state = mlib.init_model_state(rng, config)
    weight_state = mlib.init_weight_state(
        rng,
        config,
        # model state for soba to get shape for linear parameters
        model_state_params=model_state.params,
    )
    return model_state, weight_state, uniform_weights


def init_task(rng, config, workdir):
    assert dp.FIELDS.INPUTS in config.field_names
    (
        generic_train_batch_fn,
        specific_train_batch_fn,
        meta_train_batch_fn,
        eval_sets,
        _,
    ) = init_data_from_tsv(config, workdir)
    model_state, weight_state, weight_fn = init_model(rng, config)
    return (
        model_state,
        weight_state,
        generic_train_batch_fn,
        specific_train_batch_fn,
        meta_train_batch_fn,
        eval_sets,
        weight_fn,
        None,
    )

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Contrastive data selection

as defined in

Denoising Neural Machine Translation Training with
Trusted Data and Online Data Selection, Wei Wang et al 2018.

See config specification in the function doc string.
"""
import functools

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import shared.data_pipeline as dp
from absl import logging
from shared import learning, meta_learning
from shared import model as sm
from shared import parallel


def _overload_config_dict(base_dict, overloads):
    """Overload selected keys in a dictionary"""
    overloads = flax.traverse_util.flatten_dict(flax.core.FrozenDict(overloads))
    output = (
        ml_collections.ConfigDict(base_dict)
        .copy_and_resolve_references()
        .unlock()
    )
    for k_path, new_value in overloads.items():
        d = output
        for k in k_path[:-1]:
            assert k in d, "Overload structure should match the base dictionary"
            d = d[k]
        d[k_path[-1]] = new_value
    return ml_collections.FrozenConfigDict(output)


def _generic_loss_per_example(
    model_state,
    default_weight_fn,
    train_batch_fn,
    num_sub_batches=1,  # optionally break batches in sub-pieces
):
    """Score the training set and return a mapping ids: scores"""
    model_state = parallel.replicate(model_state)
    generic_fn = learning.apply_model(
        model_state.eval_generic_fn,  # this assume no packing.
        loss_fn=model_state.generic_loss_fn,
        need_gradient=False,
        weight_fn=default_weight_fn,
        reduce=False,
    )
    generic_fn = parallel.pmap(generic_fn, axis_name="batch")

    def loss_per_sample(batch):
        loss, _, weights = generic_fn(model_state.params, parallel.shard(batch))
        loss = loss * weights
        while loss.ndim > 2:
            loss = loss.sum(axis=-1)
            weights = weights.sum(axis=-1)
        return np.array(parallel.unshard(loss / weights))

    batch_idx, results = 0, {}
    while True:
        batch = parallel.unshard(train_batch_fn(batch_idx))
        batch_idx += 1

        ids = np.array(batch[dp.FIELDS.IDENTIFIER]).flatten()
        if ids[0] in results:  # only do one epoch.
            break

        for b in meta_learning.split_batch(batch, num_sub_batches):
            ids = np.array(b[dp.FIELDS.IDENTIFIER]).flatten()
            if ids[0] in results:  # only do one epoch.
                break
            loss = loss_per_sample(b)
            assert loss.shape[0] == ids.shape[0]
            results.update({i: l for i, l in zip(ids, loss)})

        if batch_idx % 100 == 0:
            logging.info("scored: %d" % len(results))

    return results


def _specific_train(model_state, weight_fn, batch_fn, num_steps, rng):
    """Train a model on specific data for num_steps."""
    model_state = parallel.replicate(model_state)
    specific_fn = learning.apply_model(
        model_state.specific_fn,
        loss_fn=model_state.specific_loss_fn,
        need_gradient=True,
        weight_fn=weight_fn,
    )

    @functools.partial(parallel.pmap, axis_name="batch")
    def specific_train_step(model_state, s_batch):
        p = model_state.params
        s_grads, _, _ = specific_fn(p, s_batch)
        s_grads = parallel.pmean(s_grads, "batch")
        model_state, _ = learning.update_model(model_state, s_grads)
        return model_state

    for step in range(num_steps):
        batch = batch_fn(step)
        rng_cur, rng = jax.random.split(rng)
        batch.update({"rng": parallel.shard_prng_key(rng_cur)})
        model_state = specific_train_step(
            model_state,
            batch,
        )

    return parallel.unreplicate(model_state)


def _train_cds_selector(
    score_model_state,
    fine_tune_model_state,
    generic_train_batch_fn,
    specific_train_batch_fn,
    default_weight_fn,
    num_ft_step,
    num_scoring_sub_batches,
    rng,
):
    """Train a CDS model from a pretrained generic model."""
    logging.info("CDS pre-fine-tune scoring")
    pre_ft_losses = _generic_loss_per_example(
        score_model_state,
        default_weight_fn,
        generic_train_batch_fn,
        num_scoring_sub_batches,
    )

    logging.info("CDS fine-tuning")
    fine_tune_model_state = _specific_train(
        fine_tune_model_state,
        default_weight_fn,
        specific_train_batch_fn,
        num_ft_step,
        rng,
    )
    score_model_state = score_model_state.replace(
        params=fine_tune_model_state.params,
    )

    logging.info("CDS post-fine-tune scoring")
    post_ft_losses = _generic_loss_per_example(
        score_model_state,
        default_weight_fn,
        generic_train_batch_fn,
        num_scoring_sub_batches,
    )

    # Loss dictionaries might miss a few examples if the training data pipeline
    # drop the last incomplete batch.
    delta_loss = {
        k: post_ft_losses[k] - pre_ft_losses[k]
        for k in pre_ft_losses
        if k in post_ft_losses
    }
    # Missing examples receive this value so that they are not selected.
    default_value = max(delta_loss.values())

    # convert the dictionary to a jnp to allow selector parallelization.
    num_samples = max(delta_loss.keys()) + 1
    delta_loss = jnp.array(
        [delta_loss.get(i, default_value) for i in range(num_samples)],
        jnp.float16,
    )

    return delta_loss


def _delta_loss_selector(batch, delta_loss, batch_size):
    ids = batch[dp.FIELDS.IDENTIFIER].flatten()
    ids = jnp.minimum(ids, delta_loss.size)
    score = -delta_loss[ids]
    return meta_learning.sample_batch(
        None, batch, score, batch_size, sample="top"
    )


def cds_meta_step_fn(
    task,
    config,
    generic_train_batch_fn,
    specific_train_batch_fn,
    default_weight_fn,
):
    """Meta step fn for CDS.

    It expects the config to contain:

    config.cds_parameters = {
        "scoring_config_overloads": dict,
        "fine_tune_config_overloads": dict,
        "num_pretrain_steps": int,
        "num_fine_tune_steps": int,
        "num_scoring_sub_batches": int,
    }
    """
    if config.meta_gradient_method != sm.META_GRADIENT_METHOD.CDS:
        return None

    num_pretrain_steps = config.cds_parameters.num_pretrain_steps
    num_fine_tune_steps = config.cds_parameters.num_fine_tune_steps
    num_scoring_sub_batches = config.cds_parameters.get(
        "num_scoring_sub_batches", 1
    )
    batch_size = config.batch_size
    delta_loss = None
    weight_fn = parallel.pmap(default_weight_fn)
    sample_fn = parallel.pmap(
        lambda rng, batch: meta_learning.sample_batch(
            rng, batch, None, batch_size, sample="rand"
        )
    )
    selection_fn = parallel.pmap(
        lambda a, b: _delta_loss_selector(a, b, batch_size)
    )

    def init_model_states(params):
        """CDS can overload some config parameters for scoring, fine-tuning."""
        rng = jax.random.PRNGKey(config.seed)

        score_config = _overload_config_dict(
            config,
            config.cds_parameters.scoring_config_overloads,
        )
        score_model_state, _, _ = task.init_model(rng, score_config)

        fine_tune_config = _overload_config_dict(
            config,
            config.cds_parameters.fine_tune_config_overloads,
        )
        fine_tune_model_state, _, _ = task.init_model(rng, fine_tune_config)

        return (
            score_model_state.replace(params=params),
            fine_tune_model_state.replace(params=params),
        )

    def meta_step_fn(weight_state, model_state, g_batch, m_batch, rng, step):
        # Note this function cannot be jitted.
        del m_batch
        nonlocal delta_loss

        if step >= num_pretrain_steps:
            # after pretraining, selection of best delta loss.
            if delta_loss is None:
                model_state = parallel.unreplicate(model_state)
                score_state, ft_state = init_model_states(model_state.params)
                delta_loss = _train_cds_selector(
                    score_state,
                    ft_state,
                    generic_train_batch_fn,
                    specific_train_batch_fn,
                    default_weight_fn,
                    num_fine_tune_steps,
                    num_scoring_sub_batches,
                    rng=jax.random.PRNGKey(config.seed),
                )
                delta_loss = parallel.replicate(delta_loss)
            g_batch = selection_fn(g_batch, delta_loss)
        else:
            # during pretraining, select a random subset.
            g_batch = sample_fn(rng, g_batch)

        return weight_state, g_batch, weight_fn(g_batch), []

    return meta_step_fn

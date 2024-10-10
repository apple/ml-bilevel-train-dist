#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Evaluation"""
import functools
import glob
import itertools
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from shared import data_pipeline as dp
from shared import learning, meta_learning, parallel, tasks


def _save_predictions(
    file_prefix,
    predictions,
    exts=(".idx", ".loss"),
    concatenate=True,
):
    if predictions:  # check if the dataset has at least one batch.
        if concatenate:
            predictions = [np.concatenate(a) for a in zip(*predictions)]
        logging.info("Save %s...", file_prefix)
        for pred, ext in zip(predictions, exts):
            np.save(file_prefix + ext, pred)


def _loss_per_batch(batch, params, model_fn, eps=1e-6):
    idx = batch[dp.FIELDS.IDENTIFIER]
    loss, _, weights = model_fn(params, batch)
    ndim = jnp.ndim(loss)
    sum_axes = tuple(range(1, ndim))  # keep only batch axis
    loss = (loss * weights).sum(sum_axes)
    loss = loss / (weights + eps).sum(sum_axes)
    return idx, loss


def _log_losses(m_params, model_fn, w_params, weight_fn, datasets, checkpoint):
    batch_parallel = lambda f: parallel.pmap(f, axis_name="batch")
    p_loss = batch_parallel(lambda b, p: _loss_per_batch(b, p, model_fn))
    p_weight = batch_parallel(lambda b, p: weight_fn(b, p))

    for name, batch_fn in datasets:
        output = "%s_predictions_%s" % (checkpoint, name)
        logging.info("Infer %s...", output)
        losses, weights = [], []
        for batch in batch_fn():
            b_idx, b_loss = [np.array(a) for a in p_loss(batch, m_params)]
            b_weights = np.array(p_weight(batch, w_params))
            losses.extend(zip(b_idx, b_loss))
            weights.extend(zip(b_idx, b_weights))
        _save_predictions(output, losses)
        _save_predictions(output, weights, exts=(".idx", ".meta"))


def _log_inference(
    params, inference_eval_fn, specific_sets, generic_sets, checkpoint
):
    if inference_eval_fn is not None:
        all_sets = [(n, b, True) for n, b in generic_sets]
        all_sets += [(n, b, False) for n, b in specific_sets]
        for name, batch_fn, is_generic in all_sets:
            logging.info("Inference evaluation for dataset %s.", name)
            _, inf_texts = inference_eval_fn(
                params, batch_fn, 0, name, is_generic
            )
            inf_texts = list(inf_texts.values())[0]
            output = "%s_inference_%s" % (checkpoint, name)
            with open(output, "w") as fout:
                fout.write(inf_texts)


def _gradient_vs_delta_weight(
    model_state,
    model_state_before,
    model_state_after,
    batch_fn,
    weight_fn,
    max_batches=100,
    filter=None,
):
    delta_parameters = jax.tree_map(
        lambda x, y: x - y,
        model_state_after.params,
        model_state_before.params,
    )
    if filter is not None:
        f = learning.parameter_filter_from_str(filter)
        delta_parameters = learning.selective_zeros(delta_parameters, f)

    g_dot_prod = functools.partial(
        parallel.pmap(
            functools.partial(
                meta_learning.dot_prod_with_delta_params,
                is_generic=True,
                default_weight_fn=weight_fn,
            ),
            axis_name="batch",
        ),
        model_state=model_state,  # already replicated
        delta_params=delta_parameters,  # already replicated
    )

    out = []
    for _, batch in zip(range(max_batches), batch_fn()):
        batch = parallel.shard(batch)
        res = g_dot_prod(batch=batch)
        idx = np.array(batch[dp.FIELDS.IDENTIFIER]).flatten()
        res = np.array(res).flatten()
        out.append((idx, res))
    idx, dot_prods = [np.concatenate(a) for a in zip(*out)]
    return idx, dot_prods


def _gradient_cosines(
    model_state_a,
    model_state_b,
    batch_fn_a,
    batch_fn_b,
    weight_fn,
    max_batches_b=100,
    max_batches_a=100,
    batch_b_is_generic=False,
):
    g_cosine = functools.partial(
        parallel.pmap(
            functools.partial(
                meta_learning.gradient_cosine,
                batch_a_is_generic=True,
                batch_b_is_generic=batch_b_is_generic,
                default_weight_fn=weight_fn,
            ),
            axis_name="batch",
        ),
        model_state_a=model_state_a,  # already replicated
        model_state_b=model_state_b,  # already replicated
    )

    # vision pipeline is sharded with some replicated fields, the others
    # are not distributed. We use this roundabout way to unshard the data.
    def maybe_unshard(x):
        if hasattr(x["inputs"], "sharding_spec"):
            n = x["inputs"].shape[0]
            y = [jax.tree_util.tree_map(lambda l: l[i], x) for i in range(n)]
        else:
            y = [x]
        return y

    def iter_unshard():
        for sharded_b in batch_fn_b():
            for batch_b in maybe_unshard(sharded_b):
                yield batch_b

    outputs = []
    for _, batch_b in zip(range(max_batches_b), iter_unshard()):
        batch_b = parallel.replicate(batch_b)
        out = []
        for _, batch_a in zip(range(max_batches_a), batch_fn_a()):
            res = g_cosine(batch_a=batch_a, batch_b=batch_b)
            idx_a = np.array(batch_a[dp.FIELDS.IDENTIFIER]).flatten()
            res = np.array(res).flatten()
            out.append((idx_a, res))
        # concatenate over generic batches
        idx_b = np.array(batch_b[dp.FIELDS.IDENTIFIER][0])
        outputs.append([idx_b] + [np.concatenate(a) for a in zip(*out)])
    # stack over specific batches
    idx_b, generic_idx, cosines = [np.stack(a) for a in zip(*outputs)]
    # output is num_specific_batches x num_generic_examples
    return idx_b, generic_idx[0], cosines


def _log_gradient_vs_delta_weight(
    workdir,
    model_state,
    weight_state,
    generic_sets,
    weight_fn,
    filters=None,
):
    filters = [None] if filters is None else filters

    def filter_name(f):
        if f is None:
            return ""
        return "_" + f.strip("/").replace("/", "_")

    def load_model(checkpoint):
        m_state, _, step = learning.load_checkpoint(
            checkpoint, model_state, weight_state
        )
        m_state = parallel.replicate(m_state)
        return m_state, step

    for checkpoint in _checkpoint_list(workdir):
        m_state, step = load_model(checkpoint)
        for ft_checkpoint in glob.glob(checkpoint + ".fine_tuned*"):
            ext = ft_checkpoint[len(checkpoint + ".fine_tuned") :]
            m_state_ft, step_ft = load_model(ft_checkpoint)
            assert step_ft > step
            for g_name, batch_fn in generic_sets:
                for f in filters:
                    outputs = _gradient_vs_delta_weight(
                        model_state=m_state,
                        model_state_before=m_state,
                        model_state_after=m_state_ft,
                        batch_fn=batch_fn,
                        weight_fn=weight_fn,
                        filter=f,
                    )
                    filename = f"dot_prods_{g_name}_{step//1000}k{ext}"
                    filename += filter_name(f)
                    _save_predictions(
                        os.path.join(workdir, filename),
                        outputs,
                        (".individual_idx", ".dot_prods"),
                        concatenate=False,
                    )


def _log_gradient_cosine(
    workdir,
    model_state,
    weight_state,
    generic_sets,
    specific_sets,
    weight_fn,
):
    def load_model(checkpoint):
        m_state, _, step = learning.load_checkpoint(
            checkpoint, model_state, weight_state
        )
        m_state = parallel.replicate(m_state)
        return m_state, step

    def save_cosines(workdir, g_name, s_name, step_a, step_b, predictions):
        k_step_a, k_step_b = step_a // 1000, step_b // 1000
        filename = f"cosines_{g_name}_{s_name}_{k_step_a}k_{k_step_b}k"
        _save_predictions(
            os.path.join(workdir, filename),
            predictions,
            (".batch_idx", ".individual_idx", ".cosines"),
            concatenate=False,
        )

    checkpoints = _checkpoint_list(workdir)
    checkpoint_pairs = itertools.product(checkpoints, checkpoints)
    for checkpoint_a, checkpoint_b in checkpoint_pairs:
        m_state_a, step_a = load_model(checkpoint_a)
        m_state_b, step_b = load_model(checkpoint_b)
        # generic vs specific dot-prod
        dataset_pairs = itertools.product(generic_sets, specific_sets)
        for g_set, s_set in dataset_pairs:
            g_name, generic_set_fn = g_set
            s_name, specific_set_fn = s_set
            outputs = _gradient_cosines(
                m_state_a,
                m_state_b,
                generic_set_fn,
                specific_set_fn,
                weight_fn,
            )
            save_cosines(workdir, g_name, s_name, step_a, step_b, outputs)
        # generic vs generic dot-prod
        dataset_pairs = itertools.product(generic_sets, generic_sets)
        for g_set_a, g_set_b in dataset_pairs:
            g_name_a, generic_set_fn_a = g_set_a
            g_name_b, generic_set_fn_b = g_set_b
            outputs = _gradient_cosines(
                m_state_a,
                m_state_b,
                generic_set_fn_a,
                generic_set_fn_b,
                weight_fn,
                batch_b_is_generic=True,
            )
            save_cosines(workdir, g_name_a, g_name_b, step_a, step_b, outputs)


def _checkpoint_list(workdir):
    checkpoints = glob.glob(os.path.join(workdir, "*checkpoint_*"))
    checkpoints = [f for f in checkpoints if re.match(r".*checkpoint_\d+$", f)]
    return checkpoints


def evaluate(config, workdir, datadir=None):
    logging.info("Evaluation only, no training.")
    rng = jax.random.PRNGKey(config.seed)
    datadir = workdir if datadir is None else datadir
    (
        model_state,
        weight_state,
        _,  # skip generic, specific, and meta train_batch_fn
        _,
        _,
        eval_sets,
        default_weight_fn,
        inference_eval_fn,
    ) = tasks.get_task(config.task).init_task(rng, config, datadir)
    specific_sets = [tuple(s) for is_generic, *s in eval_sets if not is_generic]
    generic_sets = [tuple(s) for is_generic, *s in eval_sets if is_generic]

    generic_fn = learning.apply_model(
        model_state.eval_generic_fn,
        loss_fn=model_state.generic_loss_fn,
        need_gradient=False,
        weight_fn=default_weight_fn,
        reduce=False,
    )
    specific_fn = learning.apply_model(
        model_state.eval_specific_fn,
        loss_fn=model_state.specific_loss_fn,
        need_gradient=False,
        weight_fn=default_weight_fn,
        reduce=False,
    )

    if config.get("evaluate_gradient_vs_delta_weight", False):
        _log_gradient_vs_delta_weight(
            workdir,
            model_state,
            weight_state,
            generic_sets,
            default_weight_fn,
            filters=config.get("delta_weight_filters", None),
        )

    if config.get("evaluate_gradient_cosine", False):
        _log_gradient_cosine(
            workdir,
            model_state,
            weight_state,
            generic_sets,
            specific_sets,
            default_weight_fn,
        )

    checkpoints = _checkpoint_list(workdir)
    for checkpoint in checkpoints:
        model_state, weight_state, _ = learning.load_checkpoint(
            checkpoint, model_state, weight_state
        )
        model_state = parallel.replicate(model_state)
        weight_state = parallel.replicate(weight_state)
        # same interface for default weight and parameterized weight model.
        weight_fn = lambda b, _: default_weight_fn(b)
        w_params = None

        # loss on specific examples
        _log_losses(
            model_state.params,
            specific_fn,
            None,
            default_weight_fn,
            specific_sets,
            checkpoint,
        )

        # loss on generic examples
        if weight_state is not None:
            w_params = weight_state.params
            weight_fn = lambda x, p: weight_state.apply_fn({"params": p}, x)

        _log_losses(
            model_state.params,
            generic_fn,
            w_params,
            weight_fn,
            generic_sets,
            checkpoint,
        )

        # task-specific inference
        _log_inference(
            model_state.params,
            inference_eval_fn,
            specific_sets,
            generic_sets,
            checkpoint,
        )

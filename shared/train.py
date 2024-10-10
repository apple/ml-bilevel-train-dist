#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Training and evaluation loops"""

import functools
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from flax.core.frozen_dict import freeze
from flax.metrics import tensorboard

from shared import contrastive_data_selection
from shared import data_pipeline as dp
from shared import learning, meta_learning
from shared import model as sm
from shared import parallel, sysutils, tasks

# todo migrate to orbax.
flax.config.update("flax_use_orbax_checkpointing", False)


def metric_logging(
    metric_dict,
    extra_metrics={},
    text_metrics={},
    summary_writer=None,
):
    md = metric_dict.copy()
    step = md.pop("step")
    sysutils.send_metrics(md, iteration=step)
    msg = " step % 5d" % step
    for k, v in md.items():
        msg += " %s : %.4f" % (k, v)
    logging.info(msg)
    if summary_writer is not None:
        for k, v in md.items():
            summary_writer.scalar(k, v, step)
        for k, v in extra_metrics.items():
            summary_writer.scalar(k, v, step)
        for k, v in text_metrics.items():
            summary_writer.text(k, v, step)


def train_weight_logging(workdir: str, frequency: int):
    """Log generic samples identifiers and their weights."""
    if frequency < 0:
        return lambda *args: ()

    state = ([], [])

    def log_train_weight_fn(step, sample_id, selected_id, selected_w):
        sample_id, selected_id, selected_w = [
            parallel.unshard(a) for a in (sample_id, selected_id, selected_w)
        ]
        batch_size = sample_id.shape[0]
        sample_id = np.array(sample_id).flatten().astype(np.uint32)
        selected_id = np.array(selected_id).flatten()
        while selected_w.size > selected_id.size:  # one weight per example.
            selected_w = selected_w.sum(axis=-1)
        selected_w = np.array(selected_w).flatten()
        selected = {k: v for k, v in zip(selected_id, selected_w)}
        selected = np.array(
            [selected.get(k, 0.0) for k in sample_id], np.float16
        )

        if sample_id.size == batch_size:
            state[0].append(sample_id.reshape((1, batch_size)))
            state[1].append(selected.reshape((1, batch_size)))
        else:
            logging.warning("Dropping a smaller batch from logging.")

        step += 1  # num_train steps.
        if step % frequency == 0:
            filename = "%s/train_weights_%08d" % (workdir, step)
            with open(filename, "wb") as fout:
                for a in state:
                    np.save(fout, np.concatenate(a))
                    a.clear()

    return log_train_weight_fn


def perf_report(
    eval_fn,
    logging_fn,
    eval_every_steps,
    save_fn=None,
    save_every_steps=0,
):
    """Periodic eval and keep track of training step metrics for averaging."""
    metrics = []
    timer_start = [time.time()]

    def update_report_fn(
        step, reg_metrics, meta_metrics, model_state, weight_state
    ):
        if not meta_metrics:  # null with shape/type of other metrics.
            meta_metrics = [reg_metrics[0] * 0.0] * 4
        metrics.append(list(reg_metrics) + list(meta_metrics))
        if save_every_steps > 0 and (step % save_every_steps == 0):
            save_fn(step, model_state, weight_state)

        if step % eval_every_steps == 0:
            num_logged_steps = len(metrics)
            steps_per_sec = num_logged_steps / (time.time() - timer_start[0])
            mean_m = np.reshape(np.array(metrics), (num_logged_steps, 9, -1))
            # mean over steps and devices
            mean_m = list(np.mean(mean_m, axis=(0, -1)))
            logging_fn(
                {
                    "step": step,
                    "steps_per_sec": steps_per_sec,
                    # undo sum over devices (each worker computed the same norm)
                    "grad_norm": mean_m.pop(0),
                    "generic_loss": mean_m.pop(0),
                    "generic_accuracy": mean_m.pop(0) * 100,
                    "specific_loss": mean_m.pop(0),
                    "specific_accuracy": mean_m.pop(0) * 100,
                    "meta_grad_norm": mean_m.pop(0),
                    "specific_meta_loss": mean_m.pop(0),
                    "specific_meta_accuracy": mean_m.pop(0) * 100,
                    "specific_meta_aux_loss": mean_m.pop(0),
                }
            )
            eval_fn(step, model_state)
            metrics.clear()
            timer_start[0] = time.time()

    return update_report_fn


def train(
    init_step,
    model_state,
    default_weight_fn,
    generic_train_batch_fn,
    specific_train_batch_fn,
    generic_weight_fn,
    perf_report_update_fn,
    num_steps,
    rng,
    *,
    meta_step_fn=None,
    meta_train_schedule_fn=None,
    weight_state=None,
    meta_train_batch_fn=None,
    replace_weight_state_fn=None,
    log_train_weights_fn=None,
):
    """Train a model over a mixture of generic and specific datasets."""
    generic_fn = learning.apply_model(
        model_state.generic_fn,
        loss_fn=model_state.generic_loss_fn,
        need_gradient=True,
    )
    specific_fn = learning.apply_model(
        model_state.specific_fn,
        loss_fn=model_state.specific_loss_fn,
        need_gradient=True,
    )
    model_state = parallel.replicate(model_state)
    weight_state = parallel.replicate(weight_state)

    @functools.partial(
        parallel.pmap,
        axis_name="batch",
        static_broadcasted_argnums=(1,),
    )
    def regular_train_step_mixed(
        model_state,
        generic_w,
        g_batch,
        g_weights,
        s_batch,
    ):
        p = model_state.params
        if g_weights is None:
            g_weights = default_weight_fn(g_batch)
        s_weights = default_weight_fn(s_batch)
        g_grads, g_loss, g_acc = generic_fn(p, g_batch, g_weights)
        s_grads, s_loss, s_acc = specific_fn(p, s_batch, s_weights)

        # first map over the coarse tree of generic_w, then over the finer
        # parameter subtrees g_grads and s_grads.
        grads = jax.tree_map(
            lambda w, g_grad, s_grad: jax.tree_map(
                lambda g, s: w * g + (1 - w) * s, g_grad, s_grad
            ),
            generic_w,
            g_grads,
            s_grads,
        )
        grads = parallel.pmean(grads, "batch")
        model_state, grad_norm = learning.update_model(model_state, grads)
        return model_state, grad_norm, g_loss, g_acc, s_loss, s_acc

    @functools.partial(parallel.pmap, axis_name="batch")
    def regular_train_step_generic_only(model_state, g_batch, g_weights):
        p = model_state.params
        if g_weights is None:
            g_weights = default_weight_fn(g_batch)
        g_grads, g_loss, g_accuracy = generic_fn(p, g_batch, g_weights)
        g_grads = parallel.pmean(g_grads, "batch")
        model_state, grad_norm = learning.update_model(model_state, g_grads)
        s_loss, s_accuracy = (
            0.0 * g_loss,  # null with shape/type of other metrics.
            0.0 * g_loss,
        )
        return model_state, grad_norm, g_loss, g_accuracy, s_loss, s_accuracy

    def regular_train_step(
        model_state,
        generic_w,
        g_batch,
        g_weights,
        s_batch,
    ):
        if generic_w == 1.0:
            return regular_train_step_generic_only(
                model_state, g_batch, g_weights
            )
        else:
            return regular_train_step_mixed(
                model_state,
                generic_w,
                g_batch,
                g_weights,
                s_batch,
            )

    if meta_step_fn is not None:
        # check we have all meta learning args
        assert meta_train_schedule_fn is not None
        assert meta_train_batch_fn is not None
    else:
        is_train_step = True
        meta_metrics = []

    current_checkpoint_id = 0
    rng = parallel.device_prefetch(parallel.shard_prng_key(rng))
    rng_fn = parallel.pmap(
        lambda k, s: list(jax.random.split(jax.random.fold_in(k, s), 4)),
        in_axes=(0, None),
    )

    for step in range(init_step, num_steps):
        meta_rng, g_batch_rng, s_batch_rng, m_batch_rng = rng_fn(rng, step)

        if replace_weight_state_fn is not None:
            weight_state, current_checkpoint_id = replace_weight_state_fn(
                model_state, weight_state, step, current_checkpoint_id
            )

        # sample data with default uniform weights
        g_batch = generic_train_batch_fn(step)
        g_weights = None
        g_batch.update({"rng": g_batch_rng})
        pre_selection_g_id = g_batch[dp.FIELDS.IDENTIFIER]

        # meta training
        if meta_step_fn is not None:
            logging.log_first_n(logging.INFO, "Meta training step %d", 5, step)
            step_dict = meta_train_schedule_fn(step)
            is_train_step = step_dict["train_step"]
            m_batch = meta_train_batch_fn(step)
            m_batch.update({"rng": m_batch_rng})
            weight_state, g_batch, g_weights, meta_metrics = meta_step_fn(
                weight_state,
                model_state,
                g_batch,
                m_batch,
                meta_rng,
                step,
            )
            logging.log_first_n(logging.INFO, "Meta training step done", 5)

        # log weights and sample ids.
        if g_weights is not None and log_train_weights_fn is not None:
            post_selection_g_id = g_batch[dp.FIELDS.IDENTIFIER]
            log_train_weights_fn(
                step, pre_selection_g_id, post_selection_g_id, g_weights
            )

        # regular training
        logging.log_first_n(logging.INFO, "Regular training step %d", 5, step)
        w = generic_weight_fn(step)
        if w == 1.0:
            s_batch = None
        else:
            s_batch = specific_train_batch_fn(step)
            s_batch.update({"rng": s_batch_rng})

        new_state, *reg_metrics = regular_train_step(
            model_state,
            w,
            g_batch,
            g_weights,
            s_batch,
        )
        logging.log_first_n(logging.INFO, "Regular training step done", 5)

        if is_train_step:
            model_state = new_state
        del new_state

        # logging
        perf_report_update_fn(
            step + 1, reg_metrics, meta_metrics, model_state, weight_state
        )

    return model_state, weight_state


def evaluate_weighting_grads(
    model_state,
    weight_state,
    train_batch,
    meta_train_batch,
    default_weight_fn,
    soft_meta_lr,
):
    generic_fn = learning.apply_model(
        model_state.generic_fn,
        loss_fn=model_state.generic_loss_fn,
        need_gradient=True,
    )
    specific_fn = learning.apply_model(
        model_state.specific_fn,
        loss_fn=model_state.specific_loss_fn,
        need_gradient=False,
        weight_fn=default_weight_fn,
    )

    # Learning rate is zero? We modify the sgd update.
    # This corresponds to the limit as lr->0 without numerical issues.
    assert soft_meta_lr >= 0.0
    if soft_meta_lr == 0.0:
        sgd_update_fn = lambda p, g: p - (g - jax.lax.stop_gradient(g))
    else:
        sgd_update_fn = lambda p, g: p - soft_meta_lr * g

    def weight_fn(params, batch):
        return weight_state.apply_fn({"params": params}, batch)

    def post_update_loss_fn(
        train_weights,
        m_params,
        train_inputs,
        meta_train_input,
    ):
        train_grads, _, _ = generic_fn(m_params, train_inputs, train_weights)
        updated_m_params = jax.tree_map(sgd_update_fn, m_params, train_grads)
        meta_loss, _ = specific_fn(updated_m_params, meta_train_input)
        return meta_loss

    # current weights and dummy zero delta to be added to compute gradients.
    train_weights = weight_fn(weight_state.params, train_batch)

    grad_fn = jax.value_and_grad(post_update_loss_fn)
    _, weights_grad = grad_fn(
        train_weights,
        model_state.params,
        train_batch,
        meta_train_batch,
    )
    return train_weights, weights_grad


def maybe_replace_weight_state_fn(config, rng):
    """Function to replay a previously trained sequence of weighting models."""

    replace_weight_state_fn = None
    if "frozen_weight_curriculum" in config:
        checkpoint_dir = config.frozen_weight_curriculum.checkpoints_dir
        checkpoint_frequency = (
            config.frozen_weight_curriculum.checkpoint_frequency
        )
        checkpoint_steps = jnp.arange(
            config.save_every_steps,
            config.num_steps + 1,
            config.save_every_steps,
        )
        if config.frozen_weight_curriculum.mode == "random":
            checkpoint_steps = jax.random.permutation(rng, checkpoint_steps)
        elif config.frozen_weight_curriculum.mode == "last":  # constant weights
            checkpoint_steps = config.num_steps * jnp.ones_like(
                checkpoint_steps
            )
        elif config.frozen_weight_curriculum.mode == "fixed":
            checkpoint_id = config.frozen_weight_curriculum.checkpoint_id
            checkpoint_steps = checkpoint_id * jnp.ones_like(checkpoint_steps)

        def replace_weight_state_fn(
            model_state, weight_state, step, current_checkpoint_id
        ):
            if step % checkpoint_frequency == 0:
                checkpoint_to_load = checkpoint_steps[current_checkpoint_id]
                _, weight_state, _ = learning.load_checkpoint(
                    f"{checkpoint_dir}/checkpoint_{checkpoint_to_load}",
                    model_state,
                    weight_state,
                    load_linear_state=False,
                )
                weight_state = parallel.replicate(weight_state)
                return weight_state, current_checkpoint_id + 1
            else:
                return weight_state, current_checkpoint_id

    return replace_weight_state_fn


def train_and_evaluate(config, workdir, datadir=None):
    rng = jax.random.PRNGKey(config.seed)
    tb_writer = tensorboard.SummaryWriter(workdir)
    tb_writer.hparams(dict(config))

    datadir = workdir if datadir is None else datadir
    (
        model_state,
        weight_state,
        generic_train_batch_fn,
        specific_train_batch_fn,
        meta_train_batch_fn,
        eval_sets,
        default_weight_fn,
        inference_evaluation_fn,
    ) = tasks.get_task(config.task).init_task(rng, config, datadir)
    logging_fn = functools.partial(metric_logging, summary_writer=tb_writer)

    generic_fn = learning.apply_model(
        model_state.eval_generic_fn,
        loss_fn=model_state.generic_loss_fn,
        need_gradient=False,
        weight_fn=default_weight_fn,
    )
    specific_fn = learning.apply_model(
        model_state.eval_specific_fn,
        loss_fn=model_state.specific_loss_fn,
        need_gradient=False,
        weight_fn=default_weight_fn,
    )
    generic_fn = parallel.pmap(generic_fn, axis_name="batch")
    specific_fn = parallel.pmap(specific_fn, axis_name="batch")

    def eval_fn(step, model_state):
        params = model_state.params
        metrics, text_metrics = {"step": step}, {}
        mean_fn = lambda x: np.array(x).mean()  # mean over steps and devices

        for is_generic, name, batch_fn in eval_sets:
            f = generic_fn if is_generic else specific_fn
            loss, acc = zip(*[f(params, b) for b in batch_fn()])

            # optional inference-based evaluation
            if inference_evaluation_fn is not None:
                inf_metrics, inf_texts = inference_evaluation_fn(
                    model_state.params,
                    batch_fn,
                    step,
                    name,
                    is_generic,
                )
                metrics.update(inf_metrics)
                text_metrics.update(inf_texts)

            metrics.update(
                {
                    "eval_%s_loss" % name: mean_fn(loss),
                    "eval_%s_accuracy" % name: mean_fn(acc) * 100,
                }
            )
        logging_fn(metrics, text_metrics=text_metrics)

    def save_fn(step, model_state, weight_state):
        keep_every_n_steps = None
        if config.keep_checkpoint_frequency > 0:
            keep_every_n_steps = config.keep_checkpoint_frequency
        learning.save_checkpoint(
            workdir,
            step,
            parallel.unreplicate(model_state),
            parallel.unreplicate(weight_state),
            keep_every_n_steps=keep_every_n_steps,
        )

    generic_weight = config.generic_weight
    if not isinstance(generic_weight, float):
        generic_weight = freeze(generic_weight)

    def generic_weight_fn(step):
        del step
        return generic_weight

    meta_train_schedule_fn = meta_learning.meta_train_schedule(config)

    # if reweight method, weight state is none but same interface
    if weight_state is not None:
        meta_step_fn = meta_learning.get_meta_step_fn(
            config, default_weight_fn, meta_train_schedule_fn
        )
    elif config.meta_gradient_method == sm.META_GRADIENT_METHOD.CDS:
        # CDS is not a meta method but it has the same interface.
        meta_step_fn = contrastive_data_selection.cds_meta_step_fn(
            tasks.get_task(config.task),
            config,
            generic_train_batch_fn,
            specific_train_batch_fn,
            default_weight_fn,
        )
    else:
        meta_step_fn = None

    perf_report_update_fn = perf_report(
        eval_fn,
        logging_fn,
        config.eval_every_steps,
        save_fn=save_fn,
        save_every_steps=config.save_every_steps,
    )

    if config.get("disable_weight_logging", False):
        log_train_weights_fn = None
    else:
        log_train_weights_fn = train_weight_logging(
            workdir=workdir,
            frequency=config.eval_every_steps,
        )

    replace_weight_state_fn = maybe_replace_weight_state_fn(config, rng)

    model_state, weight_state, step = learning.load_checkpoint(
        workdir, model_state, weight_state
    )

    logging.info("Training model...")
    train(
        step,
        model_state,
        default_weight_fn,
        generic_train_batch_fn,
        specific_train_batch_fn,
        generic_weight_fn,
        perf_report_update_fn,
        config.num_steps,
        rng,
        weight_state=weight_state,
        meta_step_fn=meta_step_fn,
        meta_train_schedule_fn=meta_train_schedule_fn,
        meta_train_batch_fn=meta_train_batch_fn,
        replace_weight_state_fn=replace_weight_state_fn,
        log_train_weights_fn=log_train_weights_fn,
    )

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Learning utils: model forward, backward and update."""

import os
from typing import Callable, Optional

import flax
import jax
import jax.numpy as jnp
import optax
from absl import logging
from flax import serialization
from flax.training import checkpoints

import shared.data_pipeline as dp


def weighted_cross_entropy_loss(
    model_fn,
    params,
    batch,
    weights,
    label_smoothing: float = 0.0,
):
    """Default loss function."""
    labels = batch[dp.FIELDS.LABELS]
    rngs = module_rngs(batch)
    logits = model_fn({"params": params}, batch, rngs=rngs)

    target = jax.nn.one_hot(labels, logits.shape[-1])

    if label_smoothing > 0.0:
        eps = label_smoothing / (logits.shape[-1] - 1)
        ones = jnp.ones_like(target)
        target = (1.0 - label_smoothing - eps) * target + eps * ones

    loss = optax.softmax_cross_entropy(logits=logits, labels=target)
    acc = (jnp.argmax(logits, -1) == labels).astype(weights.dtype)
    return loss, acc, weights


def apply_model(
    model_fn,
    loss_fn: Optional[Callable] = None,
    need_gradient: bool = True,
    weight_fn: Optional[Callable] = None,
    reduce: bool = True,
    need_hvp: bool = False,
    normalized_loss: bool = True,
    eps=1e-6,
):
    """Computes gradients, loss and accuracy for a single batch."""
    assert reduce or not need_gradient, "reduce is required with need_gradient."
    if loss_fn is None:
        loss_fn = weighted_cross_entropy_loss

    def fprop_fn(params, batch, weights):
        loss, acc, weights = loss_fn(model_fn, params, batch, weights)
        if not reduce:
            return loss, acc, weights
        if normalized_loss:
            weights = weights / (eps + jnp.sum(weights))
        return jnp.sum(weights * loss), jnp.sum(weights * acc)

    def backprop_fn(params, batch, weights):
        grad_fn = jax.value_and_grad(fprop_fn, has_aux=True)
        (loss, acc), grads = grad_fn(params, batch, weights)
        return grads, loss, acc

    def hvp_fn(params, direction, batch, weights):
        grad_fn = jax.grad(
            lambda params: fprop_fn(params, batch, weights),
            has_aux=True,
        )
        return jax.jvp(
            lambda params: grad_fn(params)[0], [params], [direction]
        )[1]

    if need_gradient:
        fn = backprop_fn
    else:
        fn = fprop_fn

    if weight_fn is not None:
        out_fn = lambda params, batch: fn(params, batch, weight_fn(batch))
    else:
        out_fn = fn

    if need_hvp:
        return (out_fn, hvp_fn)

    return out_fn


def update_model(state, grads, linear: bool = False):
    grad_norm = tree_l2_norm(grads)
    if linear:
        updated_state = state.apply_linear_gradients(grads=grads)
    else:
        updated_state = state.apply_gradients(grads=grads)
    return updated_state, grad_norm


def module_rngs(batch):
    if "rng" in batch:
        return {"dropout": batch["rng"]}
    else:
        return {}


def tree_l2_norm(tree):
    """Compute the l2 norm of a pytree.."""
    leaves, _ = jax.tree_util.tree_flatten(tree)
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def tree_sum(tree):
    """Compute the l2 norm of a pytree.."""
    leaves, _ = jax.tree_util.tree_flatten(tree)
    return sum(jnp.sum(x) for x in leaves)


def log_parameter_shapes(filename, parameters):
    parameters = flax.traverse_util.flatten_dict(parameters)
    total_size = sum([v.size for v in parameters.values()])
    with open(filename, "w") as fout:
        for k, v in parameters.items():
            name = "/".join(k)
            frac = v.size / total_size
            shape = str(v.shape)
            fout.write("%.6f %s %s\n" % (frac, name, shape))


def selective_zeros(
    params,
    filter_fn: Optional[Callable[[str], bool]],
    verbose: bool = True,
    return_num_preserved_params: bool = False,
):
    """Keep the gradients only for some params based on the layer names."""
    # see https://github.com/google/flax/discussions/1931
    params = flax.traverse_util.flatten_dict(params)
    if filter_fn is None:
        zeroed = {}
    else:
        zeroed = {
            k: jnp.zeros_like(v)
            for k, v in params.items()
            if not filter_fn("/".join(k))
        }
    non_zeros_keys = [k for k in params if k not in zeroed]
    num_zeros = len(zeroed)
    num_non_zeros = len(params) - num_zeros
    assert num_non_zeros > 0, "All gradients zeroed."
    num_preserved_params = sum([params[k].size for k in non_zeros_keys])

    if verbose:
        if num_non_zeros < num_zeros:
            for k in non_zeros_keys:
                logging.info("Keeping grads/params for layer: %s", "/".join(k))
        else:
            for k in zeroed:
                logging.info("Zeroing grads/params for layer: %s", "/".join(k))

    params.update(zeroed)
    params = flax.traverse_util.unflatten_dict(params)
    # do we need this?
    # params = flax.core.frozen_dict.freeze(params)
    if return_num_preserved_params:
        return params, num_preserved_params
    return params


def parameter_filter_from_str(
    specs: Optional[str],
) -> Optional[Callable[[str], bool]]:
    """Simple syntax for parameter filters."""
    if specs is None or not specs:
        # keep all gradients.
        return None
    elif specs.startswith("only_"):
        # keep only gradients of params matching the given name
        name = specs[len("only_") :]
        return lambda x: name in x
    elif specs.startswith("not_"):
        # keep all gradients except for params matching the given name
        name = specs[len("not_") :]
        return lambda x: name not in x
    else:
        raise ValueError(
            "Parameter filter specs be empty or"
            + " they should start with 'only_' or 'not_'."
        )


def load_checkpoint(
    dir_or_ckpt, model_state, weight_state, load_linear_state=True
):
    def load_fn(ws):
        if load_linear_state:
            return checkpoints.restore_checkpoint(
                dir_or_ckpt, (0, model_state, ws)
            )
        else:
            if ws is None:
                return None, None, None
            output = checkpoints.restore_checkpoint(dir_or_ckpt, None)
            loaded_weight_dict = output[
                "2"
            ]  # need only the main params of weighting networks
            new_ws = ws.replace(
                step=loaded_weight_dict["step"],
                params=loaded_weight_dict["params"],
                opt_state=loaded_weight_dict["opt_state"],
                linear_params=model_state.params,  # XXX dirty hack...
                linear_opt_state=model_state.opt_state,  # XXX dirty hack...
                linear_step=model_state.step,
            )
            return None, None, new_ws

    if weight_state is None:
        # We do not need the weight state even when the checkpoint has one.
        step, model_state, _ = load_fn(None)
    else:
        try:
            step, model_state, weight_state = load_fn(weight_state)
        except AttributeError:
            # If the checkpoint does not have a weight state,
            # keep the initial weight state.
            step, model_state, _ = load_fn(None)
    return model_state, weight_state, step


def save_checkpoint(
    workdir,
    step,
    model_state,
    weight_state,
    keep_every_n_steps=None,
):
    checkpoints.save_checkpoint(
        workdir,
        (step, model_state, weight_state),
        step,
        keep_every_n_steps=keep_every_n_steps,
    )
    log_parameter_shapes(
        os.path.join(workdir, "main_model.param_shapes"),
        model_state.params,
    )
    if weight_state is not None:
        log_parameter_shapes(
            os.path.join(workdir, "weight_model.param_shapes"),
            weight_state.params,
        )

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
from typing import Any, Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training import train_state


class META_GRADIENT_METHOD:
    SOFT = "soft"
    SOBA = "soba"
    CLASSIFIER = "classifier"
    ANOGRAD = "anograd"
    CDS = "cds"
    NONE = "none"


DTYPES = {"bf16": jnp.bfloat16, "f32": jnp.float32, "f16": jnp.float16}


class MLP(nn.Module):
    """
    Same architecture as in Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting
    """

    num_hidden_units: int = 100
    reduce: bool = True
    dtype: Any = jnp.float32
    normalize: bool = True
    use_softmax: bool = True

    @nn.compact
    def __call__(self, inputs):
        if self.reduce:
            x = inputs.sum(axis=-1)
        x = nn.Dense(self.num_hidden_units, dtype=self.dtype)(inputs)
        x = nn.relu(x)
        x = nn.Dense(1, dtype=self.dtype)(x)
        if self.use_softmax:
            x = nn.softmax(x)
        else:
            x = nn.sigmoid(x)
            if self.normalize:
                x = x / jnp.sum(x)

        return x


class TrainState(train_state.TrainState):
    """Expand train state to have different apply functions."""

    # generic task loss function
    generic_loss_fn: Callable = struct.field(pytree_node=False)

    # specific task loss function
    specific_loss_fn: Callable = struct.field(pytree_node=False)

    # generic task, train-time apply_fn
    generic_fn: Callable = struct.field(pytree_node=False)

    # specific task, train-time apply_fn
    specific_fn: Callable = struct.field(pytree_node=False)

    # generic task, inference-time apply_fn
    eval_generic_fn: Callable = struct.field(pytree_node=False)

    # specific task, inference-time apply_fn
    eval_specific_fn: Callable = struct.field(pytree_node=False)


class EMATrainState(train_state.TrainState):
    """Train state with additional EMA parameters."""

    ema_params: flax.core.frozen_dict.FrozenDict

    def apply_gradients(self, *, grads, **kwargs):
        grads, ema_params = grads
        new_state = super().apply_gradients(grads=grads, **kwargs)
        new_state = new_state.replace(ema_params=ema_params)
        return new_state


class LinearTrainState(train_state.TrainState):
    """Expand train state to have different apply functions for SOBA."""

    # additionnal linear parameters for SOBA
    linear_params: flax.core.frozen_dict.FrozenDict

    # additionnal optimizer for the linear system
    linear_tx: optax.GradientTransformation = struct.field(pytree_node=False)

    # additionnal optimizer for the linear system
    linear_opt_state: optax.OptState

    linear_step: int

    def apply_linear_gradients(self, *, grads, **kwargs):
        """Updates `step`, `linear_params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx_linear.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
        grads: Gradients that have the same pytree structure as `.params`.
        **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
        An updated instance of `self` with `linear_step` incremented by one, `linear_params`
        and `linear_opt_state` updated by applying `grads`, and additional attributes
        replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.linear_tx.update(
            grads, self.linear_opt_state, self.linear_params
        )
        new_params = optax.apply_updates(self.linear_params, updates)
        return self.replace(
            linear_step=self.linear_step + 1,
            linear_params=new_params,
            linear_opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(
        cls, *, apply_fn, params, tx, linear_params, linear_tx, **kwargs
    ):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        linear_opt_state = linear_tx.init(linear_params)
        return cls(
            step=0,
            linear_step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            linear_params=linear_params,
            linear_tx=linear_tx,
            linear_opt_state=linear_opt_state,
            **kwargs,
        )


def get_monitor_grad(grad_dict, name_prefix="GradientMonitor"):
    grad_dict = flax.traverse_util.flatten_dict(grad_dict)
    select_key = lambda k: any([s.startswith(name_prefix) for s in k])
    transform_key = lambda k: "/".join(k[:-1])
    grad_dict = {
        transform_key(k): v for k, v in grad_dict.items() if select_key(k)
    }
    return grad_dict


def zero_monitor_grad(grad_dict, name_prefix="GradientMonitor"):
    grad_dict = flax.traverse_util.flatten_dict(grad_dict)
    select_key = lambda k: any([s.startswith(name_prefix) for s in k])
    transform_value = lambda v: jnp.zeros_like(v)
    grad_dict = {
        k: transform_value(v) if select_key(k) else v
        for k, v in grad_dict.items()
    }
    grad_dict = flax.traverse_util.unflatten_dict(grad_dict)
    return flax.core.frozen_dict.freeze(grad_dict)


class GradientMonitor(nn.Module):
    """Identity layer at forward with parameter gradients of x -> x + param."""

    num_features: int = 64
    param_dtype = jnp.float32
    param_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        bias = self.param(
            "bias", self.param_init, (self.num_features,), self.param_dtype
        )
        if self.num_features > jnp.size(x):
            bias = bias[: jnp.size(x)]
        elif self.num_features < jnp.size(x):
            pad = jnp.size(x) - self.num_features
            bias = jnp.concatenate(bias, jnp.zeros(pad, self.param_dtype))
        bias = jnp.reshape(bias, x.shape)
        return x + bias - jax.lax.stop_gradient(bias)


def save_then_zero_monitored_grad(
    train_state,
    grads,
    path,
    keys=("GradientMonitor_0",),
    save_frequency=1,
):
    """Save monitor gradients in numpy files and zero them."""
    grads_mon = get_monitor_grad(grads)
    step = int(train_state.step)
    if step % save_frequency == 0:
        for k in keys:
            np.save(
                os.path.join(path, "grad_%s_%05d.npy" % (k, step)),
                np.array(grads_mon[k]),
            )
    return zero_monitor_grad(grads)

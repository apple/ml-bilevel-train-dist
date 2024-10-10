#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Unified interface for no parallelism and multi-accelerator parallelism."""
import os

import jax
from flax import jax_utils
from flax.training import common_utils

JAX_NO_PMAP = not os.getenv("JAX_NO_PMAP", 0) == 0

if JAX_NO_PMAP is None or JAX_NO_PMAP == 0:
    enabled = True
    devices = jax.local_devices()
    pmap = jax.pmap

    def shard(xs):
        local_device_count = jax.local_device_count()
        return jax.tree_util.tree_map(
            lambda x: x.reshape((local_device_count, -1) + x.shape[1:])
            if not hasattr(x, "sharding_spec")
            else x,
            xs,
        )

    unshard = lambda t: jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:]), t
    )
    shard_prng_key = common_utils.shard_prng_key
    psum = jax.lax.psum
    pmean = jax.lax.pmean
    replicate = lambda x: jax_utils.replicate(x) if x is not None else None
    unreplicate = lambda x: jax_utils.unreplicate(x) if x is not None else None
    device_prefetch = lambda x: jax.device_put_sharded(list(x), devices)
else:
    enabled = False
    shard = lambda x: x
    unshard = lambda x: x
    shard_prng_key = lambda x: x
    psum = lambda x, _: x
    pmean = lambda x, _: x
    replicate = lambda x: x
    unreplicate = lambda x: x
    device_prefetch = lambda x: x

    def pmap(*args, **kwargs):
        kwargs = {  # discard extra kwargs
            k: v
            for k, v in kwargs.items()
            if k not in ["donate_argnums", "axis_name", "in_axes"]
        }
        kwargs["static_argnums"] = kwargs.pop(
            "static_broadcasted_argnums", None
        )
        return jax.jit(*args, **kwargs)

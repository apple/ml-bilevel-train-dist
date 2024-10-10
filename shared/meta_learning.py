#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Meta-learning methods."""

import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import ml_collections

from shared import data_pipeline as dp
from shared import learning
from shared import model as sm
from shared import parallel


def _tree_vdot(a, b):
    """Dot product between two trees of vectors."""
    a, _ = jax.tree_util.tree_flatten(a)
    b, _ = jax.tree_util.tree_flatten(b)
    return sum(jnp.vdot(x, y) for x, y in zip(a, b))


def _tree_cosine(a, b, eps):
    """(Batched) Dot product between two trees of vectors.
    note: <a> can be contain batch of vectors (matrices).
    """

    def dot_prod_norm_fn(x, y):
        dim = y.size
        x = x.reshape((-1, dim))
        y = y.flatten()
        x2 = (x * x).sum(axis=-1)
        y2 = (y * y).sum()
        return jnp.matmul(x, y), x2, y2

    a, _ = jax.tree_util.tree_flatten(a)
    b, _ = jax.tree_util.tree_flatten(b)
    dpn = zip(*[dot_prod_norm_fn(x, y) for x, y in zip(a, b)])
    dot_prod, norm_a2, norm_b2 = [jnp.stack(x).sum(axis=0) for x in dpn]

    cosine = dot_prod / (eps + jnp.sqrt(norm_a2 * norm_b2))
    cosine = cosine[0] if cosine.shape[0] == 1 else cosine
    return cosine


def split_batch(inputs, n=2):
    """Split a batch in equal subparts."""
    outputs = {
        k: (
            jax.random.split(v, n)
            if k == "rng"
            else v.reshape((n, -1) + v.shape[1:])
        )
        for k, v in inputs.items()
    }
    return [{k: v[i] for k, v in outputs.items()} for i in range(n)]


def sample_batch(rng, batch, weights, batch_size, sample="rand"):
    if sample == "rand":
        ids = jax.random.permutation(rng, batch["inputs"].shape[0])[:batch_size]
    elif sample == "top":
        per_example_weights = jnp.sum(weights, axis=range(1, weights.ndim))
        ids = jnp.argsort(-per_example_weights)[:batch_size]
    elif sample == "multi" or sample == "multi_repl":
        per_example_weights = jnp.sum(weights, axis=range(1, weights.ndim))
        ids = jax.random.choice(
            rng,
            batch["inputs"].shape[0],
            (batch_size,),
            p=per_example_weights,
            replace=(sample == "multi_repl"),
        )

    return {
        k: v if k in dp.NON_SELECTABLE_FIELDS else v[ids]
        for k, v in batch.items()
    }


def _adam_grad_scaling(grad, model_state, eps=1e-8, decay=0.999):
    adam_state = model_state.opt_state[0]
    mu, count = adam_state.mu, adam_state.count
    bias_c = 1.0 - decay**count
    scale_fn = lambda g, v: (g / (jnp.sqrt(v / bias_c) + eps)).astype(g.dtype)
    return jax.tree_map(scale_fn, grad, mu)


def _scaled_default_weighting(default_weight_fn):
    """Combine default weights with a single weight per example for 2D+ dim."""

    def fn(batch, weights):
        default_w = default_weight_fn(batch)
        weights = jnp.expand_dims(weights, range(weights.ndim, default_w.ndim))
        return default_w * weights.astype(default_w.dtype)

    return fn


def specific_vs_generic_classifier_gradient(
    classifier_state,
    generic_batch,
    specific_batch,
    default_weight_fn,
    top_frac: float = 0.25,
):
    assert top_frac <= 1.0 and top_frac > 0

    def classify_fn(params, batch):
        return classifier_state.apply_fn({"params": params}, batch).flatten()

    def classification_loss_fn(logits, labels):
        """Binary log loss."""
        m = labels * logits
        acc = (m > 0).mean()
        loss = -jax.nn.log_sigmoid(m).mean()
        return loss, acc

    def specific_vs_generic_loss_fn(w_params, generic_inputs, specific_inputs):
        # Negative examples from the generic batch.
        generic_logits = classify_fn(w_params, generic_inputs)
        examples = [(generic_logits, -1.0)]
        # Positive examples from the specific batch.
        specific_logits = classify_fn(w_params, specific_inputs)
        examples += [(specific_logits, 1.0)]

        loss, acc = zip(*[classification_loss_fn(x, y) for x, y in examples])
        # Same weight for both sets regardless of their batch size.
        loss, acc = jnp.array(loss).mean(), jnp.array(acc).mean()

        # Most positive examples from the generic set.
        top_size = int(jnp.size(generic_logits) * top_frac)
        top_set = sample_batch(
            None, generic_inputs, generic_logits, top_size, "top"
        )
        return loss, (acc, top_set)

    grad_fn = jax.value_and_grad(specific_vs_generic_loss_fn, has_aux=True)
    (loss, (acc, subset)), classif_grad = grad_fn(
        classifier_state.params,
        generic_batch,
        specific_batch,
    )
    metrics = loss, acc, 0.0
    return classif_grad, metrics, default_weight_fn(subset), subset


def meta_gradient(
    model_state,
    weight_state,
    train_batch,
    meta_train_batch,
    rng,
    config,
    default_weight_fn,
    parameter_filter: Optional[Callable[[str], bool]] = None,
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

    # Read config.
    lr = config.learning_rate
    replace = config.get("replace", False)

    # Learning rate is zero? We modify the sgd update.
    # This corresponds to the limit as lr->0 without numerical issues.
    assert lr >= 0.0
    if lr == 0.0:
        sgd_update_fn = lambda p, g: p - (g - jax.lax.stop_gradient(g))
    else:
        sgd_update_fn = lambda p, g: p - lr * g

    # We use the same batch size the meta model and the
    # downsampled generic batch.
    batch_size = meta_train_batch["inputs"].shape[0]

    rng, subrng = jax.random.split(rng, 2)

    def weight_fn(params, batch):
        return weight_state.apply_fn({"params": params}, batch)

    def post_update_loss_fn(w_params):
        train_weights = weight_fn(w_params, train_batch)
        batch = sample_batch(rng, train_batch, train_weights, batch_size)

        # Recompute the weights for proper normalization and faster backward:
        weights = weight_fn(w_params, batch)
        m_params = model_state.params
        train_grads, *_ = generic_fn(m_params, batch, weights)

        # Optional filtering of the training gradients.
        train_grads = learning.selective_zeros(
            train_grads,
            parameter_filter,
        )
        updated_m_params = jax.tree_map(sgd_update_fn, m_params, train_grads)
        meta_loss, meta_acc = specific_fn(updated_m_params, meta_train_batch)
        return meta_loss, (meta_acc, train_weights)

    weight_params = weight_state.params
    grad_fn = jax.value_and_grad(post_update_loss_fn, has_aux=True)
    (meta_loss, (meta_acc, train_weights)), meta_grad = grad_fn(weight_params)

    # Get the top scoring samples from the batch:
    sample_train_batch = sample_batch(
        subrng,
        train_batch,
        train_weights,
        batch_size,
        sample="multi_repl" if replace else "multi",
    )
    sample_train_weights = default_weight_fn(sample_train_batch)

    aux_loss = jnp.zeros_like(meta_loss)  # no auxilary loss for now.
    metrics = meta_loss, meta_acc, aux_loss
    return meta_grad, metrics, sample_train_weights, sample_train_batch


def anograd(
    model_state,
    weight_state,
    generic_batch,
    specific_batch,
    rng,
    default_weight_fn,
    parameter_filter: Optional[Callable[[str], bool]] = None,
    eps=1e-6,
    sampling="multi",
    hessian_free=True,
):
    """Aligned normalized gradient method."""
    generic_fn = learning.apply_model(
        model_state.generic_fn,
        loss_fn=model_state.generic_loss_fn,
        need_gradient=True,
        normalized_loss=False,
    )
    specific_fn, hvp_specific_fn = learning.apply_model(
        model_state.specific_fn,
        loss_fn=model_state.specific_loss_fn,
        need_gradient=True,
        normalized_loss=False,
        need_hvp=True,
    )
    rngs = list(jax.random.split(rng, 2))
    small_bsz = specific_batch["inputs"].shape[0]
    g_sample = sample_batch(rngs[0], generic_batch, None, small_bsz)
    scaled_default_weighting = _scaled_default_weighting(default_weight_fn)

    def weight_fn(params, batch):
        return weight_state.apply_fn({"params": params}, batch).flatten()

    def concentration_metric(w):
        w2 = w * w
        arg_top = jnp.argsort(-w2)[:small_bsz]
        return w2[arg_top].sum() / w2.sum()

    def meta_loss(w_params):
        # mean generic gradient
        m_params = model_state.params
        w = scaled_default_weighting(g_sample, weight_fn(w_params, g_sample))
        g_grads = generic_fn(m_params, g_sample, w)[0]
        g_grads = learning.selective_zeros(g_grads, parameter_filter)

        # mean specific gradient
        w = default_weight_fn(specific_batch)
        s_grads = specific_fn(m_params, specific_batch, w)[0]

        loss = -_tree_vdot(g_grads, s_grads)
        if hessian_free:
            norm2 = _tree_vdot(g_grads, g_grads)
        else:
            hvp = hvp_specific_fn(
                m_params,
                g_grads,
                specific_batch,
                default_weight_fn(specific_batch),
            )
            norm2 = _tree_vdot(g_grads, hvp)
        loss /= jnp.sqrt(norm2) + eps
        return loss

    # the weighting model minimizes this loss.
    grad_fn = jax.value_and_grad(meta_loss)
    meta_loss, w_grad = grad_fn(weight_state.params)

    # select the subset of the initial large batch.
    w = weight_fn(weight_state.params, generic_batch)
    c_metric = concentration_metric(w)
    w = scaled_default_weighting(generic_batch, w)
    prob = w / w.sum()
    g_subset = sample_batch(
        rng=rngs[1],
        batch=generic_batch,
        weights=prob,
        batch_size=small_bsz,
        sample=sampling,
    )

    metrics = meta_loss, 0.0, c_metric
    return w_grad, metrics, default_weight_fn(g_subset), g_subset


def meta_soba_gradient(
    model_state,
    weight_state,
    train_batch,
    meta_train_batch,
    rng,
    default_weight_fn,
    sampling="multi",
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

    batch_size = meta_train_batch["inputs"].shape[0]

    rng, subrng = jax.random.split(rng, 2)

    def weight_fn(params, batch):
        return weight_state.apply_fn({"params": params}, batch)

    def train_grads_from_weights(w_params, m_params, train_inputs):
        train_weights = weight_fn(w_params, train_inputs)
        train_grads, *_ = generic_fn(m_params, train_inputs, train_weights)
        return train_grads

    def cross_vector_product(w_params, m_params, linear_params, train_inputs):
        grad_fn = lambda w_params: train_grads_from_weights(
            w_params, m_params, train_inputs
        )
        _, vjp = jax.vjp(grad_fn, (w_params))
        return vjp(linear_params)[0]

    model_params = model_state.params
    weight_params = weight_state.params
    linear_params = weight_state.linear_params
    train_weights = weight_fn(weight_params, train_batch)

    # Get the top scoring samples from the batch:
    train_batch = sample_batch(
        subrng,
        train_batch,
        train_weights,
        batch_size,
        sample=sampling,
    )
    train_weights = default_weight_fn(train_batch)

    meta_grad = cross_vector_product(
        weight_params, model_params, linear_params, train_batch
    )
    meta_loss, meta_acc = specific_fn(model_params, meta_train_batch)
    aux_loss = jnp.zeros_like(meta_loss)  # no auxilary loss for now.
    metrics = meta_loss, meta_acc, aux_loss
    return meta_grad, metrics, train_weights, train_batch


def linear_soba_gradient(
    model_state,
    weight_state,
    train_batch,
    meta_train_batch,
    rng,
    default_weight_fn,
    hessian_free=False,
    change_output_batch=False,
    hessian_regul=0.0,
):
    _, generic_hvp = learning.apply_model(
        model_state.generic_fn,
        loss_fn=model_state.generic_loss_fn,
        need_gradient=True,
        need_hvp=True,
    )
    specific_fn = learning.apply_model(
        model_state.specific_fn,
        loss_fn=model_state.specific_loss_fn,
        need_gradient=True,
        weight_fn=default_weight_fn,
    )

    batch_size = meta_train_batch["inputs"].shape[0]

    rng, subrng = jax.random.split(rng, 2)

    def weight_fn(params, batch):
        return weight_state.apply_fn({"params": params}, batch)

    model_params = model_state.params
    weight_params = weight_state.params
    linear_params = weight_state.linear_params
    train_weights = weight_fn(weight_params, train_batch)

    # Get the top scoring samples from the batch:
    new_train_batch = sample_batch(
        subrng, train_batch, train_weights, batch_size, sample="multi"
    )
    new_train_weights = default_weight_fn(new_train_batch)

    specific_grad, meta_loss, meta_acc = specific_fn(
        model_params,
        meta_train_batch,
    )  # gradient of outer loss wrt inner params
    if hessian_free:
        linear_direction = specific_grad
    else:
        hvp_val = generic_hvp(
            model_params, linear_params, new_train_batch, new_train_weights
        )  # hvp of inner fn in the linear direction
        linear_direction = jax.tree_map(
            lambda a, b, c: (1.0 - hessian_regul) * a + hessian_regul * b + c,
            hvp_val,
            linear_params,
            specific_grad,
        )
    aux_loss = jnp.zeros_like(meta_loss)  # no auxilary loss for now.
    metrics = meta_loss, meta_acc, aux_loss
    if change_output_batch:
        train_weights = new_train_weights
        train_batch = new_train_batch
    return linear_direction, metrics, train_weights, train_batch


def reweight_function(
    model_state,
    weight_state,
    train_batch,
    meta_train_batch,
    rng,
    train_lr,
    default_weight_fn,
):
    """
    Implementation of Learning to Reweight Examples for Robust Deep Learning
    """
    generic_fn = learning.apply_model(
        model_state.generic_fn,
        loss_fn=model_state.generic_loss_fn,
        need_gradient=True,
        normalized_loss=False,
    )
    specific_fn = learning.apply_model(
        model_state.specific_fn,
        loss_fn=model_state.specific_loss_fn,
        need_gradient=False,
        weight_fn=default_weight_fn,
    )

    rng, subrng = jax.random.split(rng, 2)

    scaled_default_weighting = _scaled_default_weighting(default_weight_fn)

    def sgd_update_fn(params, grad_params, lr):
        return jax.tree_map(lambda a, b: a - lr * b, params, grad_params)

    def post_update_loss_fn(
        weights, m_params, train_inputs, meta_train_input, train_lr
    ):
        weights = scaled_default_weighting(train_inputs, weights)
        train_grads, _, _ = generic_fn(m_params, train_inputs, weights)
        updated_m_params = sgd_update_fn(m_params, train_grads, train_lr)
        meta_loss, meta_acc = specific_fn(updated_m_params, meta_train_input)
        return meta_loss, meta_acc

    model_params = model_state.params
    batch_size = meta_train_batch["inputs"].shape[0]
    grad_fn = jax.value_and_grad(post_update_loss_fn, has_aux=True)
    (meta_loss, meta_acc), weight_grad = grad_fn(
        jnp.zeros(batch_size),
        model_params,
        train_batch,
        meta_train_batch,
        train_lr,
    )
    # Compute predicted weights from gradients: same as commented code below with lax
    # if jnp.all(weight_grad > 0.):  # default to uniform
    #     train_weights = default_weight_fn(train_batch)
    # else:
    #     train_weights = jnp.maximum(-weight_grad, 0.)

    default_weights_1d = default_weight_fn(train_batch)
    default_weights_1d = jnp.sum(
        default_weights_1d,
        axis=list(range(1, default_weights_1d.ndim)),
    )
    train_weights = jax.lax.cond(
        jnp.all(weight_grad > 0.0),
        lambda _: default_weights_1d,
        functools.partial(jnp.maximum, 0.0),
        -weight_grad,
    )
    train_weights /= jnp.sum(train_weights)  # normalize
    train_weights = scaled_default_weighting(train_batch, train_weights)
    aux_loss = jnp.zeros_like(meta_loss)  # no auxilary loss for now.
    metrics = meta_loss, meta_acc, aux_loss
    return None, metrics, train_weights, train_batch


def frozen_evaluation(
    model_state,
    weight_state,
    train_batch,
    meta_train_batch,
    rng,
    default_weight_fn,
):
    """
    Get the weights from the batch and don't do meta learning.
    """
    del model_state  # the model state is not used.
    # We use the same batch size the meta model and the
    # downsampled generic batch.
    batch_size = meta_train_batch["inputs"].shape[0]

    rng, subrng = jax.random.split(rng, 2)

    def weight_fn(params, batch):
        return weight_state.apply_fn({"params": params}, batch)

    weight_params = weight_state.params

    train_weights = weight_fn(weight_params, train_batch)
    # Get the top scoring samples from the batch:
    train_batch = sample_batch(
        subrng, train_batch, train_weights, batch_size, sample="multi"
    )
    train_weights = default_weight_fn(train_batch)

    metrics = 0.0, 0.0, 0.0
    meta_grad = None
    return meta_grad, metrics, train_weights, train_batch


def dot_prod_with_delta_params(
    model_state,
    batch,
    is_generic,
    delta_params,
    default_weight_fn,
    eps=1e-6,
):
    """Compute dot product between grad_loss of indivual samples of batch_a
    and grad_mean_loss over batch_b.

    The dot product is normalized on for batch_b (normalizing by the norm of
    individual example gradients of batch_a see implementation below).
    """

    def grad_fn(*args, is_generic=True):
        if is_generic:
            apply_model = functools.partial(
                learning.apply_model,
                model_fn=model_state.eval_generic_fn,
                loss_fn=model_state.generic_loss_fn,
            )
        else:
            apply_model = functools.partial(
                learning.apply_model,
                model_fn=model_state.eval_specific_fn,
                loss_fn=model_state.specific_loss_fn,
            )
        return apply_model(need_gradient=True, normalized_loss=False)(*args)[0]

    def mean_dot_prod(w):
        w = _scaled_default_weighting(default_weight_fn)(batch, w)
        grads = grad_fn(model_state.params, batch, w, is_generic=is_generic)
        norm = eps + jnp.sqrt(_tree_vdot(delta_params, delta_params))
        # delta_params is a model update: opposite direction of loss gradient.
        return -_tree_vdot(grads, delta_params) / norm

    batch_size = batch["inputs"].shape[0]
    uniform = jnp.ones(batch_size) / batch_size
    per_item_dot_prod = jax.grad(mean_dot_prod)(uniform)
    return per_item_dot_prod


def gradient_cosine(
    model_state_a,
    batch_a,
    batch_a_is_generic,
    model_state_b,
    batch_b,
    batch_b_is_generic,
    default_weight_fn,
    eps=1e-6,
):
    """Compute dot product between grad_loss of indivual samples of batch_a
    and grad_mean_loss over batch_b.

    The dot product is normalized on for batch_b (normalizing by the norm of
    individual example gradients of batch_a see implementation below).
    """
    m_state = model_state_a  # we assume a, b function pointers are the same.

    def grad_fn(*args, is_generic=True):
        if is_generic:
            apply_model = functools.partial(
                learning.apply_model,
                model_fn=m_state.eval_generic_fn,
                loss_fn=m_state.generic_loss_fn,
            )
        else:
            apply_model = functools.partial(
                learning.apply_model,
                model_fn=m_state.eval_specific_fn,
                loss_fn=m_state.specific_loss_fn,
            )
        return apply_model(need_gradient=True, normalized_loss=False)(*args)[0]

    def mean_dot_prod(w):
        a_grads = grad_fn(
            model_state_a.params,
            batch_a,
            _scaled_default_weighting(default_weight_fn)(batch_a, w),
            is_generic=batch_a_is_generic,
        )
        b_grads = grad_fn(
            model_state_b.params,
            batch_b,
            default_weight_fn(batch_b),
            is_generic=batch_b_is_generic,
        )
        norm = eps + jnp.sqrt(_tree_vdot(b_grads, b_grads))
        return _tree_vdot(a_grads, b_grads) / norm

    batch_size_a = batch_a["inputs"].shape[0]
    uniform_a = jnp.ones(batch_size_a) / batch_size_a
    per_item_dot_prod = jax.grad(mean_dot_prod)(uniform_a)
    return per_item_dot_prod


def oom_gradient_cosine(
    model_state_a,
    batch_a,
    batch_a_is_generic,
    model_state_b,
    batch_b,
    batch_b_is_generic,
    default_weight_fn,
    eps=1e-6,
):
    """Compute cosine between grad_loss of indivual samples of batch_a
    and grad_mean_loss over batch_b. Does not run OOM. Naive implementation
    to investigate later.
    """
    fns = [
        (model_state_a.eval_specific_fn, model_state_a.specific_loss_fn),
        (model_state_a.eval_generic_fn, model_state_a.generic_loss_fn),
    ]

    m_fn, l_fn = fns[batch_a_is_generic]
    loss_fn_a = learning.apply_model(
        m_fn,
        loss_fn=l_fn,
        need_gradient=False,
        reduce=False,
        weight_fn=default_weight_fn,
    )

    m_fn, l_fn = fns[batch_b_is_generic]
    loss_fn_b = learning.apply_model(
        m_fn,
        loss_fn=l_fn,
        need_gradient=True,
        reduce=True,
        weight_fn=default_weight_fn,
    )

    def loss_a_per_example(p):
        """Loss reduced along all but the first dimensions."""
        loss, _, w = loss_fn_a(p, batch_a)
        return (loss * w).sum(axis=tuple(range(1, w.ndim)))

    grad_a = jax.jacrev(loss_a_per_example)(model_state_a.params)
    grad_b = loss_fn_b(model_state_b.params, batch_b)[0]

    return _tree_cosine(grad_a, grad_b, eps)


def get_meta_step_fn(config, default_weight_fn, meta_train_schedule_fn):
    method = config.meta_gradient_method
    if method == sm.META_GRADIENT_METHOD.SOFT:
        parameter_filter = learning.parameter_filter_from_str(
            config.soft_select_params.get("parameter_filter", None),
        )

        meta_gradient_fn = functools.partial(
            meta_gradient,
            config=config.soft_select_params,
            default_weight_fn=default_weight_fn,
            parameter_filter=parameter_filter,
        )

    elif method == sm.META_GRADIENT_METHOD.SOBA:
        linear_gradient_fn = functools.partial(
            linear_soba_gradient,
            default_weight_fn=default_weight_fn,
            hessian_free=config.soba_params.hessian_free,
            hessian_regul=config.soba_params.get("hessian_regul", 0.0),
        )

        meta_gradient_fn = functools.partial(
            meta_soba_gradient,
            default_weight_fn=default_weight_fn,
            sampling=config.soba_params.get("sampling", "multi"),
        )

    elif method == sm.META_GRADIENT_METHOD.ANOGRAD:
        meta_gradient_fn = functools.partial(
            anograd,
            default_weight_fn=default_weight_fn,
            parameter_filter=learning.parameter_filter_from_str(
                config.anograd_params.parameter_filter,
            ),
            sampling=config.anograd_params.sampling,
            hessian_free=config.anograd_params.hessian_free,
        )

    elif method == sm.META_GRADIENT_METHOD.CLASSIFIER:

        def meta_gradient_fn(
            m_state,
            w_state,
            g_batch,
            m_batch,
            rng,
        ):
            del rng  # deterministic
            del m_state  # does not depends on the current model.
            args = (w_state, g_batch, m_batch)
            return specific_vs_generic_classifier_gradient(
                *args,
                top_frac=config.meta_classifier_params.top_frac,
                default_weight_fn=default_weight_fn,
            )

    else:
        raise ValueError(
            "meta_gradient_method should be in "
            + ", ".join(
                v
                for k, v in sm.META_GRADIENT_METHOD.__dict__.items()
                if "__" not in k
            )
        )

    use_soba = config.meta_gradient_method == sm.META_GRADIENT_METHOD.SOBA
    use_frozen_update = "frozen_weight_curriculum" in config
    if use_frozen_update:
        meta_gradient_fn = functools.partial(
            frozen_evaluation,
            default_weight_fn=default_weight_fn,
        )

        @functools.partial(
            parallel.pmap,
            axis_name="batch",
            static_broadcasted_argnums=(5,),
        )
        def meta_train_step(
            weight_state,
            model_state,
            g_batch,
            m_batch,
            rng,
            meta_step,
        ):
            del meta_step
            _, meta_metrics, g_weights, g_batch = meta_gradient_fn(
                model_state,
                weight_state,
                g_batch,
                m_batch,
                rng,
            )
            meta_grad_norm = 0.0
            meta_metrics = (meta_grad_norm,) + meta_metrics
            return weight_state, g_batch, g_weights, meta_metrics

    elif use_soba:

        @functools.partial(
            parallel.pmap,
            axis_name="batch",
            donate_argnums=(0,),
            static_broadcasted_argnums=(5, 6),
        )
        def soba_meta_train_step(
            weight_state,
            model_state,
            g_batch,
            m_batch,
            rng,
            linear_step,
            meta_step,
        ):
            if linear_step:
                (
                    linear_grads,
                    meta_metrics,
                    g_weights,
                    g_batch,
                ) = linear_gradient_fn(
                    model_state,
                    weight_state,
                    g_batch,
                    m_batch,
                    rng,
                )
                linear_grads = parallel.pmean(linear_grads, "batch")
                weight_state, meta_grad_norm = learning.update_model(
                    weight_state, linear_grads, linear=True
                )
            meta_grads, meta_metrics, g_weights, g_batch = meta_gradient_fn(
                model_state,
                weight_state,
                g_batch,
                m_batch,
                rng,
            )
            meta_grads = parallel.pmean(meta_grads, "batch")
            new_weight_state, meta_grad_norm = learning.update_model(
                weight_state, meta_grads
            )
            meta_metrics = (meta_grad_norm,) + meta_metrics
            if meta_step:
                weight_state = new_weight_state
            return weight_state, g_batch, g_weights, meta_metrics

    else:

        @functools.partial(
            parallel.pmap,
            axis_name="batch",
            donate_argnums=(0,),
            static_broadcasted_argnums=(5,),
        )
        def meta_train_step(
            weight_state,
            model_state,
            g_batch,
            m_batch,
            rng,
            meta_step,
        ):
            meta_grads, meta_metrics, g_weights, g_batch = meta_gradient_fn(
                model_state,
                weight_state,
                g_batch,
                m_batch,
                rng,
            )
            meta_grads = parallel.pmean(meta_grads, "batch")

            new_weight_state, meta_grad_norm = learning.update_model(
                weight_state, meta_grads
            )
            meta_metrics = (meta_grad_norm,) + meta_metrics
            if meta_step:
                weight_state = new_weight_state
            return weight_state, g_batch, g_weights, meta_metrics

    def meta_train_step_fn(
        weight_state, model_state, g_batch, m_batch, rng, step
    ):
        step_dict = meta_train_schedule_fn(step)
        if use_soba and not use_frozen_update:
            return soba_meta_train_step(
                weight_state,
                model_state,
                g_batch,
                m_batch,
                rng,
                step_dict["linear_step"],
                step_dict["meta_step"],
            )
        else:
            return meta_train_step(
                weight_state,
                model_state,
                g_batch,
                m_batch,
                rng,
                step_dict["meta_step"],
            )

    return meta_train_step_fn


def meta_train_schedule(config):
    meta_start = config.get("meta_train_full_start", 0)
    schedule = config.get("meta_train_schedule", (1, 1))
    if config.meta_gradient_method == sm.META_GRADIENT_METHOD.SOBA:
        if len(schedule) == 2:  # linear_step = meta_step
            schedule = (schedule[0],) + tuple(schedule)
        meta, linear_step, trn = schedule
        return lambda step: {
            "meta_step": step % meta == 0 or step < meta_start,
            "linear_step": step % linear_step == 0 or step < meta_start,
            "train_step": step % trn == 0,
        }
    elif isinstance(schedule, (tuple, list)):
        meta, trn = schedule
        return lambda step: {
            "meta_step": step % meta == 0 or step < meta_start,
            "train_step": step % trn == 0,
        }
    elif isinstance(schedule, (dict, ml_collections.ConfigDict)):
        # list of step intervals where each type of training is active.
        meta_intervals = schedule["meta_step"]
        train_intervals = schedule["train_step"]

        def in_interval(step, interval):
            s, e = interval
            return step >= s and step <= e

        return lambda step: {
            "meta_step": any(in_interval(step, i) for i in meta_intervals),
            "train_step": any(in_interval(step, i) for i in train_intervals),
        }
    raise ValueError(
        "config.meta_train_schedule = %s unsupported"
        % str(config.meta_train_schedule)
    )

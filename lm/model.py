#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import enum
import functools
import warnings
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state

import lm.data_pipeline as dp
from lm import transformer
from shared import model as sm


class WEIGHTING_MODEL_ARCH:
    NONE = "none"
    CNN = "cnn"
    MLP = "mlp"
    BOW = "bow"


class MODEL_ARCH:
    CNN = "cnn"
    TRANSFORMER = "transformer"


class WEIGHT_TYPE(enum.Enum):
    FLOAT = "float"
    PROBABILITY = "probability"
    POSITIVE = "positive"


class TransformerLM(nn.Module):
    """Wrapper around flax wmt transformer."""

    num_hidden_units: Tuple[int, int, int] = (120_000, 128, 512)
    num_heads: int = 4
    num_layers: int = 3
    shared_embeddings: bool = False
    max_length: int = 128
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    dtype: str = "bf16"
    inference: bool = False
    deterministic: bool = False

    def setup(self):
        vocab_size, emb_dim, mlp_dim = self.num_hidden_units
        config = transformer.TransformerConfig(
            vocab_size=vocab_size,
            output_vocab_size=vocab_size,
            share_embeddings=False,
            logits_via_embedding=self.shared_embeddings,
            dtype=sm.DTYPES[self.dtype],
            emb_dim=emb_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            qkv_dim=emb_dim,
            mlp_dim=mlp_dim,
            max_len=self.max_length,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            deterministic=self.deterministic,
            decode=self.inference,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )
        self.decoder = transformer.Decoder(config=config)

    def decode(
        self,
        targets,
        targets_positions=None,
        targets_segmentation=None,
    ):
        """Applies Transformer decoder-branch on encoded-input and target.

        Args:
          encoded: encoded input data from encoder.
          inputs: input data (only needed for masking).
          targets: target data.
          targets_positions: target subsequence positions for packed examples.
          inputs_segmentation: input segmentation info for packed examples.
          targets_segmentation: target segmentation info for packed examples.

        Returns:
          logits array from transformer decoder.
        """
        dtype = sm.DTYPES[self.dtype]

        # Make padding attention masks.
        if self.inference:
            decoder_mask = None
        else:
            decoder_mask = nn.combine_masks(
                nn.make_attention_mask(targets > 0, targets > 0, dtype=dtype),
                nn.make_causal_mask(targets, dtype=dtype),
            )

        # Add segmentation block-diagonal attention masks if using segmented data.
        if targets_segmentation is not None:
            decoder_mask = nn.combine_masks(
                decoder_mask,
                nn.make_attention_mask(
                    targets_segmentation,
                    targets_segmentation,
                    jnp.equal,
                    dtype=dtype,
                ),
            )

        logits = self.decoder(
            None,
            targets,
            targets_positions=targets_positions,
            decoder_mask=decoder_mask,
        )
        return logits.astype(dtype)

    def __call__(
        self,
        inputs,
    ):
        """Applies Transformer model on the inputs."""
        # Note: we take the label here since the decoder module shift the text
        # to the right at forward during training.
        targets = inputs[dp.FIELDS.LABELS]

        # +1 since datagen gives positions in [1, max_len] with zero
        # reserved for padding but transformer code
        # expect positions to range in [0, max_len - 1]
        targets_positions = inputs[dp.FIELDS.POS_INPUTS]
        targets_positions = jnp.maximum(targets_positions, 1) - 1

        return self.decode(
            targets=targets,
            targets_positions=targets_positions,
            targets_segmentation=None,  # no packing for now.
        )


class WeightingHead(nn.Module):
    replicate_field: Optional[str] = None
    output_type: WEIGHT_TYPE = WEIGHT_TYPE.FLOAT
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, inputs):
        original_inputs, predicted_weights = inputs
        if self.output_type == WEIGHT_TYPE.FLOAT:
            output_transform = lambda x: x.flatten()
        else:
            output_transform = lambda x: jax.nn.softmax(x.flatten())

        x = predicted_weights
        # single weight per sequence or replication over non padding tokens.
        if not self.replicate_field:
            x = x[..., 0]
            y = output_transform(x)
        else:
            mask = original_inputs[self.replicate_field] > 0
            mask = mask.astype(x.dtype)
            negligible_weight = -1.0 / self.epsilon
            x = x * mask + (1 - mask) * negligible_weight
            y = output_transform(x)  # softmax over every tokens in the batch
            y = jnp.reshape(y, mask.shape)  # restore initial shape

        # post softmax scaling
        if self.output_type == WEIGHT_TYPE.POSITIVE:
            a = self.param(
                "post_transform_scale",
                jax.nn.initializers.zeros,
                (),
            )
            y = jax.nn.softplus(a) * y

        return y


class CNNLM(nn.Module):
    """Convolutional neural net."""

    input_field: str = dp.FIELDS.INPUTS
    num_hidden_units: Sequence[int] = (120_000, 768)
    kernel_size: int = 7
    shared_embeddings: bool = False
    output_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs):
        # embeddings
        vocab_size, embed_dim, *conv_features = self.num_hidden_units
        embed = nn.Embed(
            num_embeddings=vocab_size,
            features=embed_dim,
            embedding_init=nn.initializers.normal(stddev=1.0),
        )
        x = inputs[self.input_field]  # shape (bsz, len)
        x = x.astype(jnp.int32)
        x = embed(x)  # shape (bsz, len, dim)

        # conv
        for f in conv_features:
            x = nn.Conv(
                features=f,
                kernel_init=nn.initializers.xavier_uniform(),
                kernel_size=(self.kernel_size,),
                padding=((self.kernel_size - 1, 0),),
            )(x)
            x = nn.relu(x)

        # re-use word embeddings.
        if self.shared_embeddings:  # re-use word embeddings.
            assert self.output_dim is None
            logits = embed.attend(x.astype(jnp.float32))
            # normalize as embedding weights are not init as a linear kernel.
            logits = logits / jnp.sqrt(x.shape[-1])
        else:
            out_dim = vocab_size if self.output_dim is None else self.output_dim
            logits = nn.Dense(features=out_dim)(x)
        return logits


class CNNWeights(nn.Module):
    output_type: str
    input_field: str = dp.FIELDS.INPUTS
    replicate_field: str = dp.FIELDS.POS_INPUTS
    num_hidden_units: Sequence[int] = (120_000, 768)
    kernel_size: int = 7
    eps: float = 1e-6

    @nn.compact
    def __call__(self, inputs):
        # weight per word
        x = CNNLM(
            input_field=self.input_field,
            kernel_size=self.kernel_size,
            num_hidden_units=self.num_hidden_units,
            output_dim=1,
        )(inputs)[..., 0]

        # weight per sequence: mean over non-pad tokens.
        input_pos = "pos_" + self.input_field
        input_mask = (inputs[input_pos] > 0).astype(x.dtype)
        x = (x * input_mask).sum(axis=-1) / (input_mask.sum(axis=-1) + self.eps)
        x = x[..., jnp.newaxis]

        head = WeightingHead(
            replicate_field=self.replicate_field,
            output_type=self.output_type,
        )
        return head((inputs, x))


class BagOfWordsWeight(nn.Module):
    output_type: str
    num_hidden_units: Sequence[int] = (120_000, 768)
    input_field: Union[str, Sequence[str]] = dp.FIELDS.LABELS
    replicate_field: Optional[str] = dp.FIELDS.LABELS
    eps: float = 1e-6

    @nn.compact
    def __call__(self, inputs):
        input_fields = self.input_field
        if isinstance(input_fields, str):
            input_fields = [input_fields]
        num_hidden_units = list(self.num_hidden_units) + [1]

        vocab_size = num_hidden_units.pop(0)
        emb_dim = num_hidden_units.pop(0)
        emb = nn.Embed(  # shared embeddings
            num_embeddings=vocab_size,
            features=emb_dim,
            embedding_init=nn.initializers.normal(stddev=1.0),
        )

        # mean embedding over time for each field
        x = []
        for f in input_fields:
            h = inputs[f]
            mask = (h > 0).astype(h.dtype)
            mask = mask[..., jnp.newaxis]
            h = emb(h)
            h = (h * mask).sum(axis=-2) / (self.eps + mask.sum(axis=-2))
            x.append(h)

        # concatenate field vectors
        x = jnp.concatenate(x, axis=-1)

        # optional MLP
        h = x
        for nhu in num_hidden_units:
            x = nn.Dense(
                nhu,
                kernel_init=nn.initializers.xavier_uniform(),
            )(h)
            h = nn.relu(x)  # no relu for the last layer

        head = WeightingHead(
            output_type=self.output_type,
            replicate_field=self.replicate_field,
        )
        return head((inputs, x))


def init_cnn_model_state(rng, config):
    """Creates initial `TrainState`."""

    cnn = CNNLM(
        num_hidden_units=config.num_hidden_units,
        shared_embeddings=config.shared_embeddings,
    )

    inputs = dp.input_placeholder()
    params = cnn.init(rng, inputs)["params"]
    tx = optax.sgd(config.learning_rate, config.momentum)
    state = sm.TrainState.create(
        apply_fn=None,
        generic_loss_fn=None,
        specific_loss_fn=None,
        generic_fn=cnn.apply,
        specific_fn=cnn.apply,
        eval_generic_fn=cnn.apply,
        eval_specific_fn=cnn.apply,
        params=params,
        tx=tx,
    )
    return state


def init_tranformer_model_state(
    rng,
    config,
    model_cls=TransformerLM,
):
    """Creates initial `TrainState`."""
    model_ctor = functools.partial(
        model_cls,
        num_hidden_units=config.num_hidden_units,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        shared_embeddings=config.shared_embeddings,
        max_length=config.max_length,
        dtype=config.dtype,
    )
    train_model = model_ctor()
    eval_model = model_ctor(deterministic=True)

    # optimizer: Adam with linear warmup + rsqrt learning rate schedule.
    warmup_steps = config.warmup_steps
    max_lr = config.learning_rate
    learning_rate_fn = optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=0, end_value=max_lr, transition_steps=warmup_steps
            ),
            lambda s: max_lr * (s + warmup_steps) ** -0.5 * warmup_steps**0.5,
        ],
        boundaries=[warmup_steps],
    )
    tx = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=0.9,
        b2=0.98,
        eps=1e-9,
        weight_decay=0.0,
    )

    inputs = dp.input_placeholder(config.batch_size, config.max_length)
    params = eval_model.init(rng, inputs)["params"]

    state = sm.TrainState.create(
        apply_fn=None,
        generic_loss_fn=None,
        specific_loss_fn=None,
        generic_fn=train_model.apply,
        specific_fn=train_model.apply,
        eval_generic_fn=eval_model.apply,
        eval_specific_fn=eval_model.apply,
        params=params,
        tx=tx,
    )
    return state, train_model


def init_model_state(rng, config):
    """Creates initial `TrainState`."""
    if config.model == MODEL_ARCH.CNN:
        return init_cnn_model_state(rng, config)
    elif config.model == MODEL_ARCH.TRANSFORMER:
        state, _ = init_tranformer_model_state(rng, config)
        return state
    else:
        raise ValueError("Unkown model architecture.")


def create_weight_state(rng, config, weight_model, inner_model_params):
    inputs = dp.input_placeholder()
    params = weight_model.init(rng, inputs)["params"]
    if not hasattr(config, "meta_optimizer"):
        config.meta_optimizer = "sgd"
        warnings.warn(
            "Config should now have a meta_optimizer attribute"
            + " set to either sgd or adam. Defaulting to sgd.",
            DeprecationWarning,
        )
    if config.meta_optimizer == "sgd":
        tx = optax.sgd(config.meta_learning_rate, config.meta_momentum)
    elif config.meta_optimizer == "adam":
        tx = optax.adam(config.meta_learning_rate)
    else:
        raise (ValueError("invalid config.meta_optimizer"))
    if config.meta_gradient_method == sm.META_GRADIENT_METHOD.SOBA:
        assert inner_model_params is not None
        # define linear optimizers
        linear_params = jax.tree_map(jnp.zeros_like, inner_model_params)
        # same optimizer as for the lm
        warmup_steps = config.warmup_steps
        max_lr = config.learning_rate
        learning_rate_fn = optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=0,
                    end_value=max_lr,
                    transition_steps=warmup_steps,
                ),
                lambda s: max_lr
                * (s + warmup_steps) ** -0.5
                * warmup_steps**0.5,
            ],
            boundaries=[warmup_steps],
        )
        linear_tx = optax.adamw(
            learning_rate=learning_rate_fn,
            b1=0.9,
            b2=0.98,
            eps=1e-9,
            weight_decay=0.0,
        )  # same as the solver for the generic pb
        return sm.LinearTrainState.create(  # take linear params into account
            apply_fn=weight_model.apply,
            params=params,
            tx=tx,
            linear_params=linear_params,
            linear_tx=linear_tx,
        )
    else:
        return train_state.TrainState.create(
            apply_fn=weight_model.apply,
            params=params,
            tx=tx,
        )


def init_weight_state(rng, config, model_state_params=None):
    if config.meta_gradient_method in [
        sm.META_GRADIENT_METHOD.CDS,
        sm.META_GRADIENT_METHOD.NONE,
    ]:
        assert config.meta_model == WEIGHTING_MODEL_ARCH.NONE
        return None

    meta_fields = config.get(
        "meta_fields",
        {"inputs": dp.FIELDS.INPUTS, "replicate": dp.FIELDS.POS_INPUTS},
    )

    out_type = WEIGHT_TYPE.FLOAT
    replicate_f = None
    if config.meta_gradient_method in [
        sm.META_GRADIENT_METHOD.SOFT,
        sm.META_GRADIENT_METHOD.SOBA,
    ]:
        out_type = WEIGHT_TYPE.PROBABILITY
        replicate_f = meta_fields["replicate"]
    elif config.meta_gradient_method == sm.META_GRADIENT_METHOD.ANOGRAD:
        out_type = WEIGHT_TYPE.POSITIVE

    model_ctor = {
        WEIGHTING_MODEL_ARCH.CNN: CNNWeights,
        WEIGHTING_MODEL_ARCH.BOW: BagOfWordsWeight,
        WEIGHTING_MODEL_ARCH.MLP: sm.MLP,
    }
    assert config.meta_model in model_ctor, "Unsupported meta model type."
    if config.meta_model == WEIGHTING_MODEL_ARCH.MLP:
        model = sm.MLP()
    else:
        model = model_ctor[config.meta_model](
            input_field=config.meta_fields["inputs"],
            replicate_field=replicate_f,
            num_hidden_units=config.num_hidden_unit_weights,
            output_type=out_type,
        )
    return create_weight_state(rng, config, model, model_state_params)

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import ml_collections

true = True
false = False

def get_config():
    return ml_collections.ConfigDict({
        "task": "lm",
        "shuffle_once": false,
        "disable_weight_logging": true,
        "tokenizer": {
            "type": "byte"
        },
        "model": "transformer",
        "num_hidden_units": [
            256,
            256,
            1024
        ],
        "num_heads": 8,
        "num_layers": 12,
        "dropout_rate": 0.1,
        "attention_dropout_rate": 0.1,
        "shared_embeddings": true,
        "max_length": 256,
        "dtype": "bf16",
        "learning_rate": 0.002,
        "warmup_steps": 1000,
        "field_names": [
            "inputs"
        ],
        "generic_tsv": "data/c4/train.txt",
        "generic_split": [
            35000000,
            1000
        ],
        "specific_tsv": "data/pile_100k_freelaw/train.txt",
        "specific_split": [
            0,
            10000,
            0,
            1000
        ],
        "padded_lengths": [
            256
        ],
        "eval_sets": [
            [
                "c4",
                "data/c4/validation.txt",
                false,
                6000
            ],
            [
                "freelaw",
                "data/pile_100k_freelaw/valid.txt",
                false,
                2000
            ]
        ],
        "meta_model": "cnn",
        "meta_optimizer": "adam",
        "meta_gradient_method": "soft",
        "meta_fields": {
            "inputs": "inputs",
            "replicate": "pos_inputs"
        },
        "num_hidden_unit_weights": [
            256,
            128,
            128
        ],
        "meta_learning_rate": 0.001,
        "meta_momentum": 0.9,
        "meta_train_schedule": [
            1,
            1
        ],
        "soft_select_params": {
            "learning_rate": 0.1,
            "parameter_filter": "",
            "replace": false
        },
        "generic_weight": 1.0,
        "generic_batch_size": 16384,
        "batch_size": 128,
        "num_steps": 300000,
        "seed": 0,
        "eval_every_steps": 2000,
        "save_every_steps": 10000,
        "keep_checkpoint_frequency": 10000,
        "name": "c4lm_freelaw soft lr=0.00200 downsampling=128",
        "tags": [
            "soft",
            "c4_pile_lm"
        ],
        "init_checkpoint": ""
    })

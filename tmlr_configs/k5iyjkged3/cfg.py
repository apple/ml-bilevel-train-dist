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
        "tokenizer": {
            "type": "byte"
        },
        "field_names": [
            "inputs"
        ],
        "generic_tsv": "data/c4/train.txt",
        "generic_split": [
            35000000,
            1000
        ],
        "specific_tsv": "data/pile_100k_wikipedia_en/train.txt",
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
                "wikipedia_en",
                "data/pile_100k_wikipedia_en/valid.txt",
                false,
                2000
            ]
        ],
        "meta_model": "none",
        "meta_optimizer": "adam",
        "meta_gradient_method": "none",
        "meta_fields": [
            {
                "inputs": "inputs",
                "replicate": "pos_inputs"
            }
        ],
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
        "meta_classifier_params": {
            "top_frac": 1.0
        },
        "soft_select_params": {
            "learning_rate": 0.1,
            "parameter_filter": ""
        },
        "most_aligned_grad_parameters": {
            "parameter_filter": "",
            "top_frac": 1.0
        },
        "soba_params": {
            "hessian_free": false
        },
        "anograd_params": {
            "parameter_filter": "",
            "sampling": "multi"
        },
        "generic_weight": 0.9,
        "generic_batch_size": 128,
        "batch_size": 128,
        "num_steps": 300000,
        "seed": 0,
        "eval_every_steps": 2000,
        "save_every_steps": 10000,
        "keep_checkpoint_frequency": 10000,
        "name": "c4lm_wikipedia_en downsampling=1.00 lr=0.00200 gen_w=0.900",
        "tags": [
            "c4_pile_lm",
            "mixing"
        ],
        "init_checkpoint": ""
    })

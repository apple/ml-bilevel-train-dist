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
        "specific_tsv": "data/pile_100k_stackexchange/train.txt",
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
                "stackexchange",
                "data/pile_100k_stackexchange/valid.txt",
                false,
                2000
            ]
        ],
        "meta_model": "none",
        "meta_optimizer": "adam",
        "meta_gradient_method": "cds",
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
        "generic_weight": 1.0,
        "generic_batch_size": 8192,
        "batch_size": 128,
        "num_steps": 300000,
        "seed": 0,
        "eval_every_steps": 2000,
        "save_every_steps": 10000,
        "keep_checkpoint_frequency": 10000,
        "cds_parameters": {
            "scoring_config_overloads": {
                "disable_packing": true
            },
            "fine_tune_config_overloads": {
                "learning_rate": 0.0002
            },
            "num_pretrain_steps": 100000,
            "num_fine_tune_steps": 100000,
            "num_scoring_sub_batches": 1
        },
        "name": "c4lm_stackexchange cds num_pre=100000 ds=64 lr=0.00200",
        "tags": [
            "cds",
            "c4_pile_lm"
        ],
        "init_checkpoint": ""
    })

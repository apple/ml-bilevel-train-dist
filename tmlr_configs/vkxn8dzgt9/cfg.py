#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import ml_collections

true = True
false = False


def get_config():
    return ml_collections.ConfigDict(
        {
            "task": "lm",
            "shuffle_once": false,
            "disable_weight_logging": true,
            "tokenizer": {"type": "byte"},
            "model": "transformer",
            "num_hidden_units": [256, 256, 1024],
            "num_heads": 8,
            "num_layers": 12,
            "dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "shared_embeddings": true,
            "max_length": 256,
            "dtype": "bf16",
            "learning_rate": 0.002,
            "warmup_steps": 1000,
            "field_names": ["inputs"],
            "generic_tsv": "data/c4/train.txt",
            "generic_split": [35000000, 1000],
            "specific_tsv": "data/rcv1/train.txt",
            "specific_split": [0, 10000, 0, 1000],
            "padded_lengths": [256],
            "eval_sets": [
                ["c4", "data/c4/validation.txt", false, 6000],
                ["reuters", "data/rcv1/valid.txt", false, 5000],
            ],
            "meta_model": "none",
            "meta_optimizer": "adam",
            "meta_gradient_method": "cds",
            "meta_fields": {"inputs": "inputs", "replicate": "pos_inputs"},
            "num_hidden_unit_weights": [256, 128, 128],
            "meta_learning_rate": 0.001,
            "meta_momentum": 0.9,
            "meta_train_schedule": [1, 1],
            "meta_classifier_params": {"top_frac": 0.03125},
            "soft_select_params": {
                "learning_rate": 0.1,
                "parameter_filter": "",
                "replace": false,
            },
            "most_aligned_grad_parameters": {
                "parameter_filter": "",
                "top_frac": 0.03125,
            },
            "soba_params": {"hessian_free": false},
            "anograd_params": {"parameter_filter": "", "sampling": "multi"},
            "generic_weight": 1.0,
            "generic_batch_size": 4096,
            "batch_size": 128,
            "num_steps": 500000,
            "seed": 0,
            "eval_every_steps": 2000,
            "save_every_steps": 10000,
            "keep_checkpoint_frequency": 10000,
            "cds_parameters": {
                "scoring_config_overloads": {"disable_packing": true},
                "fine_tune_config_overloads": {"learning_rate": 0.0002},
                "num_pretrain_steps": 280000,
                "num_fine_tune_steps": 100000,
                "num_scoring_sub_batches": 1,
            },
            "name": "c4lm cds lr=0.00200 cds pre=280k ft=100k",
            "tags": ["sweep_cds", "c4_reuters_lm"],
            "init_checkpoint": "",
        }
    )

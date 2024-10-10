# Adaptive Training Distributions <br> with Scalable Online Bilevel Optimization


This code supports the LM experiments from the paper:

*Adaptive Training Distributions with Scalable Online Bilevel Optimization.*<br>
David Grangier, Pierre Ablin and Awni Hannun.
Arxiv 2311.11973, 2023.<br>
https://arxiv.org/abs/2311.11973


## Installation

You need jax + flax. On most systems, this is installed via

```
pip install flax
pip install ml_dtypes==0.2.0
```

## Data preparation

### All datasets except RCV1

Run
```
bash dataprep/prepare_data.sh
```

### The Reuters RCV1 dataset

This dataset can be obtained after signing an agreement at
https://trec.nist.gov/data/reuters/reuters.html

Once you have the data, you will uncompress it and provide the path to the
uncompressed data to our script.

```
bash dataprep/rcv1.sh <rcv1_dir>
```

## Training

```
python train.py --config=path/to/cfg.py
```

where cfg.py is a configuration file with hyper-parameters.

We provide configuration files to reproduce the experiments in the paper.
These configurations along with reference tensorboard metrics are in the directory tmlr_configs.
The [list](tmlr_configs/README.md) of experiments with their description is also in this directory.


## Evaluation

The training runs write tensorboard files in `artifacts/<uniq_id>/` where `<uniq_id>` is a random identifier assigned when training starts.
Model checkpoints are periodically saved in the same directory.

## Citation

```
@misc{grangier2023bilevel_train_dist,
      title={Adaptive Training Distributions with Scalable Online Bilevel Optimization},
      author={David Grangier and Pierre Ablin and Awni Hannun},
      year={2023},
      eprint={2311.11973},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2311.11973},
}
```

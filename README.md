# ProbKT

ProbKT, a framework based on probabilistic reasoning to train object detection models with weak supervision, by transferring knowledge from a source domain where rich image annotations are available.

## Prerequisites and installation

To take advantage of full features of code we recommend to create an account on [WANDB](https://wandb.ai/) and login.

ProbKT finetuning also depends on [DeepProbLog](https://github.com/ML-KULeuven/deepproblog).

For easy of use we recommend to also first install and set up a [poetry environment](https://python-poetry.org)

Then execute:

``
poetry install
``


## Get Datasets
All datasets can be downloaded with instructions [Here](datasets/README.md)

For setting up for the MNIST experiments you should execute:

```
cd generate_data
wget --no-check-certificate -O mnist.tar.gz https://figshare.com/ndownloader/files/35142142?private_link=c760de026f000524db5a
tar -xvzf mnist.tar.gz
```

## Train Baseline model

For training the baseline model on the MNIST dataset execute:

```
poetry run python robust_detection/baselines/train.py --data_dir mnist/mnist3_all
```

## Pretrain RCNN Model

Pretrain the RCNN model on source domain of MNIST dataset:

```
poetry run python robust_detection/train/train_rcnn.py --data_path mnist/mnist3_skip
```

## Pretrain DETR Model

Pretrain the DETR model on source domain of MNIST dataset:

```
poetry run python robust_detection/train/train_detr.py --data_path mnist/mnist3_skip --rgb True
```

## ProbKT Finetune RCNN Pretrained model

For finetuning a sweep is assumed on your wandb account for the 5 fold pretrained RCNN model. Example sweep configuration:
```
method: grid
parameters:
  batch_size:
    values:
      - 1
  data_path:
    values:
      - mnist/mnist3_skip
  epochs:
    values:
      - 30
  fold:
    values:
      - 0
      - 1
      - 2
      - 3
      - 4
  pre_trained:
    values:
      - true
program: train_rcnn.py
```

Once sweep has ran succesful finetuning can start using:

```
poetry run python robust_detection/train/train_fine_tune.py --og_data_path mnist/mnist3_skip --target_data_path mnist/mnist3_all --agg_case True --fold 0 --sweep_id <sweepid>
```

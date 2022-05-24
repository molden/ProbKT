# ProbKT

ProbKT, a framework based on probabilistic reasoning to train object detection models with weak supervision, by transferring knowledge from a source domain where rich image annotations are available.

## Prerequisites and installation

For easy of use we recommend to first install and set up a poetry environment: https://python-poetry.org

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

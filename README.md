# ProbKT

ProbKT, a framework based on probabilistic reasoning to train object detection models with weak supervision, by transferring knowledge from a source domain where rich image annotations are available.

If you find this code or idea useful, please consider citing our work:

```

@article{oldenhof2023weakly,
  title={Weakly Supervised Knowledge Transfer with Probabilistic Logical Reasoning for Object Detection},
  author={Oldenhof, Martijn and Arany, Adam and Moreau, Yves and De Brouwer, Edward},
  journal={arXiv preprint arXiv:2303.05148},
  year={2023}
}

```

## Prerequisites and installation

ProbKT finetuning depends on [DeepProbLog](https://github.com/ML-KULeuven/deepproblog).

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

For finetuning a pretrained RCNN model is assumed logged in the folder ``logger/RCNN/version_0``. If the folder is different you can specify it using the command line option ``--experiment_path``. The type of supervision used for finetuning can be set using the ``--target_data_type`` option. For example:

```
poetry run python robust_detection/train/train_fine_tune.py --og_data_path mnist/mnist3_skip --target_data_path mnist/mnist3_all --target_data_type MNIST_Sum --fold 0 --experiment_path logger/RCNN/version_0
```

## Retrain ProbKT Finetuned RCNN  model

Once finetuned the RCNN model can be retrained for several iterations to improve performance. Again the option ``--experiment_path`` points to the previous finetuned model. For example:

```
poetry run python robust_detection/train/retrain_rcnn.py --data_path mnist/mnist3_all --target_data_type MNIST_Sum --fold 0 --experiment_path logger/RCNN-finetune/version_0
```

## ProbKT extras and extension

ProbKT can also be used to finetune a DETR model or use other types of supervision besides ``MNIST_Sum`` like ``Objects_Counter`` of ``Range_Counter``.  New types of supervision can be easily integrated and documentation will be provided. 

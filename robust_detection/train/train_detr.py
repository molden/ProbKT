# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import os
import torch

from robust_detection import utils
from robust_detection.models.detr import DETR

from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
   # parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--lr_step_size', default=7, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--bck_bone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--pos_emb', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--early_stopping', default=7, type=int,
                                    help='patience of the early stopping')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='.',
                        help='path where to save')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
   # parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--num_classes', default=11, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def main(model_cls, data_cls, args, logger = None):

    if args.paired_og:
        if args.data_path == "mnist/alldigits":
            args.og_data_dir = "mnist/skip789"
        elif args.data_path == "mnist/alldigits_2":
            args.og_data_dir = "mnist/skip789_2"
        elif args.data_path == "mnist/alldigits_5":
            args.og_data_dir = "mnist/skip789_5"
        elif args.data_path == "mnist/alldigits_20":
            args.og_data_dir = "mnist/skip789_20"
        elif args.data_path == "molecules/molecules_all":
            args.og_data_dir = "molecules/molecules_skip"
        elif args.data_path == "clevr/clevr_all":
            args.og_data_dir == "clevr/clevr_skip_cube"
        elif args.data_path == "mnist/mnist3_all":
            args.og_data_dir = "mnist/mnist3_skip"
        else:
            raise("Invalid data dir name for paired og")


    if args.output_dir:
        utils.mkdir(args.output_dir)

    # use our dataset and defined transformations
    dataset = data_cls(**vars(args))
    dataset.prepare_data()

    model = model_cls(len_dataloader = len(dataset.val_dataloader()),**vars(args))
#    model = model_cls(**vars(args))

    if logger is None:
            logger = CSVLogger(
            f"{args.output_dir}/logger/",
            name=f"DETR",
        )
    checkpoint_cb = ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor='val_acc',
        mode='max',
        verbose=True
    )
    early_stopping_cb = EarlyStopping(monitor="val_acc", patience=args.early_stopping, mode = "max")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    #model.train()
    trainer = pl.Trainer(gpus = 1, logger = logger, callbacks = [checkpoint_cb, early_stopping_cb, lr_monitor], max_epochs = args.epochs, log_every_n_steps = 10)
    trainer.fit(model, datamodule = dataset)
#    import ipdb; ipdb.set_trace()
    checkpoint_path = checkpoint_cb.best_model_path
    trainer2 = pl.Trainer(logger=False, gpus = 1)

  #  model = model_cls.load_from_checkpoint(
  #      checkpoint_path, len_dataloader = len(dataset.val_dataloader()), **vars(args))
    model = model_cls.load_from_checkpoint(checkpoint_path)
    val_results = trainer2.test(
        model,
        dataloaders=dataset.val_dataloader()
    )[0]

    val_results = {
        name.replace('test', 'val'): value
        for name, value in val_results.items()
    }

    test_results = trainer2.test(
        model,
        dataloaders=dataset.test_dataloader()
    )[0]
    test_results_dict = {}
    val_results_dict = {}
    for name, value in {**test_results}.items():
        test_results_dict[name]=value
        #logger.experiment.summary['restored_' + name] = value
    for name, value in {**val_results}.items():
        val_results_dict[name]=value
        #logger.experiment.summary['restored_' + name] = value
    import pickle
    with open(os.path.join(logger.log_dir,"restored_test_result.pkl"), "wb") as output_file:
         pickle.dump(test_results_dict, output_file)
    with open(os.path.join(logger.log_dir,"restored_val_result.pkl"), "wb") as output_file:
         pickle.dump(val_results_dict, output_file)
    print(f"experiment logged in {logger.log_dir}")
    return logger.log_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    model_cls = DETR
    data_cls = Objects_RCNN
    parser = data_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(model_cls, data_cls, args)

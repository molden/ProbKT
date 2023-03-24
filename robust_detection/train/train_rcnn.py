import os
import torch


#from robust_detection import wandb_config
from robust_detection import utils
from robust_detection.models.rcnn import RCNN
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN, COCO_RCNN

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

def main(model_cls, data_cls, args, logger = None):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    # use our dataset and defined transformations
    dataset = data_cls(**vars(args))
    dataset.prepare_data()
    model = model_cls(len_dataloader = len(dataset.val_dataloader()),**vars(args))
   
    if logger is None:
            logger = CSVLogger(
            f"{args.output_dir}/logger/",
            name=f"RCNN",
        )
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor='val_acc',
        mode='max',
        verbose=True
    )
    early_stopping_cb = EarlyStopping(monitor="val_acc", patience=args.early_stopping, mode = "max")
    lr_monitor = LearningRateMonitor(logging_interval='step')
   
    trainer = pl.Trainer(gpus = args.gpus, logger = logger, callbacks = [checkpoint_cb, early_stopping_cb, lr_monitor], max_epochs = args.epochs, log_every_n_steps = 10)
    trainer.fit(model, datamodule = dataset)

    checkpoint_path = checkpoint_cb.best_model_path
    trainer2 = pl.Trainer(logger=False, gpus = args.gpus)

    model = model_cls.load_from_checkpoint(
        checkpoint_path)
    val_results = trainer2.test(
        model,
    #    test_dataloaders=dataset.val_dataloader()
        dataloaders=dataset.val_dataloader()
    )[0]

    val_results = {
        name.replace('test', 'val'): value
        for name, value in val_results.items()
    }

    test_results = trainer2.test(
        model,
       # test_dataloaders=dataset.test_dataloader()
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

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training')

    #parser.add_argument('--dataset', default='coco', help='dataset')
    #parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--gpus', default=1, help='number of gpus to use', type = int)
    parser.add_argument('--num_classes', default=11, type=int, help='specify number of classes to train')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early_stopping', default=7, type=int, 
                        help='patience of the early stopping')
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    #parser.add_argument('--lr-scheduler', default="multisteplr", help='the lr scheduler (default: multisteplr)')
    parser.add_argument('--lr_step_size', default=5, type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--output-name', default='MNIST_2_large_relabeled_fold_0', help='name of file where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy (default: hflip)')
    parser.add_argument('--data_type', default="Objects", type=str, help='type of data to use (Objects or COCO).')
    
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    partial_args, _ = parser.parse_known_args()

    if partial_args.data_type=="Objects":
        data_cls = Objects_RCNN
    elif partial_args.data_type == "COCO":
        data_cls = COCO_RCNN
    
    model_cls = RCNN
    
    parser = data_cls.add_dataset_specific_args(parser)
    parser = model_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()

    main(model_cls,data_cls, args)


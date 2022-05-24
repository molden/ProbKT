import wandb
from robust_detection.wandb_config import ENTITY
from robust_detection.models.rcnn import RCNN
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN
from robust_detection.train.train_rcnn import main as train_rcnn
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger

if __name__=="__main__":

    parser = ArgumentParser()
    # figure out which model to use and other basic params
    parser.add_argument('--sweep_id', type=str, help='wandb sweep id to use')
    parser.add_argument('--og_data_path', type=str, help='the data the rcnn is pretrained on')
    parser.add_argument('--target_data_path', type=str, help='the model to train on (on top of the original one)')
    parser.add_argument('--output-dir', default='.', help='path where to save')
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
    parser.add_argument('--output-name', default='MNIST_2_large_relabeled_fold_0', help='name of file where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy (default: hflip)')
    parser.add_argument('--data_type', default="Objects", type=str, help='type of data to use (Objects or COCO).')
    parser.add_argument('--fold', type=int, help='the fold we want to fine tune')
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--re_train', type=bool, default=False)
    parser.add_argument('--in_memory', type=bool, default=False)
    parser.add_argument('--hidden_layer', type=int, default=256)
    parser.add_argument('--score_thresh', type=float, default=0.65, help = "score_threshold for the rcnn")
    parser.add_argument('--model_type', type=str, default="rcnn", help = "type of model to use")
    parser.add_argument('--pre_trained', type=bool, default=False)

    args = parser.parse_args()

    sweep_id = args.sweep_id
    target_data_path = args.target_data_path
    og_data_path = args.og_data_path
    
    api = wandb.Api()
    sweep = api.sweep(f"/{ENTITY}/object_detection/"+sweep_id)
    sweep_runs = sweep.runs
    
    best_runs = []
    fold = args.fold
    #import ipdb; ipdb.set_trace()
    runs_fold = [r for r in sweep_runs if ((r.config.get("fold")==fold) and (r.config.get("pre_trained")==True))]
    runs_fold_sorted = sorted(runs_fold,key = lambda run: run.summary.get("restored_val_acc"), reverse = False)
    #import ipdb; ipdb.set_trace()
    best_run = runs_fold_sorted[0]

    model_cls = RCNN
    data_cls = Objects_RCNN
    
    run_name = best_run.id

    logger = WandbLogger(
        name=f"RCNN-ownbackbone",
        project="object_detection",
        log_model=False
    )
    setattr(args, "backbone_run_name", run_name)
    setattr(args, "data_path", args.target_data_path)
    run_id = train_rcnn(model_cls,data_cls,args, logger = logger)

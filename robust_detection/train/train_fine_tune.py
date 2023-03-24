from robust_detection.train import fine_tune
import sys
import os
from robust_detection.models.rcnn import RCNN
from robust_detection.models.detr import DETR
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN_Pred, Objects_RCNN
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from robust_detection.data_utils.problog_data_utils import MNIST_Prod, MNIST_Sum, Objects_Counter, Range_Counter

"""
    print("Filtering data .....")
    if detr:
        # if agg_case:
        #    filter_level = 1
        # else:
        filter_level = 0.92
    else:
        if range_case > -1:
            filter_level = 0.998
        else:
            filter_level = 0.99


    if detr == False:
        initial_score_thresh = model.model.roi_heads.score_thresh
        model.model.roi_heads.score_thresh = 0.05
    if range_case > -1:
        model.model.roi_heads.score_thresh = 0.65
"""

if __name__ == "__main__":

    parser = ArgumentParser()
    # figure out which model to use and other basic params
#    parser.add_argument('--sweep_id', type=str, help='wandb sweep id to use')
    parser.add_argument('--experiment_path', type=str, help='path where pretrained model was logged')
    parser.add_argument('--output_dir', type=str, default='.', help='path where model will be logged')
    parser.add_argument('--og_data_path', type=str,
                        help='the data the rcnn was trained on')
    parser.add_argument('--target_data_path', type=str,
                        help='the model to retrain on (on top of the original one)')
    parser.add_argument('--fold', type=int,
                        help='the fold we want to fine tune')
    parser.add_argument('--detr', type=bool,
                        help='set to true if DETR model', default=False)
   # parser.add_argument('--dpl_path', type=str,
   #                     help='the path to the dpl model for fine tuning')
    parser.add_argument('--filter_level', type=float,
                        help='probability threshold for filtering out objects during fine tuning', default=0.99)
    parser.add_argument('--score_thresh', type=float,
                        help='thresholding score for the prediction model - if None, use default', default=0.05)
    parser.add_argument('--target_data_type', type=str, choices=['MNIST_Prod','MNIST_Sum','Objects_Counter', 'Range_Counter'],
                        help='Name of Data Class to use for fine tuning')
    #parser.add_argument('--agg_case', type=bool, help='set to true to fine  tune in the aggregation mode', default=False)
    parser.add_argument('--range_case', type=int, help='the upper limit of number of objects in order to start using range, set to -1 to ignore ranges', default=-1)
    args = parser.parse_args()

    experiment_path = args.experiment_path
    target_data_path = args.target_data_path
    og_data_path = args.og_data_path

    #api = wandb.Api()
    #sweep = api.sweep(f"/{ENTITY}/object_detection/"+sweep_id)
    #sweep_runs = sweep.runs

    #best_runs = []
    fold = args.fold
    dir_list = os.listdir(experiment_path)
    dir_list = [os.path.join(experiment_path,f) for f in dir_list]
    dir_list.sort(key=os.path.getctime, reverse=True)
    checkpoint_file = [f for f in dir_list if "ckpt" in f][0]
    #checkpoint_file = os.path.join(experiment_path, fname)
    if args.detr:
        model_cls = DETR
    else:
        model_cls = RCNN
    data_cls = Objects_RCNN_Pred

    # Selector for the target data class
    #if args.target_data_type == "MNIST_prod":
    #    target_data_cls = MNIST_Prod
    target_data_cls = getattr(sys.modules[__name__], args.target_data_type)
    #import ipdb; ipdb.set_trace()
    if args.detr:
        logger = CSVLogger(
            f"{args.output_dir}/logger/",
            name=f"DETR-finetune",
        )
    else:
        logger = CSVLogger(
            f"{args.output_dir}/logger/",
            name="RCNN-finetune",
        )

    #fine_tune.fine_tune(run_name, model_cls, data_cls, target_data_path, num_epochs_dpl = 20, logger = logger, detr=args.detr, agg_case = args.agg_case, range_case = args.range_case)
    fine_tune.fine_tune(checkpoint_file, model_cls, data_cls, target_data_cls, target_data_path, args, num_epochs_dpl=20,
                        logger=logger, detr=args.detr, filter_level=args.filter_level)
    if args.detr:
        data_cls = Objects_RCNN
        re_run_id = fine_tune.re_train_detr(
            checkpoint_file, model_cls, data_cls, target_data_path, target_data_cls, logger=logger)
    else:
        data_cls = Objects_RCNN
        re_run_id = fine_tune.re_train(checkpoint_file, model_cls, data_cls, target_data_path, target_data_cls,
                                       logger=logger)

from robust_detection.train import fine_tune
from robust_detection.models import fine_tune_utils
import sys
import os
from robust_detection.models.detr import DETR
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from robust_detection.data_utils.problog_data_utils import MNIST_Prod, MNIST_Sum, Objects_Counter, Range_Counter

if __name__=="__main__":

    parser = ArgumentParser()
    # figure out which model to use and other basic params
   # parser.add_argument('--og_data_path', type=str, help='the data the rcnn was trained on')
    parser.add_argument('--experiment_path', type=str, help='path where pretrained model was logged')
    parser.add_argument('--output_dir', type=str, default='.', help='path where model will be logged')
    parser.add_argument('--data_path', type=str, help='the model to retrain on (on top of the original one)')
    parser.add_argument('--fold', type=int, help='the fold we want to fine tune')
    #parser.add_argument('--agg_case', type=bool, help='set to true to fine  tune in the aggregation mode', default=False)
    parser.add_argument('--range_case', type=int, help='the upper limit of number of objects in order to start using range, set to -1 to ignore ranges', default=-1)
    parser.add_argument('--target_data_type', type=str, choices=['MNIST_Prod','MNIST_Sum','Objects_Counter', 'Range_Counter'],
                                    help='Name of Data Class to use for fine tuning')
    args = parser.parse_args()
    experiment_path = args.experiment_path

    dir_list = os.listdir(experiment_path)
    dir_list = [os.path.join(experiment_path,f) for f in dir_list]
    dir_list.sort(key=os.path.getctime, reverse=True)
    checkpoint_file = [f for f in dir_list if "ckpt" in f][0]
    data_path = args.data_path
    model_cls = DETR
    data_cls = Objects_RCNN
    target_data_cls = getattr(sys.modules[__name__], args.target_data_type)
    logger = CSVLogger(
            f"{args.output_dir}/logger/",
            name=f"DETR-retrain",
        )

    target_data_cls.set_extra_vars(args)
    print("Creating new labels .....")
    #TODO refactor from here
    #fine_tune_utils.relabel_detr(run_name, model_cls, data_cls, data_path = data_path, agg_case=args.agg_case)
    fine_tune_utils.relabel_data(checkpoint_file, model_cls, data_cls, target_data_cls, target_data_path=data_path,
                                             classif=None)
    #re_run_id = fine_tune.re_train_detr(run_name, model_cls, data_cls, data_path, logger = logger, agg_case=args.agg_case)
    re_run_id = fine_tune.re_train_detr(
            checkpoint_file, model_cls, data_cls, data_path, target_data_cls, logger=logger)
    #fine_tune.fine_tune(run_name, model_cls, data_cls, target_data_path, num_epochs_dpl = 20, logger = logger)
    #re_run_id = fine_tune.re_train(run_name, model_cls, data_cls, target_data_path, logger = logger)


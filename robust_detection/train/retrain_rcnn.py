from robust_detection.train import fine_tune
from robust_detection.models import fine_tune_utils
import os
import sys
from robust_detection.models.rcnn import RCNN, RCNN_Predictor
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from robust_detection.data_utils.problog_data_utils import MNIST_Prod, MNIST_Sum, Objects_Counter, Range_Counter

if __name__=="__main__":

    parser = ArgumentParser()
    # figure out which model to use and other basic params
#    parser.add_argument('--sweep_id', type=str, help='wandb sweep id to use')
#    parser.add_argument('--run_name', type=str, help='wandb run_name to use', default=None)
    parser.add_argument('--experiment_path', type=str, help='path where pretrained model was logged')
    parser.add_argument('--output_dir', type=str, default='.', help='path where model will be logged')
   # parser.add_argument('--sweep_id_classifier', default=None, type=str, help='wandb sweep id to use')
   # parser.add_argument('--og_data_path', type=str, help='the data the rcnn was trained on')
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
    model_cls = RCNN
    data_cls = Objects_RCNN
    target_data_cls = getattr(sys.modules[__name__], args.target_data_type)
    logger = CSVLogger(
            f"{args.output_dir}/logger/",
            name=f"RCNN-retrain",
        )
    classifier = None
    #GET (FINETUNED) CLASSIFIER IF NEEDED:
    target_data_cls.set_extra_vars(args)
    print("Creating new labels .....")
    #fine_tune_utils.relabel_data(run_name, model_cls, data_cls, target_data_path = data_path, classif=classifier, agg_case = args.agg_case, range_case=args.range_case)
    fine_tune_utils.relabel_data(checkpoint_file, model_cls, data_cls, target_data_cls, target_data_path=data_path,
                                             classif=classifier)
    print("Done.")
    #fine_tune_utils.relabel_detr(run_name, model_cls, data_cls, data_path = data_path)
    #re_run_id = fine_tune.re_train_detr(run_name, model_cls, data_cls, data_path, logger = logger)
    #fine_tune.fine_tune(run_name, model_cls, data_cls, target_data_path, num_epochs_dpl = 20, logger = logger)
    #re_run_id = fine_tune.re_train(run_name, model_cls, data_cls, data_path, logger = logger, agg_case = args.agg_case, range_case=args.range_case)
    re_run_id = fine_tune.re_train(checkpoint_file, model_cls, data_cls, data_path, target_data_cls,
                                                   logger=logger)


from robust_detection.wandb_config import ENTITY
from robust_detection.train import fine_tune
from robust_detection.models import fine_tune_utils
import wandb
import sys
from robust_detection.models.detr import DETR
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from robust_detection.data_utils.problog_data_utils import MNIST_Prod, MNIST_Sum, Objects_Counter, Range_Counter

if __name__=="__main__":

    parser = ArgumentParser()
    # figure out which model to use and other basic params
    parser.add_argument('--sweep_id', type=str, help='wandb sweep id to use')
    parser.add_argument('--run_name', type=str, help='wandb run_name to use', default=None)
    parser.add_argument('--sweep_id_classifier', default=None, type=str, help='wandb sweep id to use')
   # parser.add_argument('--og_data_path', type=str, help='the data the rcnn was trained on')
    parser.add_argument('--data_path', type=str, help='the model to retrain on (on top of the original one)')
    parser.add_argument('--fold', type=int, help='the fold we want to fine tune')
    #parser.add_argument('--agg_case', type=bool, help='set to true to fine  tune in the aggregation mode', default=False)
    parser.add_argument('--range_case', type=int, help='the upper limit of number of objects in order to start using range, set to -1 to ignore ranges', default=-1)
    parser.add_argument('--target_data_type', type=str, choices=['MNIST_Prod','MNIST_Sum','Objects_Counter', 'Range_Counter'],
                                    help='Name of Data Class to use for fine tuning')
    args = parser.parse_args()

    data_path = args.data_path
    if args.run_name is None:
       sweep_id = args.sweep_id
       #og_data_path = args.og_data_path

       api = wandb.Api()
       sweep = api.sweep(f"/{ENTITY}/object_detection/"+sweep_id)
       sweep_runs = sweep.runs

       best_runs = []
       fold = args.fold

       runs_fold = [r for r in sweep_runs if (r.config.get("fold")==fold)]
       runs_fold_sorted = sorted(runs_fold,key = lambda run: run.summary.get("restored_val_acc"), reverse = False)
       best_run = runs_fold_sorted[0]

       run_name = f"/{ENTITY}/object_detection/{best_run.id}"
    else:
       run_name = args.run_name
    #model_cls = RCNN
    model_cls = DETR
    data_cls = Objects_RCNN
    target_data_cls = getattr(sys.modules[__name__], args.target_data_type)
 
    logger = WandbLogger(
        name=f"DETR-retrain",
        project="object_detection",
        log_model=False
    )

    target_data_cls.set_extra_vars(args)
    print("Creating new labels .....")
    #TODO refactor from here
    #fine_tune_utils.relabel_detr(run_name, model_cls, data_cls, data_path = data_path, agg_case=args.agg_case)
    fine_tune_utils.relabel_data(run_name, model_cls, data_cls, target_data_cls, target_data_path=data_path,
                                             classif=None)
    #re_run_id = fine_tune.re_train_detr(run_name, model_cls, data_cls, data_path, logger = logger, agg_case=args.agg_case)
    re_run_id = fine_tune.re_train_detr(
            run_name, model_cls, data_cls, data_path, target_data_cls, logger=logger)
    #fine_tune.fine_tune(run_name, model_cls, data_cls, target_data_path, num_epochs_dpl = 20, logger = logger)
    #re_run_id = fine_tune.re_train(run_name, model_cls, data_cls, target_data_path, logger = logger)


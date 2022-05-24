from robust_detection import wandb_config
from robust_detection.train import fine_tune
import wandb
from robust_detection.models.rcnn import RCNN
from robust_detection.models.detr import DETR
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger

if __name__=="__main__":

    parser = ArgumentParser()
    # figure out which model to use and other basic params
    parser.add_argument('--sweep_id', type=str, help='wandb sweep id to use')
    parser.add_argument('--og_data_path', type=str, help='the data the rcnn was trained on')
    parser.add_argument('--target_data_path', type=str, help='the model to retrain on (on top of the original one)')
    parser.add_argument('--fold', type=int, help='the fold we want to fine tune')
    parser.add_argument('--detr', type=bool, help='set to true if DETR model', default=False)
    parser.add_argument('--agg_case', type=bool, help='set to true to fine  tune in the aggregation mode', default=False)
    args = parser.parse_args()

    sweep_id = args.sweep_id
    target_data_path = args.target_data_path
    og_data_path = args.og_data_path
    
    api = wandb.Api()
    sweep = api.sweep(f"/{ENTITY}/object_detection/"+sweep_id)
    sweep_runs = sweep.runs
    
    best_runs = []
    fold = args.fold

    runs_fold = [r for r in sweep_runs if ((r.config.get("fold")==fold) and (r.config.get("data_path")==og_data_path))]
    #runs_fold = [r for r in sweep_runs if (r.config.get("fold")==fold)]
    runs_fold_sorted = sorted(runs_fold,key = lambda run: run.summary.get("restored_val_acc"), reverse = True)
    best_run = runs_fold_sorted[0]
    if args.detr:
        model_cls = DETR
    else:
        model_cls = RCNN
    data_cls = Objects_RCNN
    
    run_name = best_run.id
    print(run_name)
    #import ipdb; ipdb.set_trace()
    if args.detr:
        logger = WandbLogger(
            name=f"DETR-finetune",
            project="object_detection",
            log_model=False 
        )
    else:
        logger = WandbLogger(
            name=f"RCNN-finetune",
            project="object_detection",
            log_model=False
        )

    fine_tune.fine_tune(run_name, model_cls, data_cls, target_data_path, num_epochs_dpl = 20, logger = logger, detr=args.detr, agg_case = args.agg_case)
    if args.detr:
        re_run_id = fine_tune.re_train_detr(run_name, model_cls, data_cls, target_data_path, logger = logger, agg_case=args.agg_case)
    else:
       re_run_id = fine_tune.re_train(run_name, model_cls, data_cls, target_data_path, logger = logger, agg_case=args.agg_case)


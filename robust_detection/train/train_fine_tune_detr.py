from robust_detection.wandb_config import ENTITY
from robust_detection.train import fine_tune
from robust_detection.models import fine_tune_utils
import wandb
from robust_detection.models.detr import DETR
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger

if __name__=="__main__":

    parser = ArgumentParser()
    # figure out which model to use and other basic params
    parser.add_argument('--sweep_id', type=str, help='wandb sweep id to use')
   # parser.add_argument('--skip_data_path', type=str, help='the data the detr was trained on')
    parser.add_argument('--data_path', type=str, help='the model to retrain on (on top of the original one)')
    parser.add_argument('--fold', type=int, help='the fold we want to fine tune')
    args = parser.parse_args()

    sweep_id = args.sweep_id
    data_path = args.data_path
    #skip_data_path = args.skip_data_path
    
    api = wandb.Api()
    sweep = api.sweep(f"/{ENTITY}/object_detection/"+sweep_id)
    sweep_runs = sweep.runs
    
    best_runs = []
    fold = args.fold

    #runs_fold = [r for r in sweep_runs if ((r.config.get("fold")==fold) and (r.config.get("skip_data_path")==skip_data_path))]
    runs_fold = [r for r in sweep_runs if (r.config.get("fold")==fold)]
    runs_fold_sorted = sorted(runs_fold,key = lambda run: run.summary.get("restored_val_acc"), reverse = False)
    best_run = runs_fold_sorted[0]

    model_cls = DETR
    data_cls = Objects_RCNN
    
    run_name = best_run.id

    logger = WandbLogger(
        name=f"DETR-finetune",
        project="object_detection",
        log_model=False
    )


    fine_tune_utils.fine_tune_detr(run_name, model_cls, data_cls, data_path, logger=logger)
#    fine_tune_utils.relabel_detr(run_name, model_cls, data_cls, data_path = data_path)
   # re_run_id = fine_tune.re_train_detr(run_name, model_cls, data_cls, data_path, logger = logger)
    #fine_tune.fine_tune(run_name, model_cls, data_cls, target_data_path, num_epochs_dpl = 20, logger = logger)
    #re_run_id = fine_tune.re_train(run_name, model_cls, data_cls, target_data_path, logger = logger)


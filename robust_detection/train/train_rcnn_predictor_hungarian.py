from robust_detection.wandb_config import ENTITY
from robust_detection.train import fine_tune
import wandb
from robust_detection.models.rcnn import RCNN
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN_Predictor
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger

if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # figure out which model to use and other basic params
    parser.add_argument('--sweep_id', type=str, help='wandb sweep id to use')
    parser.add_argument('--og_data_path', type=str, help='the data the rcnn was trained on')
    parser.add_argument('--target_data_path', type=str, help='the model to retrain on (on top of the original one)')
    parser.add_argument('--fold', type=int, help='the fold we want to fine tune')
    parser.add_argument('--epochs', type=int, help='the fold we want to fine tune')
    parser.add_argument('--lr', type=float,default = 0.001, help='learning rate for fine-tuning')
    parser.add_argument('--gradient_clip_val', type=float,default = 0., help='gradient clipping value - 0 is no clipping')
    parser.add_argument('--gpus', default=1, help='number of gpus to use', type = int)
    parser.add_argument('--early_stopping', default=7, type=int, 
                        help='patience of the early stopping')
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
    runs_fold_sorted = sorted(runs_fold,key = lambda run: run.summary.get("restored_val_acc"), reverse = False)
    best_run = runs_fold_sorted[0]

    model_cls = RCNN
    data_cls = Objects_RCNN
    
    run_name = f"{ENTITY}/object_detection/{best_run.id}"

    logger = WandbLogger(
        name=f"RCNN-predictor-finetune-hungarian",
        project="object_detection",
        log_model=False
    )

    fine_tune.hungarian_predictor_fine_tune(run_name, model_cls, data_cls, target_data_path, args = args, logger = logger)
    #re_run_id = fine_tune.re_train(run_name, model_cls, data_cls, target_data_path, logger = logger)




from robust_detection.wandb_config import ENTITY
from robust_detection.train import fine_tune
from robust_detection.models import fine_tune_utils
import wandb
import os
from robust_detection.models.rcnn import RCNN, RCNN_Predictor
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger

if __name__=="__main__":

    parser = ArgumentParser()
    # figure out which model to use and other basic params
    parser.add_argument('--sweep_id', type=str, help='wandb sweep id to use')
    parser.add_argument('--sweep_id_classifier', default=None, type=str, help='wandb sweep id to use')
   # parser.add_argument('--og_data_path', type=str, help='the data the rcnn was trained on')
    parser.add_argument('--data_path', type=str, help='the model to retrain on (on top of the original one)')
    parser.add_argument('--fold', type=int, help='the fold we want to fine tune')
    parser.add_argument('--agg_case', type=bool, help='set to true to fine  tune in the aggregation mode', default=False)
    args = parser.parse_args()

    sweep_id = args.sweep_id
    data_path = args.data_path
    #og_data_path = args.og_data_path
    
    api = wandb.Api()
    sweep = api.sweep(f"/{ENTITY}/object_detection/"+sweep_id)
    sweep_runs = sweep.runs
    
    best_runs = []
    fold = args.fold

    runs_fold = [r for r in sweep_runs if (r.config.get("fold")==fold)]
    runs_fold_sorted = sorted(runs_fold,key = lambda run: run.summary.get("restored_val_acc"), reverse = False)
    best_run = runs_fold_sorted[0]

    model_cls = RCNN
    data_cls = Objects_RCNN
    
    run_name = best_run.id

    logger = WandbLogger(
        name=f"RCNN-retrain",
        project="object_detection",
        log_model=False
    )
    classifier = None
    #GET (FINETUNED) CLASSIFIER IF NEEDED:
    if args.sweep_id_classifier is not None:
        sweep_id_classifier = args.sweep_id_classifier
        sweep = api.sweep(f"/{ENTITY}/object_detection/"+sweep_id_classifier)
        sweep_runs = sweep.runs
        best_runs = []
        fold = args.fold

        runs_fold = [r for r in sweep_runs if (r.config.get("fold")==fold)]
        runs_fold_sorted = sorted(runs_fold,key = lambda run: run.summary.get("restored_val_acc"), reverse = False)
        best_run = runs_fold_sorted[0]

        run_name_classifier = best_run.id
#    api = wandb.Api()
        run = api.run(f"{ENTITY}/object_detection/{run_name_classifier}")

        fname = [f.name for f in run.files() if "ckpt" in f.name][0]
        run.file(fname).download(replace = True, root = ".")
        classifier = RCNN_Predictor.load_from_checkpoint(fname)
        classifier = classifier.classifier
        os.remove(fname)

    print("Creating new labels .....")
    fine_tune_utils.relabel_data(run_name, model_cls, data_cls, target_data_path = data_path, classif=classifier, agg_case = args.agg_case)
    print("Done.")
    #fine_tune_utils.relabel_detr(run_name, model_cls, data_cls, data_path = data_path)
    #re_run_id = fine_tune.re_train_detr(run_name, model_cls, data_cls, data_path, logger = logger)
    #fine_tune.fine_tune(run_name, model_cls, data_cls, target_data_path, num_epochs_dpl = 20, logger = logger)
    re_run_id = fine_tune.re_train(run_name, model_cls, data_cls, data_path, logger = logger, agg_case = args.agg_case)


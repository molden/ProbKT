from robust_detection.train import fine_tune
from robust_detection.models import fine_tune_utils
from robust_detection.models.detr import DETR
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import os

if __name__=="__main__":

    parser = ArgumentParser()
    # figure out which model to use and other basic params
    parser.add_argument('--experiment_path', type=str, help='path where pretrained model was logged')
    parser.add_argument('--output_dir', type=str, default='.', help='path where model will be logged')
   # parser.add_argument('--skip_data_path', type=str, help='the data the detr was trained on')
    parser.add_argument('--data_path', type=str, help='the model to retrain on (on top of the original one)')
    parser.add_argument('--fold', type=int, help='the fold we want to fine tune')
    args = parser.parse_args()

    experiment_path = args.experiment_path

    dir_list = os.listdir(experiment_path)
    dir_list = [os.path.join(experiment_path,f) for f in dir_list]
    dir_list.sort(key=os.path.getctime, reverse=True)
    checkpoint_file = [f for f in dir_list if "ckpt" in f][0]

    data_path = args.data_path
    #skip_data_path = args.skip_data_path
    
    fold = args.fold

    model_cls = DETR
    data_cls = Objects_RCNN
    
    logger = CSVLogger(
            f"{args.output_dir}/logger/",
            name=f"DETR-finetune",
        )

    fine_tune_utils.fine_tune_detr(checkpoint_file, model_cls, data_cls, data_path, logger=logger)
#    fine_tune_utils.relabel_detr(run_name, model_cls, data_cls, data_path = data_path)
   # re_run_id = fine_tune.re_train_detr(run_name, model_cls, data_cls, data_path, logger = logger)
    #fine_tune.fine_tune(run_name, model_cls, data_cls, target_data_path, num_epochs_dpl = 20, logger = logger)
    #re_run_id = fine_tune.re_train(run_name, model_cls, data_cls, target_data_path, logger = logger)


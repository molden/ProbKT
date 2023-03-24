from robust_detection.train import fine_tune
import sys
import os
from robust_detection.models.rcnn import RCNN
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN_Predictor
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from robust_detection.data_utils.problog_data_utils import Objects_Counter

if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # figure out which model to use and other basic params
    parser.add_argument('--experiment_path', type=str, help='path where pretrained model was logged')
    parser.add_argument('--output_dir', type=str, default='.', help='path where model will be logged')
    parser.add_argument('--og_data_path', type=str, help='the data the rcnn was trained on')
    parser.add_argument('--target_data_path', type=str, help='the model to retrain on (on top of the original one)')
    parser.add_argument('--fold', type=int, help='the fold we want to fine tune')
    parser.add_argument('--epochs', type=int, help='the fold we want to fine tune')
    parser.add_argument('--lr', type=float,default = 0.001, help='learning rate for fine-tuning')
    parser.add_argument('--gradient_clip_val', type=float,default = 0., help='gradient clipping value - 0 is no clipping')
    parser.add_argument('--gpus', default=1, help='number of gpus to use', type = int)
    parser.add_argument('--early_stopping', default=7, type=int, 
                        help='patience of the early stopping')
    parser.add_argument('--target_data_type', type=str, choices=['Objects_Counter'],
                                    help='Name of Data Class to use for fine tuning')
    args = parser.parse_args()
    experiment_path = args.experiment_path

    dir_list = os.listdir(experiment_path)
    dir_list = [os.path.join(experiment_path,f) for f in dir_list]
    dir_list.sort(key=os.path.getctime, reverse=True)
    checkpoint_file = [f for f in dir_list if "ckpt" in f][0]

    target_data_path = args.target_data_path
    og_data_path = args.og_data_path
    

    model_cls = RCNN
    data_cls = Objects_RCNN
    
    logger = CSVLogger(
            f"{args.output_dir}/logger/",
            name=f"RCNN-predictor-finetune-hungarian",
        )
    target_data_cls = getattr(sys.modules[__name__], args.target_data_type)
    fine_tune.hungarian_predictor_fine_tune(checkpoint_file, model_cls, data_cls, target_data_cls, target_data_path, args = args, logger = logger)
    #re_run_id = fine_tune.re_train(run_name, model_cls, data_cls, target_data_path, logger = logger)




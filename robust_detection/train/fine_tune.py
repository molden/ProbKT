from robust_detection.wandb_config import ENTITY
from robust_detection.models import fine_tune_utils
import wandb
import os
from robust_detection.models.rcnn import RCNN
from robust_detection.models.rcnn import RCNN_Predictor
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN_Predictor
from deepproblog.train import train_model
from robust_detection.train.train_rcnn import main as re_train_rcnn
from robust_detection.train.train_detr import main as re_train_detr_main
import copy
import torch
from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

def hungarian_predictor_fine_tune(run_name, model_cls, data_cls, target_data_path, args, logger):

    print("Creating Tensor data (and Filtering data) .....")
    fine_tune_utils.create_tensors_data(run_name, model_cls, data_cls, target_data_path = target_data_path, classif = None, filter_level = 1.)
    print("Tensor Data created.")
    data_cls = Objects_RCNN_Predictor
    api = wandb.Api()
    run = api.run(f"{ENTITY}/object_detection/{run_name}")

    fname = [f.name for f in run.files() if "ckpt" in f.name][0]
    run.file(fname).download(replace = True, root = ".")
    model = model_cls.load_from_checkpoint(fname)
    classif = model.box_predictor.cls_score
    #hparams = Namespace(**model.hparams)
    hparams = model.hparams
    
    hparams.data_path = target_data_path
    hparams["filtered"] = True
    #setattr(hparams, 'filtered', True)
    hparams = Namespace(**hparams)
   # import ipdb; ipdb.set_trace()
    dataset = data_cls(**vars(hparams))
    dataset.prepare_data()
    #len_dataloader = len(dataset.val_dataloader())
    wrap_model = RCNN_Predictor(len_dataloader = len(dataset.val_dataloader()), rcnn_head_model=classif, lr = args.lr)

    checkpoint_cb = ModelCheckpoint(
        dirpath=logger.experiment.dir,
        monitor='val_acc',
        mode='max',
        verbose=True
    )
    early_stopping_cb = EarlyStopping(monitor="val_acc", patience=args.early_stopping, mode = "max")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(gpus = args.gpus, logger = logger, callbacks = [checkpoint_cb, early_stopping_cb, lr_monitor], max_epochs = args.epochs, log_every_n_steps = 10, gradient_clip_val = args.gradient_clip_val)
    trainer.fit(wrap_model, datamodule = dataset)

    checkpoint_path = checkpoint_cb.best_model_path
    trainer2 = pl.Trainer(logger=False, gpus = args.gpus)

    model = RCNN_Predictor.load_from_checkpoint(
        checkpoint_path)
    val_results = trainer2.test(
        model,
        test_dataloaders=dataset.val_dataloader()
    )[0]

    val_results = {
        name.replace('test', 'val'): value
        for name, value in val_results.items()
    }

    test_results = trainer2.test(
        model,
        test_dataloaders=dataset.test_dataloader()
    )[0]

    for name, value in {**test_results}.items():
        logger.experiment.summary['restored_' + name] = value
    for name, value in {**val_results}.items():
        logger.experiment.summary['restored_' + name] = value

    return logger.experiment.id

def hungarian_fine_tune(run_name, model_cls, data_cls, target_data_path, args, logger):
    
    api = wandb.Api()
    run = api.run(f"{ENTITY}/object_detection/{run_name}")

    fname = [f.name for f in run.files() if "ckpt" in f.name][0]
    run.file(fname).download(replace = True, root = ".")
    model = model_cls.load_from_checkpoint(fname)
    
    
    for param in model.model.parameters():
        param.requires_grad = False
    
    
    for param in model.model.roi_heads.parameters():
        param.requires_grad = True
    
    model.switch_to_hungarian()
    model.model.roi_heads.score_thresh = 0.05    
    hparams = Namespace(**model.hparams)
    hparams.data_path = target_data_path
    dataset = data_cls(**vars(hparams))
    dataset.prepare_data()
        
    checkpoint_cb = ModelCheckpoint(
        dirpath=logger.experiment.dir,
        monitor='val_acc',
        mode='max',
        verbose=True
    )
    early_stopping_cb = EarlyStopping(monitor="val_acc", patience=args.early_stopping, mode = "max")
    lr_monitor = LearningRateMonitor(logging_interval='step')
   
    trainer = pl.Trainer(gpus = args.gpus, logger = logger, callbacks = [checkpoint_cb, early_stopping_cb, lr_monitor], max_epochs = args.epochs, log_every_n_steps = 10)
    trainer.fit(model, datamodule = dataset)

    checkpoint_path = checkpoint_cb.best_model_path
    trainer2 = pl.Trainer(logger=False, gpus = args.gpus)

    model = model_cls.load_from_checkpoint(
        checkpoint_path)
    val_results = trainer2.test(
        model,
        test_dataloaders=dataset.val_dataloader()
    )[0]

    val_results = {
        name.replace('test', 'val'): value
        for name, value in val_results.items()
    }

    test_results = trainer2.test(
        model,
        test_dataloaders=dataset.test_dataloader()
    )[0]

    for name, value in {**test_results}.items():
        logger.experiment.summary['restored_' + name] = value
    for name, value in {**val_results}.items():
        logger.experiment.summary['restored_' + name] = value

    return logger.experiment.id


        

def fine_tune(run_name, model_cls, data_cls, target_data_path, num_epochs_dpl = 20, logger = None, detr=False, agg_case = False, range_case = -1):

    print("Filtering data .....")
    if detr:
        #if agg_case:
        #    filter_level = 1
        #else:
        filter_level = 0.92
    else:
        if range_case > -1:
            filter_level = 0.998
        else:
            filter_level = 0.99
    fine_tune_utils.create_tensors_data(run_name, model_cls, data_cls, target_data_path = target_data_path, classif = None, filter_level = filter_level, detr=detr, agg_case = agg_case, range_case = range_case)
    print("Data Filtered.")

    print("Loading DPL model ....")
    classifier = None
    dpl_loader, model_dpl, fold = fine_tune_utils.prepare_problog_model(run_name, model_cls, batch_size = 16, target_data_path = target_data_path, classif = classifier, detr=detr, agg_case=agg_case, range_case = range_case)
    print("DPL model loaded.")
        
    classif = copy.deepcopy(model_dpl.networks['mnist_net'].network_module.classifier)
    val_acc = fine_tune_utils.evaluate_classifier(classif, data_path = target_data_path, fold = fold, fold_type = "val", agg_case=agg_case, range_case=range_case)
    print(f"Validation accuracy : {val_acc}")

    print("Training DPL model ...")
    epochs = num_epochs_dpl
    val_acc_best = val_acc
    val_acc_best = 0 #in case init val_acc is already quite good
    for epoch in range(epochs):
        train = train_model(model_dpl, dpl_loader, 1, log_iter=10, profile=0)
        classif = copy.deepcopy(model_dpl.networks['mnist_net'].network_module.classifier)
        val_acc = fine_tune_utils.evaluate_classifier(classif, data_path = target_data_path, fold = fold, fold_type = "val", agg_case=agg_case, range_case=range_case)
        train_acc = fine_tune_utils.evaluate_classifier(classif, data_path = target_data_path, fold = fold, fold_type = "train", agg_case=agg_case, range_case=range_case)
        print(f"Training accuracy : {train_acc:.3f} - Validation accuracy : {val_acc:.3f}")
        if logger is not None:
            logger.experiment.log({"dpl_train_accuracy":train_acc,"dpl_val_accuracy":val_acc})
        if val_acc > val_acc_best:
            #checkpoint model
            print("best model, checkpointing ...")
            best_model_path = os.path.join(logger.experiment.dir, "classifier=epoch="+str(epoch)+"=model.pth")
            torch.save(classif.state_dict(), os.path.join(logger.experiment.dir, "classifier=epoch="+str(epoch)+"=model.pth"))
            val_acc_best = val_acc
    print(f"restoring from {best_model_path} with val_acc {val_acc_best}")
    classif.load_state_dict(torch.load(best_model_path))
    print("Creating new labels .....")
    fine_tune_utils.relabel_data(run_name, model_cls, data_cls, target_data_path = target_data_path, classif = classif, agg_case=agg_case, range_case=range_case)
    print("Done.")
    return 

#def fine_tune_detr(run_name, model_cls, data_cls, target_data_path, logger = None):
#    fine_tune_utils.tune_detr(run_name, model_cls, data_cls, target_data_path

def re_train(run_name, model_cls, data_cls, target_data_path, logger = None, agg_case=False, range_case=-1):

    api = wandb.Api()
    run = api.run(f"{ENTITY}/object_detection/{run_name}")

    fname = [f.name for f in run.files() if "ckpt" in f.name][0]
    run.file(fname).download(replace = True, root = ".")
    model = model_cls.load_from_checkpoint(fname)
    os.remove(fname)

    hparams = model.hparams
    hparams.re_train = True
    hparams.og_data_path = hparams.data_path
    if "molecules" in target_data_path:
              hparams.og_data_path = "molecules/molecules_skip" #TODO
    elif "mnist" in target_data_path: # shifting the label index by 1 in the mnist case (0 is background in RCNN)
              hparams.og_data_path = "mnist/mnist3_skip" #TODO
    elif "clevr" in target_data_path: # shifting the label index by 1 in the mnist case (0 is background in RCNN)
              hparams.og_data_path = "clevr/clevr_skip_cube"
    hparams.data_path = target_data_path
    #hparams.batch_size = 8
    hparams.batch_size = 1
    if hasattr(hparams, 'agg_case'):
        hparams.agg_case=agg_case
    else:
        setattr(hparams, 'agg_case', agg_case)
    if hasattr(hparams, 'range_case'):
        hparams.range_case=range_case
    else:
        setattr(hparams, 'range_case', range_case)
    #hparams.hungarian_fine_tuning = hungarian_fine_tuning
    hparams = Namespace(**hparams)
    del hparams.len_dataloader
    
    print("Re-training RCNN....")
    run_id = re_train_rcnn(model_cls,data_cls,hparams, logger = logger)
    print("Done.")
    return run_id

def re_train_detr(run_name, model_cls, data_cls, target_data_path, logger = None, agg_case=False):

    api = wandb.Api()
    run = api.run(f"{ENTITY}/object_detection/{run_name}")

    fname = [f.name for f in run.files() if "ckpt" in f.name][0]
    run.file(fname).download(replace = True, root = ".")
    model = model_cls.load_from_checkpoint(fname)
    os.remove(fname)

    hparams = model.hparams
    hparams.re_train = True
    hparams.og_data_path = hparams.skip_data_path
    if hparams.skip_data_path is None:
       if "molecules" in target_data_path:
              hparams.og_data_path = "molecules/molecules_skip" #TODO
       elif "mnist" in target_data_path: # shifting the label index by 1 in the mnist case (0 is background in RCNN)
              hparams.og_data_path = "mnist/mnist3_skip" #TODO
       elif "clevr" in target_data_path: # shifting the label index by 1 in the mnist case (0 is background in RCNN)
              hparams.og_data_path = "clevr/clevr_skip_cube"

    hparams.data_path = target_data_path
    #hparams.batch_size = 8
    hparams.batch_size = 8
    hparams.lr           = 0.0001
    hparams.lr_step_size = 20
    hparams.early_stopping = 12
    hparams.set_cost_bbox= 5
    hparams = Namespace(**hparams)
    del hparams.len_dataloader
    if hasattr(hparams, 'box_loss_mask'):
       del hparams.box_loss_mask
    if hasattr(hparams, 'agg_case'):
       hparams.agg_case=agg_case
    else:
       setattr(hparams, 'agg_case', agg_case)
    if hasattr(hparams, 'paired_og'):
        hparams.paired_og=False
    else:
        setattr(hparams, 'paired_og', False)

    print("Re-training DETR....")
    run_id = re_train_detr_main(model_cls,data_cls,hparams, logger = logger)
    print("Done.")
    return run_id


if __name__ == "__main__":
    #--CONFIG--
    #run_name = "11s4i1qm"
    #run_name = "2t8t51ef"
    run_name = "2j6hebk5"
    model_cls = RCNN

    data_cls = Objects_RCNN
    #target_data_path = "mnist/alldigits_2/"
    target_data_path = "mnist/alldigits/"
    #----------------

    hungarian_fine_tune(run_name, model_cls, data_cls, target_data_path, num_epochs_hungarian = 20)
    
    re_run_id = re_train(run_name, model_cls, data_cls, target_data_path)
    print(f"Retrained RCNN available at {re_run_id}.")
    

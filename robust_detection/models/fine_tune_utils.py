from robust_detection.wandb_config import ENTITY
import wandb
import os
from robust_detection.models.rcnn import RCNN
from robust_detection.models.matcher import build_matcher
from robust_detection.models.detr import SetCriterion
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from robust_detection.utils import DATA_DIR
import torch
import shutil
import pandas as pd
from robust_detection.data_utils.problog_data_utils import process_labels
from robust_detection.models.rcnn_utils import WrapModel
import numpy as np
import copy

from robust_detection.data_utils.problog_data_utils import MNIST_Sum, Objects_Counter, MNIST_Images, Range_Counter

from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.network import MNIST_Net

from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.utils import get_configuration, format_time_precise, config_to_string
import argparse
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

from robust_detection.engine import train_one_epoch, evaluate
from robust_detection.dpl_utils import MNIST_Classifier, MNIST_Detection_Dataset, MNIST_detect_wrapper, MNIST_SingleDetection_Dataset
from robust_detection.dpl_utils import myRoIHeads
from deepproblog.dataset import DataLoader

from robust_detection.utils import DATA_DIR


def prepare_problog_model(run_name, model_cls, target_data_cls, batch_size=16, target_data_path=None, classif=None, detr=False):
    """
    - Feed the problog model directly
    - Feed the datasets
    """

    api = wandb.Api()

    #run = api.run(f"{ENTITY}/object_detection/{run_name}")
    run = api.run(run_name)

    fname = [f.name for f in run.files() if "ckpt" in f.name][0]
    run.file(fname).download(replace=True, root=".")
    model = model_cls.load_from_checkpoint(fname)
    os.remove(fname)

    hparams = model.hparams

    if target_data_path is not None:
        hparams.data_path = target_data_path

    fold = hparams.fold


    datasets = {
            "train": target_data_cls(f"{hparams.data_path}", fold, fold_type="train"),
            "val": target_data_cls(f"{hparams.data_path}", fold, fold_type="val"),
            "test": target_data_cls(f"{hparams.data_path}", fold, fold_type="test"),
    }

    if classif is None:
        if detr:
            classif = model.class_embed
        else:
            classif = model.box_predictor.cls_score

    wrap_model = WrapModel(classif)

    net = Network(wrap_model, "mnist_net", batching=True)
    net.cuda()
    net.optimizer = torch.optim.Adam(wrap_model.parameters(), lr=1e-2)

    dpl_script = target_data_cls.get_dpl_script(f"{hparams.data_path}")
    model_dpl = Model(dpl_script, [net])
    model_dpl.set_engine(ExactEngine(model_dpl), cache=True)

    # Change the name of the MNIST_Images class to a more general name
    Images_train = MNIST_Images(datasets["train"])
    Images_val = MNIST_Images(datasets["val"])
    Images_test = MNIST_Images(datasets["test"])
    model_dpl.add_tensor_source("train", Images_train)
    model_dpl.add_tensor_source("val", Images_val)
    model_dpl.add_tensor_source("test", Images_test)

    loader = DataLoader(
        dataset=datasets["train"], batch_size=batch_size, shuffle=True)

    return loader, model_dpl, fold


def evaluate_classifier(classif, target_data_cls, data_path, fold, fold_type, threshold=0.65):

    classif.to("cpu")
    wrap_model = WrapModel(classif)

    if fold_type == "val":
        fold_overal_type = "train"
    else:
        fold_overal_type = fold_type

    train_idx = np.load(os.path.join(
        DATA_DIR, f"{data_path}", "folds", str(fold), "train_idx.npy"))
    val_idx = np.load(os.path.join(
        DATA_DIR, f"{data_path}", "folds", str(fold), "val_idx.npy"))

    accs = []
    for f in os.listdir(os.path.join(DATA_DIR, data_path, fold_overal_type, "tensors")):

        if fold_type == "train":
            if int(f[:-3]) not in train_idx:
                continue
        if fold_type == "val":
            if int(f[:-3]) not in val_idx:
                continue

        box_features = torch.load(os.path.join(
            DATA_DIR, data_path, fold_overal_type, "tensors", f))
        if fold_type == "train":
            labels = process_labels(os.path.join(
                DATA_DIR, data_path, fold_overal_type, "filtered_labels", f[:-2]+"txt"), shift=0)
        else:
            labels = process_labels(os.path.join(
                DATA_DIR, data_path, fold_overal_type, "labels", f[:-2]+"txt"), shift=0)
            #box_features = box_features[:len(labels)]
        #preds = torch.arange(10)[None,:].repeat(box_features.shape[0],1)[ wrap_model(box_features) > threshold ]
        preds = torch.argmax(wrap_model(box_features), 1)
        accs.append(target_data_cls.evaluate_classifier(preds,labels))
    accs = np.array(accs)
    return np.mean(accs)
    

def test_tensors_data(run_name, model_cls, data_cls):
    """
    Test routine to check the accuracies of the classifier from RCNN on the extracted tensors
    """
    api = wandb.Api()
    run = api.run(f"{ENTITY}/object_detection/{run_name}")

    fname = [f.name for f in run.files() if "ckpt" in f.name][0]
    run.file(fname).download(replace=True, root=".")
    model = model_cls.load_from_checkpoint(fname)
    os.remove(fname)

    hparams = model.hparams
    dataset = data_cls(**hparams)
    dataset.prepare_data()

    classif = model.box_predictor.cls_score
    wrap_model = WrapModel(classif)

    accs = []
    for f in os.listdir(os.path.join(DATA_DIR, hparams.data_path, "train", "tensors")):
        box_features = torch.load(os.path.join(
            DATA_DIR, hparams.data_path, "train", "tensors", f))
        labels = process_labels(os.path.join(
            DATA_DIR, hparams.data_path, "train", "labels", f[:-2]+"txt"), shift=0)
        preds = torch.arange(10)[None, :].repeat(box_features.shape[0], 1)[
            wrap_model(box_features) > 0.65]
        accs.append(torch.equal(
            preds.sort()[0].long(), torch.Tensor(labels).sort()[0].long()))
    accs = np.array(accs)
    print(f"Accuracy on recovered tensors with classifier : {accs.mean()}")


def filter_data(box_features, labels, boxes, classif, level=0.99, in_label_check=True):
    """
    in_label_check : if true, only removes the tensor if the predicted digit is in the label.
    """
    assert(level > 0.5)
    og_labels = copy.deepcopy(labels)
    wrap_model = WrapModel(classif)
    # get the index and the values of the confident predictions
    confident_index, confident_preds = torch.where(
        wrap_model(box_features) > level)
    # initialized the retained index with the full tensor
    retained_index = [i for i in range(box_features.shape[0])]
    removed_labels = []
    for i in range(len(confident_index)):
        np_labels = np.array(labels)
        np_removed_labels = np.array(removed_labels)
        if len(labels) <= 2:  # at least two prediction left per image
            break
        else:
            if in_label_check:
                if confident_preds[i].item() in labels:
                    # remove this tensor from the data
                    retained_index.remove(confident_index[i])
                    # remove the label
                    labels.remove(confident_preds[i].item())
            else:
                if range_case > - 1:
                   # import ipdb; ipdb.set_trace()
                    count_labels = np.count_nonzero(
                        np_labels == confident_preds[i].item())
                    count_removed_labels = np.count_nonzero(
                        np_removed_labels == confident_preds[i].item())
                    if (count_labels + count_removed_labels) > range_case:
                        if count_removed_labels < range_case:
                            # also in range_case we know which labels are present only not the exact amount
                            if confident_preds[i].item() in labels:
                                retained_index.remove(confident_index[i])
                                labels.remove(confident_preds[i].item())
                                removed_labels.append(
                                    confident_preds[i].item())
                    else:
                        # also in range_case we know which labels are present only not the exact amount
                        if confident_preds[i].item() in labels:
                            retained_index.remove(confident_index[i])
                            labels.remove(confident_preds[i].item())
                            removed_labels.append(confident_preds[i].item())
                    # count_ind_labels + count_removed_labels: based on this and range_case decide to remove labels
                   # import ipdb; ipdb.set_trace()
                    #labels, confident_preds[i]
                else:
                    # remove this tensor from the data
                    retained_index.remove(confident_index[i])
                    # remove the label
                    labels.remove(confident_preds[i].item())
    # This check can only be done when we have the exact number of labels (not in range case)
    if range_case == -1:
        if len(retained_index) > len(labels):
            # TODO : this is  to deal with mistmatch between number of windows and numbers of labels
            retained_index = retained_index[:len(labels)]

    if len(retained_index) < len(labels):
        # TODO : this is  to deal with mistmatch between number of windows and numbers of labels
        box_features = torch.cat((box_features, torch.zeros(
            len(labels)-len(retained_index), box_features.shape[1])), 0)
        retained_index += [-i for i in range(1,
                                             len(labels)-len(retained_index)+1)]
        #import ipdb; ipdb.set_trace()
        if range_case > -1:
            return None, None, None
        else:
            return None, None

    if range_case == -1:
        assert(len(retained_index) == len(labels))

    if len(retained_index) > 4 and level < 1.:
        if range_case > -1:
            return None, None, None
        else:
            return None, None

    df = pd.DataFrame(labels, columns=["label"])
   # if len(retained_index) > boxes.shape[0]:
   #    import ipdb; ipdb.set_trace()
   # df["xmin"] = boxes[retained_index,0].long()
   # df["ymin"] = boxes[retained_index,1].long()
   # df["xmax"] = boxes[retained_index,2].long()
   # df["ymax"] = boxes[retained_index,3].long()
    df["xmin"] = 0
    df["ymin"] = 0
    df["xmax"] = 0
    df["ymax"] = 0

   # if (df["xmax"]-df["xmin"]).values.min()<=0:
   # if range_case > -1:
   # return None, None, None
   # else:
   # return None, None
   # if (df["ymax"]-df["ymin"]).values.min()<=0:
   # if range_case > -1:
   # return None, None, None
   # else:
   # return None, None
   # else:
    if range_case > -1:
        df_del = pd.DataFrame(removed_labels, columns=["label"])
        df_del["xmin"] = 0
        df_del["ymin"] = 0
        df_del["xmax"] = 0
        df_del["ymax"] = 0
        return box_features[retained_index], df, df_del
    else:
        return box_features[retained_index], df


def select_data_to_label(box_features, labels, boxes, classif, agg_case=False, range_case=-1):
    wrap_model = WrapModel(classif)
    if range_case == -1: 
       box_features = box_features[:len(labels)]
       boxes = boxes[:len(labels)]
    preds = torch.argmax(wrap_model(box_features),1)
    #import ipdb; ipdb.set_trace()
    if range_case > -1:
        values_preds, count_preds = torch.unique(
            preds, sorted=True, return_counts=True)
        values_labels, count_labels = torch.unique(torch.tensor(
            labels, dtype=torch.int64), sorted=True, return_counts=True)
        #import ipdb; ipdb.set_trace()
        if torch.equal(values_preds, values_labels):
            count_preds_clamped = torch.clamp(count_preds, max=range_case)
            count_labels_clamped = torch.clamp(count_labels, max=range_case)
        # import ipdb; ipdb.set_trace()
            if torch.equal(count_preds_clamped, count_labels_clamped):
                df = pd.DataFrame(preds, columns=["label"])

                df["xmin"] = boxes[:, 0].long()
                df["ymin"] = boxes[:, 1].long()
                df["xmax"] = boxes[:, 2].long()
                df["ymax"] = boxes[:, 3].long()
                return df
            else:
                return None
    if agg_case:
        #import ipdb; ipdb.set_trace()
        if torch.sum(preds.sort()[0].long()) == torch.sum(torch.Tensor(labels).long().sort()[0]):
            df = pd.DataFrame(preds, columns=["label"])

            df["xmin"] = boxes[:, 0].long()
            df["ymin"] = boxes[:, 1].long()
            df["xmax"] = boxes[:, 2].long()
            df["ymax"] = boxes[:, 3].long()
            return df
        else:
            return None

    if torch.equal(preds.sort()[0].long(), torch.Tensor(labels).long().sort()[0]):
        df = pd.DataFrame(preds, columns=["label"])

        df["xmin"] = boxes[:, 0].long()
        df["ymin"] = boxes[:, 1].long()
        df["xmax"] = boxes[:, 2].long()
        df["ymax"] = boxes[:, 3].long()

        return df

    else:
        return None


def create_tensors_data(run_name, model_cls, data_cls, target_data_cls=None, target_data_path=None, classif=None, filter_level=0.99, detr=False, class_filtering=True, score_thresh=0.05):
    """
    If classif is None, it uses the classifier from the pre-trained RCNN 
    """
    api = wandb.Api()
    #run = api.run(f"{ENTITY}/object_detection/{run_name}")
    run = api.run(run_name)

    fname = [f.name for f in run.files() if "ckpt" in f.name][0]
    run.file(fname).download(replace=True, root=".")
    model = model_cls.load_from_checkpoint(fname)
    os.remove(fname)

    hparams = model.hparams
    hparams.re_train = False
    if target_data_path is not None:
        hparams.data_path = target_data_path
    
    dataset = data_cls(**hparams)
    dataset.prepare_data()

    if classif is None:
        if detr:
            classif = model.class_embed
        else:
            classif = model.box_predictor.cls_score

    trainer = pl.Trainer(logger=False, gpus=1)

    # disabling the score thresholding for the training set
    if detr == False:
        initial_score_thresh = model.model.roi_heads.score_thresh
        if score_thresh is not None:
           model.model.roi_heads.score_thresh = score_thresh
    train_preds = trainer.predict(model, dataset.train_dataloader())

    if detr == False:
        model.model.roi_heads.score_thresh = initial_score_thresh

    val_preds = trainer.predict(model, dataset.val_dataloader())
    test_preds = trainer.predict(model, dataset.test_dataloader())

    shutil.rmtree(os.path.join(DATA_DIR, hparams.data_path,
                  "train", "tensors"), ignore_errors=True)
    shutil.rmtree(os.path.join(DATA_DIR, hparams.data_path,
                  "test", "tensors"), ignore_errors=True)
    shutil.rmtree(os.path.join(DATA_DIR, hparams.data_path,
                  "train", "filtered_labels"), ignore_errors=True)
    shutil.rmtree(os.path.join(DATA_DIR, hparams.data_path,
                  "test", "filtered_labels"), ignore_errors=True)
    shutil.rmtree(os.path.join(DATA_DIR,hparams.data_path,"train","removed_labels"),ignore_errors = True)
    shutil.rmtree(os.path.join(DATA_DIR,hparams.data_path,"test","removed_labels"), ignore_errors = True)


    os.makedirs(os.path.join(DATA_DIR, hparams.data_path,
                "train", "filtered_labels"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, hparams.data_path,
                "test", "filtered_labels"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR,hparams.data_path,"train","removed_labels"),exist_ok = True)
    os.makedirs(os.path.join(DATA_DIR,hparams.data_path,"test","removed_labels"),exist_ok = True)
    os.makedirs(os.path.join(DATA_DIR, hparams.data_path,
                "train", "tensors"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, hparams.data_path,
                "test", "tensors"), exist_ok=True)

    num_boxes_og = []
    num_boxes_filtered = []
    idx_subset = []
    for batch in train_preds:
        for img_idx in range(len(batch["idx"])):
            idx = batch["idx"][img_idx]
            box_features = batch["box_features"][img_idx]
            boxes = batch["boxes"][img_idx]
            labels = pd.read_csv(os.path.join(
                DATA_DIR, hparams.data_path, "train", "labels", f"{idx}.txt"))
            

            num_boxes_og.append(box_features.shape[0])
            if target_data_cls is not None:
               box_features, new_labels, removed_labels = target_data_cls.filter_data(
                   box_features, labels, boxes, classif, level=filter_level)
            else:
               new_labels=labels
               removed_labels=None

            if new_labels is not None:
                torch.save(box_features, os.path.join(
                    DATA_DIR, hparams.data_path, "train", "tensors", f"{idx}.pt"))
                new_labels.to_csv(os.path.join(
                    DATA_DIR, hparams.data_path, "train", "filtered_labels", f"{idx}.txt"))
                if removed_labels is not None:
                   removed_labels.to_csv(os.path.join(DATA_DIR,hparams.data_path,"train","removed_labels",f"{idx}.txt"))

                idx_subset.append(idx)
                 
                num_boxes_filtered.append(box_features.shape[0])

    np.save(os.path.join(DATA_DIR, hparams.data_path, "folds", str(
        hparams.fold), "filtered_train_idx.npy"), np.array(idx_subset))

    print(" Num boxes originally in the data : ")
    print(np.array(num_boxes_og).sum())
    print(" Num boxes after filtering : ")
    print(np.array(num_boxes_filtered).sum())
    print("Maximum number of boxes per image")
    print(np.array(num_boxes_filtered).max())
    print("Average number of boxes per image")
    print(np.array(num_boxes_filtered).mean())
    print("Fraction of images kept : ")
    print(len(num_boxes_filtered)/len(num_boxes_og))

    for batch in val_preds:
        for img_idx in range(len(batch["idx"])):
            idx = batch["idx"][img_idx]
            box_features = batch["box_features"][img_idx]
            torch.save(box_features, os.path.join(
                DATA_DIR, hparams.data_path, "train", "tensors", f"{idx}.pt"))

    for batch in test_preds:
        for img_idx in range(len(batch["idx"])):
            idx = batch["idx"][img_idx]
            box_features = batch["box_features"][img_idx]
            torch.save(box_features, os.path.join(
                DATA_DIR, hparams.data_path, "test", "tensors", f"{idx}.pt"))


def fine_tune_detr(run_name, model_cls, data_cls, target_data_path=None, logger=None):
    api = wandb.Api()
    run = api.run(f"{ENTITY}/object_detection/{run_name}")

    fname = [f.name for f in run.files() if "ckpt" in f.name][0]
    run.file(fname).download(replace=True, root=".")
    model = model_cls.load_from_checkpoint(fname)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.class_embed.parameters():
        param.requires_grad = True
    os.remove(fname)
    hparams = model.hparams
    model.matcher = build_matcher(hparams["set_cost_class"], 0, 0)
    weight_dict = {'loss_ce': 1, 'loss_bbox': hparams["bbox_loss_coef"]}
    weight_dict['loss_giou'] = hparams["giou_loss_coef"]
    losses = ['labels', 'cardinality']
    model.criterion = SetCriterion(hparams["num_classes"], matcher=model.matcher, weight_dict=weight_dict,
                                   eos_coef=hparams["eos_coef"], losses=losses)
    if target_data_path is not None:
        hparams.data_path = target_data_path
        hparams["skip_data_path"] = None
        hparams["box_loss_mask"] = True
        hparams["set_cost_bbox"] = 0
        hparams["lr"] = 0.002
        hparams["lr_step_size"] = 20
    dataset = data_cls(**hparams)
    dataset.prepare_data()

    if logger is None:
        logger = WandbLogger(
            name=f"DETR-finetune",
            project="object_detection",
            log_model=False
        )
    checkpoint_cb = ModelCheckpoint(
        dirpath=logger.experiment.dir,
        monitor='val_acc',
        mode='max',
        verbose=True
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_acc", patience=12, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # model.train()
    trainer = pl.Trainer(gpus=1, logger=logger, callbacks=[
                         checkpoint_cb, early_stopping_cb, lr_monitor], max_epochs=100, log_every_n_steps=10)
    trainer.fit(model, datamodule=dataset)
#    import ipdb; ipdb.set_trace()
    checkpoint_path = checkpoint_cb.best_model_path
    trainer2 = pl.Trainer(logger=False, gpus=1)

  #  model = model_cls.load_from_checkpoint(
  #      checkpoint_path, len_dataloader = len(dataset.val_dataloader()), **vars(args))
    model = model_cls.load_from_checkpoint(checkpoint_path)
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


def relabel_data(run_name, model_cls, data_cls, target_data_cls, target_data_path=None, classif=None):
    """
    If classif is None, it uses the classifier from the pre-trained RCNN 
    """
    api = wandb.Api()
    #run = api.run(f"{ENTITY}/object_detection/{run_name}")
    run = api.run(f"{run_name}")
    fname = [f.name for f in run.files() if "ckpt" in f.name][0]
    run.file(fname).download(replace=True, root=".")
    model = model_cls.load_from_checkpoint(fname)
    os.remove(fname)

    hparams = model.hparams
    hparams.re_train = False
    if target_data_path is not None:
        hparams.data_path = target_data_path
    dataset = data_cls(**hparams)
    dataset.prepare_data()

    if classif is None:
        if model_cls == RCNN:
           classif = model.box_predictor.cls_score
        else: 
           classif = model.class_embed

    trainer = pl.Trainer(logger=False, gpus=1)

    # disabling the score thresholding for the training set
    if model_cls == RCNN:
        initial_score_thresh = model.model.roi_heads.score_thresh
        model.model.roi_heads.score_thresh = 0.05
    #model.model.roi_heads.box_predictor.cls_score = classif.to(model.model.roi_heads.box_predictor.cls_score[0].weight.device)
    train_preds = trainer.predict(model, dataset.train_dataloader())

    if model_cls == RCNN:
        model.model.roi_heads.score_thresh = initial_score_thresh

    shutil.rmtree(os.path.join(DATA_DIR, hparams.data_path,
                  "train", "re_labels"), ignore_errors=True)
    os.makedirs(os.path.join(DATA_DIR, hparams.data_path,
                "train", "re_labels"), exist_ok=True)

    idx_subset = []
    for batch in train_preds:
        for img_idx in range(len(batch["idx"])):
            idx = batch["idx"][img_idx]
            box_features = batch["box_features"][img_idx]
            boxes = batch["boxes"][img_idx]
            labels = process_labels(os.path.join(
                DATA_DIR, hparams.data_path, "train", "labels", f"{idx}.txt"))
            new_labels = target_data_cls.select_data_to_label(box_features, labels, boxes, classif)
            if new_labels is not None:
                new_labels.to_csv(os.path.join(
                    DATA_DIR, hparams.data_path, "train", "re_labels", f"{idx}.txt"), index=False)
                idx_subset.append(idx)
    np.save(os.path.join(DATA_DIR, hparams.data_path, "folds", str(
        hparams.fold), "re_train_idx.npy"), np.array(idx_subset))


def relabel_detr(run_name, model_cls, data_cls, data_path=None, agg_case=False):
    api = wandb.Api()
#    run = api.run(f"{ENTITY}/object_detection/{run_name}")
    run = api.run(f"{run_name}")
    fname = [f.name for f in run.files() if "ckpt" in f.name][0]
    run.file(fname).download(replace=True, root=".")
    model = model_cls.load_from_checkpoint(fname)
    os.remove(fname)

    hparams = model.hparams
    if data_path is not None:
        hparams.data_path = data_path
        hparams["skip_data_path"] = None
        hparams.re_train = False
    dataset = data_cls(**hparams)
    dataset.prepare_data()

    trainer = pl.Trainer(logger=False, gpus=1)
    train_preds = trainer.predict(model, dataset.train_dataloader())

    shutil.rmtree(os.path.join(DATA_DIR, hparams.data_path,
                  "train", "re_labels"), ignore_errors=True)
    os.makedirs(os.path.join(DATA_DIR, hparams.data_path,
                "train", "re_labels"), exist_ok=True)

    idx_subset = []
    Y = []
    Y_hat = []
    scores = []
    boxes = []
    idxs = []
    idx_subset = []
    for pred in train_preds:
        Y += pred["targets"]
        Y_hat += pred["preds"]
        scores += pred["scores"]
        boxes += pred["boxes"]
        idxs += pred["idx"]
        # for img_idx in range(len(pred["idx"])):
        #    idxs    += [pred["idx"][img_idx]]
        # import ipdb; ipdb.set_trace()
    labels = []
    new_boxes = []
    for i in range(len(Y_hat)):
        keep = [scores[i] > 0.5]
        labels.append(Y_hat[i][keep])
        new_boxes.append(boxes[i][keep])
    if agg_case:
        #import ipdb; ipdb.set_trace()
        keep_correct = [torch.sum(Y[i].sort()[0]) == torch.sum(
            labels[i][torch.where(labels[i] != 0)].sort()[0]) for i in range(len(Y))]
    else:
        keep_correct = [torch.equal(Y[i].sort()[0], labels[i][torch.where(
            labels[i] != 0)].sort()[0]) for i in range(len(Y))]
    count = 0
   # import ipdb; ipdb.set_trace()
    for idx, correct in zip(idxs, keep_correct):

        if correct:
            # imagename=f"{os.path.join(DATA_DIR,dataset.data_path,'train')}/images/{str(idx)}.png"
            boxs = new_boxes[count]
            b_labels = labels[count]-1
            df = pd.DataFrame(b_labels, columns=["label"])

            df["xmin"] = boxs[:, 0].long()
            df["ymin"] = boxs[:, 1].long()
            df["xmax"] = boxs[:, 2].long()
            df["ymax"] = boxs[:, 3].long()
            df.to_csv(os.path.join(DATA_DIR, hparams.data_path,
                      "train", "re_labels", f"{idx}.txt"), index=False)
            idx_subset.append(idx)
        count += 1
    np.save(os.path.join(DATA_DIR, hparams.data_path, "folds", str(
        hparams.fold), "re_train_idx.npy"), np.array(idx_subset))


if __name__ == "__main__":
    run_name = "388vbo0h"
    model_cls = RCNN
    data_cls = MNIST_RCNN

    create_tensors_data(run_name, model_cls, data_cls)

    # ------- TEST CHECK ------
    test_tensors_data(run_name, model_cls, data_cls)
    # -----------------------------

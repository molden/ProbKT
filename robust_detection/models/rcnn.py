import math
import os
import sys
import time
import torch
from robust_detection.wandb_config import ENTITY
import wandb

from robust_detection.baselines.cnn_model import CNN
import torchvision.models.detection.mask_rcnn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torch import nn
from typing import Callable, Dict, List, Optional, Union
import pytorch_lightning as pl
import torchvision
from robust_detection import utils

from robust_detection.models.rcnn_utils import myRoIHeads
from robust_detection.utils import str2bool
from robust_detection.models.matcher import HungarianMatcher

from torchvision.models import resnet18, resnet50
from torchvision.models.detection.backbone_utils import _validate_trainable_layers, resnet_fpn_backbone, BackboneWithFPN
#from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN
#from torchmetrics.detection.map import MeanAveragePrecision
import torch.nn.functional as F

class BoxPredictor(torch.nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = torch.nn.Sequential(torch.nn.Linear(in_channels, in_channels), torch.nn.ReLU(), torch.nn.Linear(in_channels,num_classes))
        #self.cls_score = torch.nn.Linear(in_channels,num_classes)
        
        self.bbox_pred = torch.nn.Sequential(torch.nn.Linear(in_channels,in_channels),torch.nn.ReLU(), torch.nn.Linear(in_channels, num_classes * 4))

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def get_model(num_classes, score_thresh, model_type = "mask_rcnn", pretrained = True, bckbone_rname = None):
    if model_type == "mask_rcnn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone = pretrained)
         # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one

        box_predictor = BoxPredictor(in_features, num_classes)
        model.roi_heads.box_predictor = box_predictor

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = hidden_layer
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
        model.roi_heads.score_thresh = score_thresh
        model.roi_heads.__class__ = myRoIHeads

        return model, box_predictor
    elif model_type == "rcnn":
        if pretrained:
            model = fasterrcnn_resnet50_fpn(pretrained = pretrained, pretrained_backbone = pretrained)
        elif bckbone_rname is not None:
            model = fasterrcnn_ownbackbone_fpn(bckbone_rname)
        else:
            model = fasterrcnn_resnet18_fpn()
            
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        

        box_predictor = BoxPredictor(in_features, num_classes)
        model.roi_heads.box_predictor = box_predictor
        model.roi_heads.score_thresh = score_thresh
        model.roi_heads.__class__ = myRoIHeads

        return model, box_predictor

def fasterrcnn_resnet18_fpn(num_classes=91, **kwargs):
    backbone = resnet_fpn_backbone("resnet18",pretrained = False, trainable_layers = 5)
    #backbone = _resnet_fpn_extractor(resnet18(pretrained = False, progress = True),5)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model

def fasterrcnn_ownbackbone_fpn(bckbone_rname, num_classes=91, **kwargs):
    model_cls = CNN
    api = wandb.Api()
    run = api.run(f"{ENTITY}/object_detection/{bckbone_rname}")

    fname = [f.name for f in run.files() if "ckpt" in f.name][0]
    run.file(fname).download(replace = True, root = ".")
    backbone = model_cls.load_from_checkpoint(fname)
    backbone = backbone.model[1]
    os.remove(fname)
    backbone = _resnet_fpn_extractor(backbone, trainable_layers=5)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model

def _resnet_fpn_extractor(
    backbone: nn.Module,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
) -> BackboneWithFPN:

    # select layers that wont be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=None)
    


class RCNN_Predictor(pl.LightningModule):
    def __init__(self, len_dataloader, rcnn_head_model=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
       # if rcnn_head_model is None:
       #    model = RCNN(len_dataloader = 0,**vars(kwargs))
       #    self.classifier = model.box_predictor.cls_score
       # else:
        self.classifier = rcnn_head_model
        self.matcher = HungarianMatcher(cost_class=1., cost_bbox=0., cost_giou=0.)

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        pred_mask = outputs["pred_mask"]

        idx = self._get_src_permutation_idx(indices)
        target_mask = torch.zeros((idx[0].max()+1,idx[1].max()+10),device = src_logits.device) # +10 to make sure that the matrix is big enough, 0s are ignored anyway
        target_mask[idx] = 1



        alignment = [(src_logits[i,indices[i][0],:], targets[i]["labels"][indices[i][1]], pred_mask[i,indices[i][0]], target_mask[i,indices[i][1]] ) for i in range(len(indices))]

        losses = torch.cat([F.cross_entropy(alignment[i][0],alignment[i][1], reduction = "none") * alignment[i][2] * alignment[i][3] for i in range(len(alignment))])

        loss_ce = losses.sum() / (losses!=0).sum()

        #src_logits_cat = torch.stack([src_logits[i,indices[i][0],:] for i in range(len(indices))])
        #targets_cat =  torch.stack([ targets[i]["labels"][indices[i][1]] for i in range(len(indices))])
        #pred_mask_cat =  torch.stack([ pred_mask[i,indices[i][0]] for i in range(len(indices))])
        #target_mask_cat =  torch.stack([ target_mask[i,indices[i][1]] for i in range(len(indices))])

        #target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        #target_classes = torch.full(src_logits.shape[:2], self.num_clCsses,
                                    #dtype=torch.int64, device=src_logits.device)
        #target_classes[idx] = target_classes_o

        #loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)#, self.empty_weight.to(src_logits.device))
        losses = {'loss_ce': loss_ce}

        #if log:
        #    # TODO this should probably be a separate loss, not hacked in this one here
        #    losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.classifier.parameters(), lr=self.hparams["lr"])
        return opt

    def forward(self,tensors, targets, val = False):
        tensors = torch.cat(tensors)
        output = self.classifier(tensors)
        
        target_pointers = [len(t["labels"]) for t in targets]
        target_boxes = [t["boxes"] for t in targets]
        shift = 0
        pred_logits=[]
        pred_boxes=[]
        index = 0
        #THIS (VAL) ONLY WORKS IN CASE OF BATCHSIZE 1
        #if val:
        #    pred_logits.append(output)
        #    return pred_logits
        for p in target_pointers:
            pred_logits.append(output[shift:p+shift])
            pred_boxes.append(torch.as_tensor(target_boxes[index], device=output[shift:p+shift].device))
            index +=1
            shift +=p
#        import ipdb; ipdb.set_trace()
        if val:
           return pred_logits
        pred_boxes_padded = nn.utils.rnn.pad_sequence(pred_boxes,batch_first = True)

        #    pred_logits =  [loss_["class_logits"] for loss_ in loss_dict]
        pred_logits_padded = nn.utils.rnn.pad_sequence(pred_logits,batch_first = True)
#        import ipdb; ipdb.set_trace()
        mask = torch.zeros(pred_boxes_padded.shape[:-1],device = pred_boxes_padded.device)
        for i in range(len(pred_boxes)):
            mask[i,:len(pred_boxes[i])] = 1

        outputs = {"pred_logits": pred_logits_padded,"pred_boxes": pred_boxes_padded, "pred_mask": mask}
        targets = [{"labels": t["labels"], "boxes":t["boxes"]} for t in targets]
        #import ipdb; ipdb.set_trace()
        try:
            indices = self.matcher(outputs, targets)
        except:
            import ipdb; ipdb.set_trace()
        loss_dict = self.loss_labels(outputs,targets, indices = indices)
#        import ipdb; ipdb.set_trace()
#        loss = 0
        return loss_dict
 
    def compute_accuracy(self,targets,preds):
        return torch.Tensor([torch.equal(targets[i]["labels"].sort()[0],torch.argmax(preds[i], dim=1).sort()[0]) for i in range(len(targets)) ]).mean()
       # import ipdb; ipdb.set_trace()
       # return torch.Tensor([torch.equal(targets[i]["labels"].sort()[0],preds[i]["labels"].sort()[0]) for i in range(len(targets)) ]).mean()

    def training_step(self,batch,batch_idx):
        tensors, targets, _ = batch
        loss_dict = self(tensors, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss",losses)
        return losses

    def validation_step(self,batch,batch_idx):
        tensors, targets, _ = batch
        loss_dict = self(tensors, targets, val=True)
        accuracy = self.compute_accuracy(targets,loss_dict)
        self.log("val_acc", accuracy,on_epoch = True)
        return

    def test_step(self,batch,batch_idx):
        tensors, targets, _ = batch
        loss_dict = self(tensors, targets, val = True)

        accuracy = self.compute_accuracy(targets,loss_dict)
        self.log("test_acc", accuracy,on_epoch = True)
        return

class RCNN(pl.LightningModule):
    def __init__(self, len_dataloader, hidden_layer, num_classes, score_thresh,model_type = "mask_rcnn", pre_trained = True, backbone_run_name = None, target_data_cls=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        self.model, self.box_predictor = get_model(num_classes,score_thresh, model_type, pretrained = pre_trained, bckbone_rname = backbone_run_name)
        
        self.len_dataloader = len_dataloader
        self.automatic_optimization = False

        self.pre_trained = pre_trained
        self.hungarian_fine_tuning = False
        self.target_data_cls = target_data_cls

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr= self.hparams["lr"],
                                momentum=self.hparams["momentum"], weight_decay=self.hparams["weight_decay"])
        if self.hungarian_fine_tuning:
            opt = torch.optim.Adam(self.model.roi_heads.parameters(), lr=1e-4)
            return opt

        if self.pre_trained:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, self.len_dataloader - 1)

            epoch0_lr_scheduler = utils.warmup_lr_scheduler(opt, warmup_iters, warmup_factor)
    
            lr_scheduler = torch.optim.lr_scheduler.StepLR(opt,
                                                   step_size=self.hparams["lr_step_size"],
                                                   gamma=0.1)
        
            return [opt], [epoch0_lr_scheduler,lr_scheduler]

        else:
            return opt
    
    def switch_to_hungarian(self):
        self.hungarian_fine_tuning = True
        self.matcher = HungarianMatcher(cost_class=1., cost_bbox=0., cost_giou=0.)
        
        #self.empty_weight = torch.ones(self.num_classes-1)
        #self.empty_weight[-1] = 0.1

    def compute_accuracy(self,targets,preds):
        return torch.Tensor([torch.equal(targets[i]["labels"].sort()[0],preds[i]["labels"].sort()[0]) for i in range(len(targets)) ]).mean()

    def compute_sum_accuracy(self,targets,preds):
        #import ipdb; ipdb.set_trace()
        return torch.Tensor([torch.sum(targets[i]["labels"].sort()[0]) == torch.sum(preds[i]["labels"].sort()[0]) for i in range(len(targets)) ]).mean()
 
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        pred_mask = outputs["pred_mask"]
        
        idx = self._get_src_permutation_idx(indices)
        target_mask = torch.zeros((idx[0].max()+1,idx[1].max()+10),device = src_logits.device) # +10 to make sure that the matrix is big enough, 0s are ignored anyway
        target_mask[idx] = 1

       
        
        alignment = [(src_logits[i,indices[i][0],:], targets[i]["labels"][indices[i][1]], pred_mask[i,indices[i][0]], target_mask[i,indices[i][1]] ) for i in range(len(indices))]
        
        losses = torch.cat([F.cross_entropy(alignment[i][0],alignment[i][1], reduction = "none") * alignment[i][2] * alignment[i][3] for i in range(len(alignment))])

        loss_ce = losses.sum() / (losses!=0).sum()

        #src_logits_cat = torch.stack([src_logits[i,indices[i][0],:] for i in range(len(indices))])
        #targets_cat =  torch.stack([ targets[i]["labels"][indices[i][1]] for i in range(len(indices))])
        #pred_mask_cat =  torch.stack([ pred_mask[i,indices[i][0]] for i in range(len(indices))])
        #target_mask_cat =  torch.stack([ target_mask[i,indices[i][1]] for i in range(len(indices))])

        #target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        #target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    #dtype=torch.int64, device=src_logits.device)
        #target_classes[idx] = target_classes_o
        
        #loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)#, self.empty_weight.to(src_logits.device))
        losses = {'loss_ce': loss_ce}

        #if log:
        #    # TODO this should probably be a separate loss, not hacked in this one here
        #    losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def forward(self,images, targets, val = False):

        if ((self.hungarian_fine_tuning) and (not val)):
            self.model.eval()
            loss_dict = self.model(images, targets)
            
            pred_boxes = [loss_["boxes"] for loss_ in loss_dict]
            pred_boxes_padded = nn.utils.rnn.pad_sequence(pred_boxes,batch_first = True)
            
            pred_logits =  [loss_["class_logits"] for loss_ in loss_dict]
            pred_logits_padded = nn.utils.rnn.pad_sequence(pred_logits,batch_first = True)

            mask = torch.zeros(pred_boxes_padded.shape[:-1],device = pred_boxes_padded.device)
            for i in range(len(pred_boxes)):
                mask[i,:len(pred_boxes[i])] = 1
            
            outputs = {"pred_logits": pred_logits_padded,"pred_boxes": pred_boxes_padded, "pred_mask": mask}
            targets = [{"labels": t["labels"]-1, "boxes":t["boxes"]} for t in targets]
    
            indices = self.matcher(outputs, targets)
            loss_dict = self.loss_labels(outputs,targets, indices = indices) 
        else:
            loss_dict = self.model(images, targets)
        
        return loss_dict

    def training_step(self,batch,batch_idx):
        images, targets, _ = batch
        #losses = self.dummy(images[0].sum().view(-1,1))
        opt = self.optimizers()
        opt.zero_grad()

        loss_dict = self(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        try:
            self.manual_backward(losses)
            opt.step()
        except RuntimeError:
            import ipdb; ipdb.set_trace()

        self.log("train_loss",losses)
        if self.hungarian_fine_tuning:
            return losses
        if self.trainer.current_epoch == 0:
            if self.pre_trained:
                self.lr_schedulers()[0].step()
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])

        return losses
    
    def training_epoch_end(self, data):
        if self.hungarian_fine_tuning:
            return
        if self.pre_trained:
            self.lr_schedulers()[1].step()

    def validation_step(self,batch,batch_idx):
        images, targets, _ = batch
        self.model.eval()
        loss_dict = self.model(images)
        #loss_dict = self(images, targets, val = True)
        if self.target_data_cls is not None:
            accuracy = self.target_data_cls.compute_accuracy(targets,loss_dict)
        else:
            accuracy = self.compute_accuracy(targets,loss_dict)
        self.log("val_acc", accuracy,on_epoch = True)
        return 
    
    def test_step(self,batch,batch_idx):
        images, targets, _ = batch
        self.model.eval()
        #loss_dict = self(images, targets, val = True)
        loss_dict = self.model(images)
        if self.target_data_cls is not None:
            accuracy = self.target_data_cls.compute_accuracy(targets,loss_dict)
            #accuracy = self.compute_accuracy(targets,loss_dict)
        else:
            accuracy = self.compute_accuracy(targets,loss_dict)
        self.log("test_acc", accuracy,on_epoch = True)
        return 

    def predict_step(self,batch, batch_idx):
        images, targets, idx = batch
        self.model.eval()
        loss_dict = self.model(images)

        #preds = [dict(boxes = l["boxes"], labels = l["labels"], scores = l["scores"]) for l in loss_dict]
        #target = [dict(boxes=t["boxes"], labels = t["labels"]) for t in targets]
        #metric = MeanAveragePrecision()
        #metric.update(preds, target)
        #mAP = metric.compute()

        if "boxes" in targets[0].keys():
            return {"box_features":[l["box_features"] for l in loss_dict],"boxes_true":[t["boxes"] for t in targets], "scores":[l["scores"] for l in loss_dict],"targets":[t["labels"] for t in targets],  "preds":[l["labels"] for l in loss_dict],"idx":idx, "boxes": [l["boxes"] for l in loss_dict]}
        else:
            return {"box_features":[l["box_features"] for l in loss_dict],"boxes_true":[None for t in targets], "scores":[l["scores"] for l in loss_dict],"targets":[t["labels"] for t in targets],  "preds":[l["labels"] for l in loss_dict],"idx":idx, "boxes": [l["boxes"] for l in loss_dict]}

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--hidden_layer', type=int, default=256)
        parser.add_argument('--score_thresh', type=float, default=0.65, help = "score_threshold for the rcnn")
        parser.add_argument('--model_type', type=str, default="rcnn", help = "type of model to use")
        parser.add_argument('--backbone_run_name', type=str, default=None, help = "run_name of CNN for backbone to be used")
        parser.add_argument('--pre_trained', type=str2bool, default=True)
        parser.add_argument('--target_data_cls', type=str, default=None)
        return parser



def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            for target in targets:
                print(target["image_id"])
            print(loss_dict_reduced)
            torch.save(model.state_dict(), f"chemgrapher_maskrcnn_preempted_epoch_{epoch}")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

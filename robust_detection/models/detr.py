# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from robust_detection.models.detr_utils import box_ops
from robust_detection.models.detr_utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from robust_detection.models.backbone import build_backbone
from robust_detection.models.matcher import build_matcher
from robust_detection.models.segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from robust_detection.models.transformer import build_transformer
import pytorch_lightning as pl


class DETR(pl.LightningModule):
    """ This is the DETR module that performs object detection """
    def __init__(self, len_dataloader, pretrained, hidden_dim, pos_emb, lr_backbone, masks, bck_bone, dilation, dropout, nheads, dim_feedforward, enc_layers, dec_layers, pre_norm, set_cost_class, set_cost_bbox, set_cost_giou, bbox_loss_coef, giou_loss_coef, eos_coef, num_classes, num_queries, aux_loss=False, target_data_cls=None, **kwargs):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.save_hyperparameters()
        self.len_dataloader = len_dataloader
        if pretrained:
           model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
 #          model.train()
           #import ipdb; ipdb.set_trace()
           self.backbone    = model.backbone
           self.transformer = model.transformer
           hidden_dim       = self.transformer.d_model
           self.bbox_embed  = model.bbox_embed
           self.query_embed = model.query_embed
           self.input_proj  = model.input_proj
        else:
           self.backbone = build_backbone(hidden_dim, pos_emb, lr_backbone, masks, bck_bone, dilation)
           self.transformer = build_transformer(hidden_dim, dropout, nheads, dim_feedforward, enc_layers, dec_layers, pre_norm)
           #self.num_queries = num_queries
           hidden_dim = self.transformer.d_model
        #self.num_queries = num_queries
        #self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
           self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
           self.query_embed = nn.Embedding(num_queries, hidden_dim)
           self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        self.num_queries = num_queries
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.aux_loss = aux_loss
        self.target_data_cls = target_data_cls

        self.matcher = build_matcher(set_cost_class, set_cost_bbox, set_cost_giou)
        weight_dict = {'loss_ce': 1, 'loss_bbox': bbox_loss_coef}
        weight_dict['loss_giou'] = giou_loss_coef
        # TODO this is a hack
        if aux_loss:
           aux_weight_dict = {}
           for i in range(dec_layers - 1):
               aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
           weight_dict.update(aux_weight_dict)
        if set_cost_bbox != 0:
           losses = ['labels', 'boxes', 'cardinality']
        else:
           losses = ['labels', 'cardinality']
        self.criterion = SetCriterion(num_classes, matcher=self.matcher, weight_dict=weight_dict,
                             eos_coef=eos_coef, losses=losses)
#        self.criterion.train()
        self.postprocessors = {'bbox': PostProcess()}
        self.automatic_optimization = False

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'box_features': hs[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    def compute_accuracy(self,targets,preds):
        labels = []
        for i in range(len(targets)):
            keep = [preds[i]["scores"] > 0.5]
            #import ipdb; ipdb.set_trace()
            #keep.append([preds[i]["scores"] > 0.9])
            labels.append(preds[i]["labels"][keep])
        
        #keep = [keep.append([preds[i]["scores"] > 0.9]) for i in range(len(targets))]
        #import ipdb; ipdb.set_trace()
        #preds = torch.Tensor([preds[i][preds[i]["scores"] > 0.9] for i in range(len(targets))])
        #import ipdb; ipdb.set_trace()
        #return torch.Tensor([torch.equal(targets[i]["labels"].sort()[0],preds[i]["labels"].sort()[0]) for i in range(len(targets)) ]).mean()
        return torch.Tensor([torch.equal(targets[i]["labels"].sort()[0],labels[i][torch.where(labels[i] != 0)].sort()[0]) for i in range(len(targets)) ]).mean()
    
    def compute_sum_accuracy(self,targets,preds):
        labels = []
        for i in range(len(targets)):
            keep = [preds[i]["scores"] > 0.5]
            labels.append(preds[i]["labels"][keep])
        #import ipdb; ipdb.set_trace()
        return torch.Tensor([torch.sum(targets[i]["labels"].sort()[0]) == torch.sum(labels[i][torch.where(labels[i] != 0)].sort()[0]) for i in range(len(targets)) ]).mean()

    def validation_step(self,batch,batch_idx):
        images, targets, _ = batch
        outputs = self(images)
        if "orig_size" in targets[0].keys():
           orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
           outputs = self.postprocessors['bbox'](outputs,orig_target_sizes)
        else:
           orig_target_sizes = torch.stack([torch.tensor([im.size(dim=1),im.size(dim=2)]) for im in images], dim=0)
           orig_target_sizes = orig_target_sizes.to(images[0].device)
           outputs = self.postprocessors['bbox'](outputs,orig_target_sizes)
#        import ipdb; ipdb.set_trace()
        if self.target_data_cls is not None:
            labels = []
            for i in range(len(targets)):
                keep = [outputs[i]["scores"] > 0.5]
                label_list = outputs[i]["labels"][keep]
                label_list = label_list[torch.where(label_list != 0)]
                #import ipdb; ipdb.set_trace()
                #dicttemp['labels']=label_list
                labels.append(dict(labels=label_list))
            #import ipdb; ipdb.set_trace()
            accuracy = self.target_data_cls.compute_accuracy(targets,labels)
            #if accuracy > 0.03:
            #    import ipdb; ipdb.set_trace()
            #accuracy = self.compute_accuracy(targets,outputs)
        else:
            accuracy = self.compute_accuracy(targets,outputs)
        self.log("val_acc", accuracy,on_epoch = True)
        return

    def test_step(self,batch,batch_idx):
        images, targets, _ = batch
        outputs = self(images)
        if "orig_size" in targets[0].keys():
           orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
           outputs = self.postprocessors['bbox'](outputs,orig_target_sizes)
        else:
           orig_target_sizes = torch.stack([torch.tensor([im.size(dim=1),im.size(dim=2)]) for im in images], dim=0)
           orig_target_sizes = orig_target_sizes.to(images[0].device)
           outputs = self.postprocessors['bbox'](outputs,orig_target_sizes)
        if self.target_data_cls is not None:
            labels = []
            dicttemp = {}
            for i in range(len(targets)):
                keep = [outputs[i]["scores"] > 0.5]
                label_list = outputs[i]["labels"][keep]
                label_list = label_list[torch.where(label_list != 0)]
                labels.append(dict(labels=label_list))
            accuracy = self.target_data_cls.compute_accuracy(targets,labels)
        else:
            accuracy = self.compute_accuracy(targets,outputs)
        self.log("test_acc", accuracy,on_epoch = True)
        return

    def predict_step(self,batch, batch_idx):
        images, targets, idx = batch
        outputs = self(images)
        #import ipdb; ipdb.set_trace()
        if "orig_size" in targets[0].keys():
           orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
           outputs = self.postprocessors['bbox'](outputs,orig_target_sizes)
        else:
           orig_target_sizes = torch.stack([torch.tensor([im.size(dim=1),im.size(dim=2)]) for im in images], dim=0)
           orig_target_sizes = orig_target_sizes.to(images[0].device)
           outputs = self.postprocessors['bbox'](outputs,orig_target_sizes)
        labels = []
        scores = []
        boxes = []
        box_features = []
        if "boxes" in targets[0].keys():
           boxes_true = [box_ops.box_cxcywh_to_xyxy(t["boxes"]) for t in targets]
        # and from relative [0, 1] to absolute [0, height] coordinates
           img_h, img_w = orig_target_sizes.unbind(1)
        #from IPython.core.debugger import set_trace
        #set_trace()
           scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
           boxes_true = [bt * s_fct[None, :] for bt, s_fct in zip(boxes_true,scale_fct)]
        for i in range(len(images)):
            #import ipdb; ipdb.set_trace()
            keep = [outputs[i]["scores"] > 0.05]
            #import ipdb; ipdb.set_trace()
            #keep.append([preds[i]["scores"] > 0.9])
            labels.append(outputs[i]["labels"][keep])
            scores.append(outputs[i]["scores"][keep])
            boxes.append(outputs[i]["boxes"][keep])
            box_features.append(outputs[i]["box_features"][keep])

        #return {"boxes_true":[t["boxes"] for t in targets], "scores":[l["scores"] for l in outputs],"targets":[t["labels"] for t in targets],  "preds":[l["labels"] for l in outputs],"idx":idx, "boxes": [l["boxes"] for l in outputs], "box_features": [l["box_features"] for l in outputs]}
        if "boxes" in targets[0].keys():
           return {"boxes_true":[t for t in boxes_true], "scores":scores, "targets":[t["labels"] for t in targets],  "preds":labels, "idx":idx, "boxes": boxes, "box_features": box_features}
        else:
           return {"boxes_true":[None for t in targets], "scores":scores, "targets":[t["labels"] for t in targets],  "preds":labels, "idx":idx, "boxes": boxes, "box_features": box_features}

    def configure_optimizers(self):
        opt          = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(opt,
                                                   step_size=self.hparams["lr_step_size"])
        return [opt], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        images, targets, _ = batch
        outputs = self(images)
#        import ipdb; ipdb.set_trace()
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        self.log("train_loss",losses)
        self.manual_backward(losses)
        opt.step()
#        import ipdb; ipdb.set_trace()
        return losses

    def training_epoch_end(self, data):
        sch = self.lr_schedulers()
        sch.step()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        #no_boxes_mask = torch.cat([ torch.ones(len(v["boxes"])) * (v["boxes"].cpu().sum()==0) for v in targets])
        keep = [(targets[i]["boxes"].cpu().numpy().sum()!=0) for i in range(len(targets))]
        #keep=[targets[i]['box_loss_mask']==0 for i in range(len(targets))]
        targets = [targets[i] for i in np.where(keep)[0].tolist()]
        outputs["pred_boxes"] = outputs["pred_boxes"][keep]
        indices = [indices[i] for i in np.where(keep)[0].tolist()]
       # num_boxes = sum(len(t["labels"]) for t in targets)
       # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if len(indices) == 0:
            losses = {}
            losses['loss_bbox'] = 0
            losses['loss_giou'] = 0
            return losses
        #import ipdb; ipdb.set_trace() 
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        import ipdb; ipdb.set_trace()
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox, box_features = outputs['pred_logits'], outputs['pred_boxes'], outputs['box_features']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b, 'box_features': bf} for s, l, b, bf in zip(scores, labels, boxes, box_features)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors

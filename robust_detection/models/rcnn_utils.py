import torch
from torchvision.models.detection.faster_rcnn import RoIHeads
import torch.nn.functional as F
from torch.jit.annotations import Optional, List, Dict, Tuple
from torchvision.ops import boxes as box_ops
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.roi_heads import maskrcnn_inference, fastrcnn_loss, maskrcnn_loss
from torchvision.transforms import functional as Ftrans

class WrapModel(torch.nn.Module):
    def __init__(self,rcnn_head_model):
        super().__init__()
        self.classifier = rcnn_head_model
    def forward(self,x):
        return torch.nn.functional.softmax(self.classifier(x)[...,1:],-1)


class myRoIHeads(RoIHeads):
  def __init__(self, *args, **kwargs):
    super(my_class, self).__init__(*args, **kwargs)

  def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes, box_features):
        # type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]])
        #import ipdb; ipdb.set_trace()
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)
        #import ipdb; ipdb.set_trace()

        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            # TODO : remove this when ONNX support dynamic split sizes
            # and just assign to pred_boxes instead of pred_boxes_list
            pred_boxes_list = [pred_boxes]
            pred_scores_list = [pred_scores]
            box_features_list = [box_features]
        else:
            pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
            pred_scores_list = pred_scores.split(boxes_per_image, 0)
            box_features_list = box_features.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_class_logits = []
        all_box_features = []
        for boxes, scores, image_shape, features in zip(pred_boxes_list, pred_scores_list, image_shapes, box_features_list):
           # import ipdb; ipdb.set_trace()
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            
            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            scores_copy = scores
            #features_copy = features
         #   import ipdb; ipdb.set_trace()
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            scores_copy = scores
            dims_score = list(scores_copy.size())
            scores_idx = torch.arange(start=0, end=dims_score[0], dtype=torch.long)
            scores_idx = torch.unsqueeze(scores_idx, 1)
            scores_idx = scores_idx.expand(dims_score)
            #import ipdb; ipdb.set_trace()
            scores_idx = scores_idx.reshape(-1)
            #import ipdb; ipdb.set_trace()
            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, scores_idx = boxes[inds], scores[inds], labels[inds], scores_idx[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, scores_idx = boxes[keep], scores[keep], labels[keep], scores_idx[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            #keep = box_ops.nms(boxes, scores, iou_threshold = 0.9)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels, scores_idx = boxes[keep], scores[keep], labels[keep], scores_idx[keep]
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            scores_copy = scores_copy[scores_idx]
            features = features[scores_idx]
            all_class_logits.append(scores_copy)
            all_box_features.append(features)
            #import ipdb; ipdb.set_trace()
        return all_boxes, all_scores, all_labels, all_class_logits, all_box_features

  def forward(self, features, proposals, image_shapes, targets=None):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        #import ipdb; ipdb.set_trace()

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels, class_logits, box_features = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes, box_features)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                        "class_logits": class_logits[i],
                        "box_features": box_features[i]
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                mask_logits = torch.tensor(0)
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses

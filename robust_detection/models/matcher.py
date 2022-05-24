# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from robust_detection.models.detr_utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, masking_empty_boxes = True):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        self.masking_empty_boxes = masking_empty_boxes
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                 "pred_mask" : Tensor of dim [batch_size, num_querries] with 1 if the prediction is observed and 0 if not. Ignored if not fed.

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

            Case with  no target box. For no target-box : boxes should be a tensor of zeros of size [num_target_boxes,4]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        if "pred_mask" in outputs:
            out_mask = outputs["pred_mask"].flatten()

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        if self.masking_empty_boxes:
            no_boxes_mask = torch.cat([ torch.ones(len(v["boxes"])).to(v["boxes"].device) * (v["boxes"].sum()==0) for v in targets])
            if self.cost_giou == 0:
               C = self.cost_bbox * cost_bbox * (1-no_boxes_mask)[None,:] + self.cost_class * cost_class
            else:
               C = self.cost_bbox * cost_bbox * (1-no_boxes_mask)[None,:] + self.cost_class * cost_class + self.cost_giou * cost_giou * (1-no_boxes_mask)[None,:]
        else:
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou 
        
        if "pred_mask" in outputs:
            #import ipdb; ipdb.set_trace()
            practical_inf = C.max() + 1000
            C[out_mask==0,:] = practical_inf

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(set_cost_class, set_cost_bbox, set_cost_giou, masking_empty_boxes = True):
    return HungarianMatcher(cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou, masking_empty_boxes = masking_empty_boxes)

if __name__ == "__main__":
    #testing

    matcher = HungarianMatcher(cost_class=0.5, cost_bbox=0.5, cost_giou=0.5)
    pred_boxes = torch.Tensor([1.,2.,100.,200.])[None,None,:].repeat(32,16,1)
    pred_logits = torch.rand((32,16,4))

    pred_boxes = [pred_boxes[i,:(12 + i//4),:] for i in range(pred_boxes.shape[0])]
    pred_boxes_padded = nn.utils.rnn.pad_sequence(pred_boxes,batch_first = True)
    pred_logits = [pred_logits[i,:(12 + i//4),:] for i in range(pred_logits.shape[0])]
    pred_logits_padded = nn.utils.rnn.pad_sequence(pred_logits,batch_first = True)

    mask = torch.zeros(pred_boxes_padded.shape[:-1])
    for i in range(len(pred_boxes)):
        mask[i,:len(pred_boxes[i])] = 1


    outputs = {"pred_logits": pred_logits_padded,"pred_boxes": pred_boxes_padded, "pred_mask": mask}
    targets = [{"labels":torch.zeros(2).long(),"boxes": torch.zeros((2,4))} for _ in range(16)] +  [{"labels":torch.zeros(2).long(),"boxes": torch.Tensor([[1.,20.,100.,200.],[1.,40.,100.,200.]])} for _ in range(16)]
    
    outs = matcher(outputs, targets)
    print(outs)

from collections import defaultdict, deque
import datetime
import errno
import os
import time
import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import *
from PIL import Image
from scipy import ndimage, misc
import skimage.draw as draw
import robust_detection.transforms as T
import pandas as pd
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import RoIHeads
import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops
from torchvision.ops import misc as misc_nn_ops

from torchvision.ops import roi_align
from torch.jit.annotations import Optional, List, Dict, Tuple
from torchvision.models.detection.roi_heads import maskrcnn_inference
from torchvision.transforms import functional as Ftrans

from rdkit import Chem


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

class MNIST_detect_wrapper(nn.Module):
  def __init__(self, conf, net):
    super().__init__()
    self.net = net

  def forward(self, input):
    img = input[:,0,:,:]
    number = input[:,1,:,:][:,0,0,0]
    number = number.to(dtype=torch.long)
   # import ipdb; ipdb.set_trace()
    #number = int(input[-1][0][0][0])
    net_preds = self.net(img.unbind())
    class_logits = [e['class_logits'] for e in net_preds]
   # import ipdb; ipdb.set_trace()
    #output_labels = net_preds[0]['class_logits']
    output_labels = [i[j] for i,j in zip(class_logits,number)]
    return output_labels

def filter_data(model_dpl, dataset, device, threshold = 0.9):

    wrap_model = model_dpl.networks['mnist_net'].network_module
    
    data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
    maskrcnn = wrap_model.net

    import copy
    og_label_len_list = np.array(dataset.label_len_list)
    index_names_og = dataset.index_names
    og_label_idx_list = dataset.label_idx_list
    
    new_dataset = copy.deepcopy(dataset)
    
    count_idx = 0
    with torch.no_grad():
        print("Filtering dataset ......")
        for data_item in tqdm.tqdm(data_loader):
            img_items = data_item[0]
            labels    = data_item[1]
            img_items = img_items.to(device)
            labels    = labels.to(device)
            outputs   = wrap_model(img_items)
            # Filtering the outputs which are certain. Currently does only remove the most sure digit
            if len(outputs.shape) == 1:
                outputs = [outputs]
            # Filtering the outputs which are certain. For some images, all digits might actually be dropped
            for i,output in enumerate(outputs):
                if (output.max() > threshold):
                    pred_digit = output.argmax()
                    if count_idx + i >= len(index_names_og):
                        import ipdb; ipdb.set_trace()
                    img_idx = index_names_og[count_idx + i][0]
                    if ((new_dataset.full_labels[img_idx][pred_digit]>0) and (new_dataset.full_labels[img_idx].sum()>2)): #only remove digits if there are more than 2 remaining digits
                        digit_idx = index_names_og[count_idx + i][1]
                        new_dataset.label_idx_list[img_idx].remove(digit_idx) # remove this digit from the output of the rcnn
                        new_dataset.full_labels[img_idx][pred_digit] -= 1 # update the label of the whole image
                        #import ipdb; ipdb.set_trace()
            count_idx += len(labels)
        print("Done")
        new_index_names = []
        new_label_len_list = []
        for i in range(len(new_dataset.label_idx_list)):
            new_label_len_list.append(len(new_dataset.label_idx_list[i]))
            for j in new_dataset.label_idx_list[i]:
                new_index_names.append((i,j))
        new_total_number = len(new_index_names)

        new_dataset.index_names = new_index_names
        new_dataset.label_len_list = new_label_len_list
        new_dataset.total_number = new_total_number
        
        new_dataset.full_labels = [l for i,l in enumerate(new_dataset.full_labels) if new_dataset.label_len_list[i]>0]
        new_dataset.label_len_list = [l for l in new_dataset.label_len_list if l>0]

        return new_dataset 


class MNIST_Classifier(nn.Module):
    def __init__(self, net, N=11, with_softmax=True):
        super(MNIST_Classifier, self).__init__()
        self.net = net
        self.with_softmax = with_softmax
        if with_softmax:
            self.softmax = nn.Softmax(1)
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
        )

    def forward(self, input):
        img = input[:,0,:,:]
        number = input[:,1,:,:][:,0,0,0]
        number = number.to(dtype=torch.long)
   # import ipdb; ipdb.set_trace()
    #number = int(input[-1][0][0][0])
        net_preds = self.net(img.unbind())
        x = [e['box_features'] for e in net_preds]
        x = [i[j] for i,j in zip(x,number)]
        x = torch.stack(x)
        #import ipdb; ipdb.set_trace()
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x.squeeze(0)

#class Approximate_MNIST_Classifier(MNIST_Classifier):
#    def __init__(self,net, N = 11, with_softmax = True):
#        super().__init__(net = net, N = N, with_softmax = with_softmax)
#
#    def forward(self,input)




class MNIST_SingleDetection_Dataset(Dataset):

      def __init__(self, data_dir):
          self.data_dir = data_dir
          self.num_images=0
          self.total_number=0
          self.index_names=[]
          self.label_len_list=[]
          self.label_idx_list = []
          self.full_labels = []

          for name in os.listdir(f"{self.data_dir}/labels/"):
              self.num_images+=1
          for i in range(self.num_images):
              labels_df = pd.read_csv(f"{self.data_dir}/labels/{i}.txt")
            
              digits = labels_df.groupby("label").size().index.values
              counts = labels_df.groupby("label").size().values
        
              num_classes = 10
              counts_vec = np.zeros(num_classes)
              counts_vec[digits] = counts
              self.full_labels.append(counts_vec)

              num_labels = len(labels_df.index)
              for j in range(num_labels):
                  self.index_names.append([i, j])
              self.total_number+=num_labels
              self.label_len_list.append(num_labels)
              self.label_idx_list.append([w for w in range(num_labels)])

      def __len__(self):
          return self.total_number

      def __getitem__(self, idx):
        #  print(idx)
         # import ipdb; ipdb.set_trace()
          image_idx = self.index_names[idx][0]
          #print(image_idx)
          imagename=f"{self.data_dir}/images/{str(image_idx)}.png"
          img = Image.open(imagename).convert("L")
          labels_df = pd.read_csv(f"{self.data_dir}/labels/{str(image_idx)}.txt")
          label_row = labels_df.iloc[self.index_names[idx][1]]
          label = int(label_row['label'])+1
          digit_number = self.index_names[idx][1]
          img = Ftrans.to_tensor(img)
          number = torch.FloatTensor([digit_number])
          number = number.expand(img.size())
          stacked = torch.stack((img,number))
          return stacked, label

class SMILES_SingleAtoms_Dataset(Dataset):
    def __init__(self, smiles_file):
        self.smiles_file   = smiles_file
        self.smiles_df     = pd.read_csv(self.smiles_file)
        self.len_df        = self.smiles_df.shape[0]
        self.total_number  = 0
        self.atom_len_list = []
        self.counter_list  = []
        self.smiles_list   = []
        for i in range(self.len_df):
            df_row             = self.smiles_df.iloc[i]
            smiles             = df_row['smiles']
            m                  = Chem.MolFromSmiles(smiles)
            #import ipdb; ipdb.set_trace()
            try:
              num_atoms          = m.GetNumAtoms()
            except:
              import ipdb; ipdb.set_trace()
            for j in range(num_atoms):
                self.counter_list.append([i,j])
            self.total_number += num_atoms
            self.atom_len_list.append(num_atoms)
            self.smiles_list.append(smiles)

    def __len__(self):
        return self.total_number

    def __getitem__(self, idx):
        counters = self.counter_list[idx]
        df_row   = self.smiles_df.iloc[counters[0]]
        smiles   = df_row['smiles']
        number_atoms   = counters[1]
        imagename = df_row['filename']
        img      = Image.open(f"{imagename}.png").convert("L")
        img      = Ftrans.to_tensor(img)
        number = torch.FloatTensor([number_atoms])
        number = number.expand(img.size())
        stacked = torch.stack((img,number))
        return stacked, smiles

class SMILES_CountAtoms_Dataset(Dataset):

    def __init__(self, smiles_file):
        self.smiles_file = smiles_file
        self.smiles_df   = pd.read_csv(self.smiles_file)
        self.len_df      = self.smiles_df.shape[0]

    def __len__(self):
        return self.len_df

    def __getitem__(self, idx):
        #self.dict_atom = {'C': 1, 'H': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'Se': 10, 'P': 11, 'B': 12, 'Si': 13}
        atomnumber_dict = {6: 1, 1: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7, 35: 8, 53: 9, 34: 10, 15: 11, 5: 12, 14: 13}
        count_atoms = np.zeros(15)
        df_row = self.smiles_df.iloc[idx]
        #import ipdb; ipdb.set_trace()
        imagename = df_row['filename'] 
        smiles   = df_row['smiles'] 
        img      = Image.open(f"{imagename}.png").convert("L")
        img      = Ftrans.to_tensor(img)
        m        = Chem.MolFromSmiles(smiles)
        for atom in m.GetAtoms():
            count_atoms[atomnumber_dict[atom.GetAtomicNum()]] +=1
            #print(atom.GetAtomicNum())
        #values, counts = np.unique(count_atoms, return_counts=True)
        return img, count_atoms


class MNIST_Detection_Dataset(Dataset):

      def __init__(self, data_dir, transforms):
          self.data_dir = data_dir
          self.transforms = transforms

      def __len__(self):
          return len([name for name in os.listdir(f"{self.data_dir}/images/")])

      def __getitem__(self, idx):
          imagename=f"{self.data_dir}/images/{str(idx)}.png"
          img = Image.open(imagename).convert("L")
          labels_df = pd.read_csv(f"{self.data_dir}/labels/{str(idx)}.txt")
          labels = []
          boxes = []
          mask_objs=np.zeros(img.size)
          obj_idx = 1
          for index, row in labels_df.iterrows():
              labels.append(int(row['label'])+1)
              xmin = int(row['xmin'])
              xmax = int(row['xmax'])
              ymin = int(row['ymin'])
              ymax = int(row['ymax'])
              boxes.append([xmin, ymin, xmax, ymax])
              start = (xmin,ymin)
              end = (xmax,ymax)
              #import ipdb; ipdb.set_trace()
              rr, cc = draw.rectangle(start, end=end, shape=img.size)
              mask_objs[rr, cc] = obj_idx
              obj_idx+=1

          # instances are encoded as different colors
          obj_ids = np.unique(mask_objs)
          # first id is the background, so remove it
          obj_ids = obj_ids[1:]

          # split the color-encoded mask into a set
          # of binary masks
          obj_masks = mask_objs == obj_ids[:, None, None]

          boxes = torch.as_tensor(boxes, dtype=torch.float32)
          # there is only one class
          labels = torch.as_tensor(labels, dtype=torch.int64)
         # labels = torch.ones((num_objs,), dtype=torch.int64)
          masks = torch.as_tensor(obj_masks, dtype=torch.uint8)

          image_id = torch.tensor([idx])
          area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
          # suppose all instances are not crowd
          iscrowd = torch.zeros((obj_idx-1,), dtype=torch.int64)

          target = {}
          target["boxes"] = boxes
          target["labels"] = labels
          target["masks"] = masks
          target["image_id"] = image_id
          target["area"] = area
          target["iscrowd"] = iscrowd
          if self.transforms is not None:
            img, target = self.transforms(img, target)
          # print(target["image_id"])
          return img, target




class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

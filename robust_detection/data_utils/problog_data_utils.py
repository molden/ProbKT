from torch.utils.data import Dataset as TorchDataset
from typing import Callable, List, Iterable, Tuple
from deepproblog.dataset import Dataset
import numpy as np
from deepproblog.query import Query
from problog.logic import Term, list2term, Constant
import os
from robust_detection.utils import DATA_DIR
import torch
import pandas as pd
import copy
from robust_detection.models.rcnn_utils import WrapModel


def process_labels(path, shift=0):
    labels_df = pd.read_csv(path)
    labels = []
    for index, row in labels_df.iterrows():
        labels.append(int(row['label']) + shift)
    return labels


class Range_Counter(Dataset, TorchDataset):
    range_case = 0 #set it default to 0 but should be set with classmethod set_extra_vars
    def __getitem__(self, index: int) -> Tuple[list, list, list]:
        import ipdb
        ipdb.set_trace()
        return

    def __init__(
        self,
        data_path,
        fold,
        fold_type,
    ):
        super(Range_Counter, self).__init__()
        self.dataset_name = fold_type
        if fold_type == "val":
            fold_subtype = "train"
        else:
            fold_subtype = fold_type

        self.data_path = os.path.join(DATA_DIR, data_path)
        if fold_type == "test":
            self.idxs = np.arange(
                len(os.listdir(os.path.join(self.data_path, fold_subtype, "tensors"))))
        elif fold_type == "val":
            self.idxs = np.load(os.path.join(
                self.data_path, "folds", str(fold), f"val_idx.npy"))
        elif fold_type == "train":
            # import ipdb; ipdb.set_trace()
            self.idxs = np.load(os.path.join(
                self.data_path, "folds", str(fold), f"filtered_train_idx.npy"))
        else:
            raise("Invalid fold type")
        self.tensors_list = [torch.load(os.path.join(
            self.data_path, fold_subtype, "tensors", f"{idx}.pt")) for idx in self.idxs]
        self.removed_list = None
        if fold_type == "train":
            self.targets_list = [process_labels(os.path.join(
                self.data_path, fold_subtype, "filtered_labels", f"{idx}.txt"), shift=0) for idx in self.idxs]
            self.removed_list = [process_labels(os.path.join(
                self.data_path, fold_subtype, "removed_labels", f"{idx}.txt"), shift=0) for idx in self.idxs]
        else:
            self.targets_list = [process_labels(os.path.join(
                self.data_path, fold_subtype, "labels", f"{idx}.txt"), shift=0) for idx in self.idxs]

        self.tensors = torch.cat(self.tensors_list)
        shift = 0
        self.img_map = {}  # dictionary mapping img_idx to the digits idx
        for idx, tensor in enumerate(self.tensors_list):
            self.img_map[idx] = np.arange(tensor.shape[0]) + shift
            shift += tensor.shape[0]

    def to_query(self, i: int) -> Query:
        mnist_indices = self.img_map[i]
        if self.removed_list is not None:
            classes, class_counts, count_ranges = self._range_count_digits(
                self.targets_list[i], self.removed_list[i])
        else:
            classes, class_counts, count_ranges = self._range_count_digits(
                self.targets_list[i])
        # TODO : TAKING ONLY THE FIRST TENSORS TO HAVE AS MANY AS THE NUMBER OF DIGITS
        mnist_indices = mnist_indices[:len(self.targets_list[i])]

        # label_len = self.dataset.label_len_list[i]
        label_len = len(mnist_indices)
        # Build substitution dictionary for the arguments
        subs = dict()
       # var_names = [Constant(-1)]
        var_names = []
        for i in range(label_len):
            t = Term(f"p{i}")
            subs[t] = Term(
                "tensor",
                Term(
                    self.dataset_name,
                    Constant(mnist_indices[i]),
                ),
            )
            var_names.append(t)
        class_constant = []
        for c in classes:
            class_constant.append(Constant(c))
        count_constant = []
        for count_c in class_counts:
            count_constant.append(Constant(count_c))
        range_constant = []
        for range_c in count_ranges:
            range_constant.append(Constant(range_c))
        # Build query
        # import ipdb; ipdb.set_trace()
        query = Query(
            Term(
                "range_count_objects",
                *([list2term(var_names)]),
                # *(e for e in var_names),
                list2term(class_constant),
                list2term(count_constant),
                list2term(range_constant),
                # Constant(0),
                # Constant(0),
            ),
            subs,
        )
        # import ipdb; ipdb.set_trace()
        return query

    def __len__(self):
        return len(self.tensors_list)

    def _range_count_digits(self, ground_truth: list, removed=None) -> Tuple[list, list, list]:
        # classes        = [0,1,2,3,4,5,6,7,8,9]
        # count_classes  = [0,0,0,0,0,0,0,0,0,0]
        classes = []
        count_classes = []
        range_counts = []
        values, counts = np.unique(ground_truth, return_counts=True)
        if removed is not None:
            rem_val, rem_counts = np.unique(removed, return_counts=True)
        for value, count in zip(values, counts):
            # import ipdb; ipdb.set_trace()
            # count_classes[value-1] = count
            # count_classes.append(count)
            if removed is not None:
                if value in rem_val:
                    index_val = np.where(rem_val == value)
                    rem_count = int(rem_counts[index_val])
                    # import ipdb; ipdb.set_trace()
                else:
                    rem_count = 0
            else:
                rem_count = 0
            classes.append(value)
            if count + rem_count > self.range_case:
                count_classes.append(self.range_case-rem_count)
                range_counts.append(1)
            else:
                range_counts.append(0)
                count_classes.append(count)
        # import ipdb; ipdb.set_trace()
        return classes, count_classes, range_counts

    @classmethod
    def set_extra_vars(cls, args):
        #the cls requires to set range_case in args
        cls.range_case = args.range_case
        return

    @classmethod
    def filter_data(self, box_features, labels_df, boxes, classif, level=0.99):
        labels = []
        shift=1
        for index, row in labels_df.iterrows():
            labels.append(int(row['label']))
  
        assert(level > 0.5)
        og_labels = copy.deepcopy(labels)
        wrap_model = WrapModel(classif)
        # get the index and the values of the confident predictions
        confident_index, confident_preds = torch.where(
            wrap_model(torch.clone(box_features)) > level)
        # initialized the retained index with the full tensor
        retained_index = [i for i in range(box_features.shape[0])]
        removed_labels = []
        for i in range(len(confident_index)):
            np_labels = np.array(labels)
            np_removed_labels = np.array(removed_labels)
            if len(labels) <= 2: # at least two prediction left per image
               break
            else:
               count_labels = np.count_nonzero(np_labels == confident_preds[i].item())
               count_removed_labels = np.count_nonzero(np_removed_labels == confident_preds[i].item())
               if (count_labels + count_removed_labels) > self.range_case: #check if range_case applies for this label
                  #import ipdb; ipdb.set_trace()
                  if count_removed_labels < self.range_case: #in case range case applies for this label we can only remove if number of removed < range_case
                     if confident_preds[i].item() in labels:#also in range_case we know which labels are present only not the exact amount
                        retained_index.remove(confident_index[i])
                        labels.remove(confident_preds[i].item())
                        removed_labels.append(confident_preds[i].item())
               else:#range_case does not apply for this label
                  if confident_preds[i].item() in labels:#also in range_case we know which labels are present only not the exact amount
                     retained_index.remove(confident_index[i])
                     labels.remove(confident_preds[i].item())
                     removed_labels.append(confident_preds[i].item())
        if len(retained_index)>4 and level < 1.:
           return None, None, None
        df = pd.DataFrame(labels, columns = ["label"])
        df["xmin"] = 0
        df["ymin"] = 0
        df["xmax"] = 0
        df["ymax"] = 0
        df_del = pd.DataFrame(removed_labels, columns = ["label"])

        return box_features[retained_index], df, df_del

    @classmethod
    def compute_accuracy(self, targets, preds):
        return torch.Tensor([torch.equal(targets[i]["labels"].sort()[0],preds[i]["labels"].sort()[0]) for i in range(len(targets)) ]).mean()

    @classmethod
    def read_labelrow(self, row):
        return int(row['label'])

    @classmethod
    def get_dpl_script(self, data_path):
        return os.path.join(DATA_DIR, "..", "models", "range_count_clevr.pl")

    @classmethod
    def evaluate_classifier(self, preds, labels):
        return torch.equal(preds.sort()[0].long(), torch.Tensor(labels).sort()[0].long())

    @classmethod
    def select_data_to_label(self, box_features, labels, boxes, classif):
        wrap_model = WrapModel(classif)
        preds = torch.argmax(wrap_model(box_features), 1)
        values_preds, count_preds = torch.unique(preds, sorted = True, return_counts = True)
        values_labels, count_labels = torch.unique(torch.tensor(labels, dtype=torch.int64), sorted = True, return_counts = True)
        #import ipdb; ipdb.set_trace()
        if torch.equal(values_preds, values_labels):
            count_preds_clamped = torch.clamp(count_preds, max=self.range_case)
            count_labels_clamped = torch.clamp(count_labels, max=self.range_case)
           # import ipdb; ipdb.set_trace()
            if torch.equal(count_preds_clamped, count_labels_clamped):
                df = pd.DataFrame(preds, columns = ["label"])

                df["xmin"] = boxes[:,0].long()
                df["ymin"] = boxes[:,1].long()
                df["xmax"] = boxes[:,2].long()
                df["ymax"] = boxes[:,3].long()
                return df
            else:
                return None

class MNIST_Prod(Dataset, TorchDataset):
    """Dataset for fine-tuning the MNIST Prod case
    """

    def __getitem__(self, index: int) -> Tuple[list, list, list]:
        import ipdb
        ipdb.set_trace()

    def __init__(
        self,
        data_path,
        fold,
        fold_type,
    ):
        super(MNIST_Prod, self).__init__()

        self.dataset_name = fold_type

        if fold_type == "val":
            fold_subtype = "train"
        else:
            fold_subtype = fold_type

        self.data_path = os.path.join(DATA_DIR, data_path)
        if fold_type == "test":
            self.idxs = np.arange(
                len(os.listdir(os.path.join(self.data_path, fold_subtype, "tensors"))))
        elif fold_type == "val":
            self.idxs = np.load(os.path.join(
                self.data_path, "folds", str(fold), f"val_idx.npy"))
        elif fold_type == "train":
            self.idxs = np.load(os.path.join(
                self.data_path, "folds", str(fold), f"filtered_train_idx.npy"))
        else:
            raise("Invalid fold type")
        self.tensors_list = [torch.load(os.path.join(
            self.data_path, fold_subtype, "tensors", f"{idx}.pt")) for idx in self.idxs]

        if fold_type == "train":
            self.targets_list = [pd.read_csv(os.path.join(
                self.data_path, fold_subtype, "filtered_labels", f"{idx}.txt")) for idx in self.idxs]
        else:
            self.targets_list = [pd.read_csv(os.path.join(
                self.data_path, fold_subtype, "labels", f"{idx}.txt")) for idx in self.idxs]

        self.tensors = torch.cat(self.tensors_list)

        self.img_map = {}  # dictionary mapping img_idx to the digits idx
        shift = 0
        for idx, tensor in enumerate(self.tensors_list):
            self.img_map[idx] = np.arange(tensor.shape[0]) + shift
            shift += tensor.shape[0]

    def to_query(self, i: int) -> Query:
        """Generate queries"""

       # import ipdb; ipdb.set_trace()
        tensor_indices = self.img_map[i]
        prod_digit = self._prod_digits(self.targets_list[i])

        label_len = len(tensor_indices)

        # Build substitution dictionary for the arguments
        subs = dict()
        var_names = []
        for i in range(label_len):
            t = Term(f"p{i}")
            subs[t] = Term(
                "tensor",
                Term(
                    self.dataset_name,
                    Constant(tensor_indices[i]),
                ),
            )
            var_names.append(t)
        # Build query
        prod_constant = Constant(prod_digit)
        query = Query(
            Term(
                "prod_digits",
                *([list2term(var_names)]),
                prod_constant,
            ),
            subs,
        )
        #import ipdb; ipdb.set_trace()
        return query

    def __len__(self):
        return len(self.tensors_list)

    def _prod_digits(self, target) -> int:
       # import ipdb; ipdb.set_trace()
        return target['label'].values[0]

    @classmethod
    def set_extra_vars(cls, args):
        return

    @classmethod
    def filter_data(self, box_features, labels, boxes, classif, level=0.99):
        assert(level > 0.5)
        og_labels = copy.deepcopy(labels)
        wrap_model = WrapModel(classif)
        # get the index and the values of the confident predictions
        confident_index, confident_preds = torch.where(
            wrap_model(torch.clone(box_features)) > level)
        # initialized the retained index with the full tensor
        retained_index = [i for i in range(box_features.shape[0])]
        removed_labels = []
        label = labels.label.item()
        number_objects = 3 #Exactly 3 digits per image in this dataset
        for i in range(len(confident_index)):
            # only one element in the list = the product of the digits in the image.

            if (len(retained_index) >= 3) and (label != 0) and (confident_preds[i] != 0):
                if (label % confident_preds[i]) == 0:
                    retained_index.remove(confident_index[i])
                    label /= confident_preds[i].long().item()
                    number_objects -= 1

        new_label = pd.DataFrame()
        new_label["label"] = [label]
        
        # By assumption, there are only 3 digits in the image
        retained_index = retained_index[:number_objects]

        return box_features[retained_index], new_label, None
   
    @classmethod
    def compute_accuracy(self, targets, preds):
        #substract 1 to deal with label shift
        return torch.Tensor([torch.prod(targets[i]["labels"].sort()[0]-1) == torch.prod(preds[i]["labels"].sort()[0]-1) for i in range(len(targets)) ]).mean()

    @classmethod
    def read_labelrow(self, row):
        return int(row['label'])  

    @classmethod
    def get_dpl_script(self, data_path):
        return os.path.join(DATA_DIR, "..", "models", "multiply_digits.pl")

    @classmethod
    def evaluate_classifier(self, preds, labels):
        #returns multiplication accuracy
        return torch.prod(preds.sort()[0].long()) == torch.prod(torch.Tensor(labels).long().sort()[0])

    @classmethod
    def select_data_to_label(self, box_features, labels, boxes, classif):
        #exploit background information (exactly 3 digits on image)
        box_features = box_features[:3]
        boxes = boxes[:3]
        wrap_model = WrapModel(classif)
        preds = torch.argmax(wrap_model(box_features), 1)
        #import ipdb; ipdb.set_trace()
        if torch.prod(preds.sort()[0].long()) == torch.prod(torch.Tensor(labels).long().sort()[0]):
            df = pd.DataFrame(preds, columns=["label"])

            df["xmin"] = boxes[:, 0].long()
            df["ymin"] = boxes[:, 1].long()
            df["xmax"] = boxes[:, 2].long()
            df["ymax"] = boxes[:, 3].long()
            return df
        else:
            return None

class MNIST_Sum(Dataset, TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, list]:
        import ipdb
        ipdb.set_trace()

    def __init__(
        self,
        data_path,
        fold,
        fold_type,
    ):
        super(MNIST_Sum, self).__init__()

        self.dataset_name = fold_type

        if fold_type == "val":
            fold_subtype = "train"
        else:
            fold_subtype = fold_type

        self.data_path = os.path.join(DATA_DIR, data_path)
        if fold_type == "test":
            self.idxs = np.arange(
                len(os.listdir(os.path.join(self.data_path, fold_subtype, "tensors"))))
        elif fold_type == "val":
            self.idxs = np.load(os.path.join(
                self.data_path, "folds", str(fold), f"val_idx.npy"))
        elif fold_type == "train":
            self.idxs = np.load(os.path.join(
                self.data_path, "folds", str(fold), f"filtered_train_idx.npy"))
        else:
            raise("Invalid fold type")
        self.tensors_list = [torch.load(os.path.join(
            self.data_path, fold_subtype, "tensors", f"{idx}.pt")) for idx in self.idxs]

        if fold_type == "train":
            self.targets_list = [process_labels(os.path.join(
                self.data_path, fold_subtype, "filtered_labels", f"{idx}.txt"), shift=0) for idx in self.idxs]
        else:
            self.targets_list = [process_labels(os.path.join(
                self.data_path, fold_subtype, "labels", f"{idx}.txt"), shift=0) for idx in self.idxs]

        self.tensors = torch.cat(self.tensors_list)
        shift = 0
        self.img_map = {}  # dictionary mapping img_idx to the digits idx
        for idx, tensor in enumerate(self.tensors_list):
            self.img_map[idx] = np.arange(tensor.shape[0]) + shift
            shift += tensor.shape[0]

    def to_query(self, i: int) -> Query:
        """Generate queries"""
        tensor_indices = self.img_map[i]
        sum_digit = self._sum_digits(self.targets_list[i])
        # TODO : TAKING ONLY THE FIRST TENSORS TO HAVE AS MANY AS THE NUMBER OF DIGITS
        tensor_indices = tensor_indices[:len(self.targets_list[i])]

        label_len = len(tensor_indices)
        # Build substitution dictionary for the arguments
        subs = dict()
        # var_names = [Constant(-1)]
        var_names = []
        for i in range(label_len):
            t = Term(f"p{i}")
            subs[t] = Term(
                "tensor",
                Term(
                    self.dataset_name,
                    Constant(tensor_indices[i]),
                ),
            )
            var_names.append(t)
        # Build query
        sum_constant = Constant(sum_digit)
        query = Query(
            Term(
                "sum_digits",
                *([list2term(var_names)]),
                # *(e for e in var_names),
                sum_constant,
                # Constant(0),
                # Constant(0),
            ),
            subs,
        )
        return query

    def __len__(self):
        return len(self.tensors_list)

    def _sum_digits(self, ground_truth: list) -> int:
        sum_digit = 0
        for value in ground_truth:
            sum_digit += value
        return sum_digit

    @classmethod
    def set_extra_vars(cls, args):
        return

    @classmethod
    def compute_accuracy(self, targets, preds):
        return torch.Tensor([torch.sum(targets[i]["labels"].sort()[0]) == torch.sum(preds[i]["labels"].sort()[0]) for i in range(len(targets)) ]).mean()

    @classmethod
    def read_labelrow(self, row):
        return int(row['label'])

    @classmethod
    def filter_data(self, box_features, labels_df, boxes, classif, level=0.99):
        labels = []
        shift=1
        for index, row in labels_df.iterrows():
            labels.append(int(row['label']))

        assert(level > 0.5)
        og_labels = copy.deepcopy(labels)
        wrap_model = WrapModel(classif)
        # get the index and the values of the confident predictions
        confident_index, confident_preds = torch.where(
            wrap_model(torch.clone(box_features)) > level)
        #import ipdb; ipdb.set_trace()
        # initialized the retained index with the full tensor
        retained_index = [i for i in range(box_features.shape[0])]
        removed_labels = []
        #label = labels.label.item()
        number_objects = 3 #Exactly 3 digits per image in this dataset
        for i in range(len(confident_index)):
            # only one element in the list = the product of the digits in the image.
            if number_objects > 2:
               retained_index.remove(confident_index[i]) # remove this tensor from the data
               #import ipdb; ipdb.set_trace()
               try:
                   labels.remove(confident_preds[i].item()) 
               except ValueError:
                   import ipdb; ipdb.set_trace()
               number_objects -= 1

        
        # By assumption, there are only 3 digits in the image
        retained_index = retained_index[:number_objects]
        df = pd.DataFrame(labels, columns = ["label"])
        df["xmin"] = 0
        df["ymin"] = 0
        df["xmax"] = 0
        df["ymax"] = 0


        return box_features[retained_index], df, None

    @classmethod
    def get_dpl_script(self, data_path):
        return os.path.join(DATA_DIR, "..", "models", "sum_digits.pl")

    @classmethod
    def evaluate_classifier(self, preds, labels):
        #returns sum accuracy
        return torch.sum(preds.sort()[0].long()) == torch.sum(torch.Tensor(labels).long().sort()[0])

    @classmethod
    def select_data_to_label(self, box_features, labels, boxes, classif):
        #exploit background information (exactly 3 digits on image)
        box_features = box_features[:len(labels)]
        boxes = boxes[:len(labels)]

        wrap_model = WrapModel(classif)
        preds = torch.argmax(wrap_model(box_features), 1)
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


class Objects_Counter(Dataset, TorchDataset):

    def __getitem__(self, index: int) -> Tuple[list, list, list]:
        import ipdb
        ipdb.set_trace()
        data_item = self.tensors_list[index]
        classes, count_classes = self._count_digits(self.targets_list[index])
        img_digits = [tensor for tensor in data_item]
        return img_digits, classes, count_classes

    def __init__(
        self,
        data_path,
        fold,
        fold_type,
    ):
        super(Objects_Counter, self).__init__()

        self.dataset_name = fold_type

        if fold_type == "val":
            fold_subtype = "train"
        else:
            fold_subtype = fold_type

        self.data_path = os.path.join(DATA_DIR, data_path)
        if fold_type == "test":
            self.idxs = np.arange(
                len(os.listdir(os.path.join(self.data_path, fold_subtype, "tensors"))))
        elif fold_type == "val":
            self.idxs = np.load(os.path.join(
                self.data_path, "folds", str(fold), f"val_idx.npy"))
        elif fold_type == "train":
            self.idxs = np.load(os.path.join(
                self.data_path, "folds", str(fold), f"filtered_train_idx.npy"))
        else:
            raise("Invalid fold type")

        self.tensors_list = [torch.load(os.path.join(
            self.data_path, fold_subtype, "tensors", f"{idx}.pt")) for idx in self.idxs]

        if fold_type == "train":
            self.targets_list = [process_labels(os.path.join(
                self.data_path, fold_subtype, "filtered_labels", f"{idx}.txt"), shift=0) for idx in self.idxs]
        else:
            self.targets_list = [process_labels(os.path.join(
                self.data_path, fold_subtype, "labels", f"{idx}.txt"), shift=0) for idx in self.idxs]

        self.tensors = torch.cat(self.tensors_list)

        shift = 0
        self.img_map = {}  # dictionary mapping img_idx to the digits idx
        for idx, tensor in enumerate(self.tensors_list):
            self.img_map[idx] = np.arange(tensor.shape[0]) + shift
            shift += tensor.shape[0]

    def to_query(self, i: int) -> Query:
        """Generate queries"""
        # import ipdb; ipdb.set_trace()
        tensor_indices = self.img_map[i]
        classes, count_classes = self._count_objects(self.targets_list[i])
        # TODO : TAKING ONLY THE FIRST TENSORS TO HAVE AS MANY AS THE NUMBER OF DIGITS
        tensor_indices = tensor_indices[:len(self.targets_list[i])]

        # label_len = self.dataset.label_len_list[i]
        label_len = len(tensor_indices)
        # Build substitution dictionary for the arguments
        subs = dict()
       # var_names = [Constant(-1)]
        var_names = []
        for i in range(label_len):
            t = Term(f"p{i}")
            subs[t] = Term(
                "tensor",
                Term(
                    self.dataset_name,
                    Constant(tensor_indices[i]),
                ),
            )
            var_names.append(t)
        class_constant = []
        for c in classes:
            class_constant.append(Constant(c))
        count_constant = []
        for count_c in count_classes:
            count_constant.append(Constant(count_c))
        # Build query
        # import ipdb; ipdb.set_trace()
        query = Query(
            Term(
                "count_objects",
                *([list2term(var_names)]),
                # *(e for e in var_names),
                list2term(class_constant),
                list2term(count_constant),
                # Constant(0),
                # Constant(0),
            ),
            subs,
        )
       # import ipdb; ipdb.set_trace()
        return query

    def __len__(self):
        return len(self.tensors_list)

    def _count_objects(self, ground_truth: list) -> Tuple[list, list]:
        # classes        = [0,1,2,3,4,5,6,7,8,9]
        # count_classes  = [0,0,0,0,0,0,0,0,0,0]
        classes = []
        count_classes = []
        values, counts = np.unique(ground_truth, return_counts=True)
        for value, count in zip(values, counts):
            # import ipdb; ipdb.set_trace()
            # count_classes[value-1] = count
            count_classes.append(count)
            classes.append(value)
        # import ipdb; ipdb.set_trace()
        return classes, count_classes

    @classmethod
    def set_extra_vars(cls, args):
        return

    @classmethod  
    def compute_accuracy(self, targets, preds):
        return torch.Tensor([torch.equal(targets[i]["labels"].sort()[0],preds[i]["labels"].sort()[0]) for i in range(len(targets)) ]).mean()

    @classmethod 
    def read_labelrow(self, row):
        return int(row['label'])

    @classmethod
    def filter_data(self, box_features, labels_df, boxes, classif, level=0.99):
        labels = []
        shift=1
        for index, row in labels_df.iterrows():
            labels.append(int(row['label']))

        assert(level > 0.5)
        og_labels = copy.deepcopy(labels)
        wrap_model = WrapModel(classif)
        # get the index and the values of the confident predictions
        confident_index, confident_preds = torch.where(
            wrap_model(torch.clone(box_features)) > level)
        #import ipdb; ipdb.set_trace()
        # initialized the retained index with the full tensor
        retained_index = [i for i in range(box_features.shape[0])]
        removed_labels = []
        #label = labels.label.item()
        for i in range(len(confident_index)):
            # only one element in the list = the product of the digits in the image.
            if len(labels) <= 2: # at least two prediction left per image
               break
            else:
               if confident_preds[i].item() in labels:
                  retained_index.remove(confident_index[i]) # remove this tensor from the data
                  labels.remove(confident_preds[i].item())
        if len(retained_index)>len(labels):
           retained_index = retained_index[:len(labels)]

        if len(retained_index)>4 and level < 1.:
           return None, None, None
        if len(retained_index) < len(labels):
           return None, None, None
        df = pd.DataFrame(labels, columns = ["label"])
        df["xmin"] = 0
        df["ymin"] = 0
        df["xmax"] = 0
        df["ymax"] = 0

        return box_features[retained_index], df, None

    @classmethod
    def get_dpl_script(self, data_path):
        if "molecules" in data_path:
           return os.path.join(DATA_DIR,"..","models","count_molecules.pl")
        elif "clevr" in data_path:
           return os.path.join(DATA_DIR, "..", "models", "count_clevr.pl")
        else:
           return os.path.join(DATA_DIR, "..", "models", "count_objects.pl")

    @classmethod
    def evaluate_classifier(self, preds, labels):
        return torch.equal(preds.sort()[0].long(), torch.Tensor(labels).sort()[0].long())

    @classmethod
    def select_data_to_label(self, box_features, labels, boxes, classif):
        #exploit background information (exactly 3 digits on image)
        box_features = box_features[:len(labels)]
        boxes = boxes[:len(labels)]

        wrap_model = WrapModel(classif)
        preds = torch.argmax(wrap_model(box_features), 1)
        #import ipdb; ipdb.set_trace()
        if torch.equal(preds.sort()[0].long(), torch.Tensor(labels).long().sort()[0]):
            df = pd.DataFrame(preds, columns=["label"])

            df["xmin"] = boxes[:, 0].long()
            df["ymin"] = boxes[:, 1].long()
            df["xmax"] = boxes[:, 2].long()
            df["ymax"] = boxes[:, 3].long()
            return df
        else:
            return None


class MNIST_Images(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        return self.dataset.tensors[int(item[0])]


if __name__ == "__main__":
    dataset = MNIST_Counter(data_path="mnist/alldigits_5",
                            fold=0, fold_type="train")
    import ipdb
    ipdb.set_trace()

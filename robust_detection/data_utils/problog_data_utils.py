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

def process_labels(path, shift = 0):
    labels_df = pd.read_csv(path)
    labels = []
    for index, row in labels_df.iterrows():
        labels.append(int(row['label'])+ shift)
    return labels

class MNIST_Sum(Dataset, TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, list]:
        import ipdb; ipdb.set_trace()
        data_item = self.tensors_list[index]
        sum_digit = self._sum_digits(self.targets_list[index])
        img_digits = [tensor for tensor in data_item]
        return img_digits, sum_digit
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

        self.data_path = os.path.join(DATA_DIR,data_path)
        if fold_type=="test":
            self.idxs = np.arange(len(os.listdir(os.path.join(self.data_path,fold_subtype,"tensors"))))
        elif fold_type == "val":
            self.idxs = np.load(os.path.join(self.data_path,"folds",str(fold),f"val_idx.npy"))
        elif fold_type == "train":
            self.idxs = np.load(os.path.join(self.data_path,"folds",str(fold),f"filtered_train_idx.npy"))
        else:
            raise("Invalid fold type")
        self.tensors_list = [torch.load(os.path.join(self.data_path,fold_subtype,"tensors",f"{idx}.pt")) for idx in self.idxs]

        if fold_type == "train":
            self.targets_list = [process_labels(os.path.join(self.data_path,fold_subtype,"filtered_labels",f"{idx}.txt"), shift = 0) for idx in self.idxs]
        else:
            self.targets_list = [process_labels(os.path.join(self.data_path,fold_subtype,"labels",f"{idx}.txt"), shift = 0) for idx in self.idxs]

        self.tensors = torch.cat(self.tensors_list)
        shift = 0
        self.img_map = {} # dictionary mapping img_idx to the digits idx
        for idx,tensor in enumerate(self.tensors_list):
            self.img_map[idx] = np.arange(tensor.shape[0]) + shift
            shift += tensor.shape[0]
    
    def to_query(self, i: int) -> Query:
        """Generate queries"""
        #import ipdb; ipdb.set_trace()
        mnist_indices = self.img_map[i]
        sum_digit = self._sum_digits(self.targets_list[i])
        # TODO : TAKING ONLY THE FIRST TENSORS TO HAVE AS MANY AS THE NUMBER OF DIGITS
        mnist_indices= mnist_indices[:len(self.targets_list[i])]

        #label_len = self.dataset.label_len_list[i]
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
        # Build query
        sum_constant = Constant(sum_digit)
        #import ipdb; ipdb.set_trace()
        query = Query(
            Term(
                "sum_digits",
                *([list2term(var_names)]),
               # *(e for e in var_names),
                sum_constant,
                #Constant(0),
                #Constant(0),
            ),
            subs,
        )
        # import ipdb; ipdb.set_trace()
        return query

    def __len__(self):
        return len(self.tensors_list)

    def _sum_digits(self, ground_truth: list) -> int:
        sum_digit = 0
        for value in ground_truth:
            sum_digit += value
        return sum_digit

class MNIST_Counter(Dataset, TorchDataset):

    def __getitem__(self, index: int) -> Tuple[list, list, list]:
        import ipdb; ipdb.set_trace()
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
        super(MNIST_Counter, self).__init__()

        self.dataset_name = fold_type

        if fold_type == "val":
            fold_subtype = "train"
        else:
            fold_subtype = fold_type

        self.data_path = os.path.join(DATA_DIR,data_path)
        if fold_type=="test":
            self.idxs = np.arange(len(os.listdir(os.path.join(self.data_path,fold_subtype,"tensors"))))
        elif fold_type == "val":
            self.idxs = np.load(os.path.join(self.data_path,"folds",str(fold),f"val_idx.npy"))
        elif fold_type == "train":
            self.idxs = np.load(os.path.join(self.data_path,"folds",str(fold),f"filtered_train_idx.npy"))
        else:
            raise("Invalid fold type")

        self.tensors_list = [torch.load(os.path.join(self.data_path,fold_subtype,"tensors",f"{idx}.pt")) for idx in self.idxs]
        
        if fold_type == "train":
            self.targets_list = [process_labels(os.path.join(self.data_path,fold_subtype,"filtered_labels",f"{idx}.txt"), shift = 0) for idx in self.idxs]
        else:
            self.targets_list = [process_labels(os.path.join(self.data_path,fold_subtype,"labels",f"{idx}.txt"), shift = 0) for idx in self.idxs]

        self.tensors = torch.cat(self.tensors_list)
        
        shift = 0
        self.img_map = {} # dictionary mapping img_idx to the digits idx
        for idx,tensor in enumerate(self.tensors_list):
            self.img_map[idx] = np.arange(tensor.shape[0]) + shift
            shift += tensor.shape[0]
        
    def to_query(self, i: int) -> Query:
        """Generate queries"""
        #import ipdb; ipdb.set_trace()
        mnist_indices = self.img_map[i]
        classes, count_classes = self._count_digits(self.targets_list[i])
        # TODO : TAKING ONLY THE FIRST TENSORS TO HAVE AS MANY AS THE NUMBER OF DIGITS
        mnist_indices= mnist_indices[:len(self.targets_list[i])]

        #label_len = self.dataset.label_len_list[i]
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
        for count_c in count_classes:
            count_constant.append(Constant(count_c))
        # Build query
        #import ipdb; ipdb.set_trace()
        query = Query(
            Term(
                "count_objects",
                *([list2term(var_names)]),
               # *(e for e in var_names),
                list2term(class_constant),
                list2term(count_constant),
                #Constant(0),
                #Constant(0),
            ),
            subs,
        )
       # import ipdb; ipdb.set_trace()
        return query

    def __len__(self):
        return len(self.tensors_list)

    def _count_digits(self, ground_truth: list) -> Tuple[list,list]:
        #classes        = [0,1,2,3,4,5,6,7,8,9]
        #count_classes  = [0,0,0,0,0,0,0,0,0,0]
        classes        = []
        count_classes  = []
        values, counts = np.unique(ground_truth, return_counts=True)
        for value,count in zip(values,counts):
           # import ipdb; ipdb.set_trace()
            #count_classes[value-1] = count
            count_classes.append(count)
            classes.append(value)
        #import ipdb; ipdb.set_trace()
        return classes, count_classes


class MNIST_Images(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        return self.dataset.tensors[int(item[0])]


if __name__ == "__main__":
    dataset = MNIST_Counter(data_path = "mnist/alldigits_5",fold = 0, fold_type = "train")
    import ipdb; ipdb.set_trace()

import datetime
import errno
import os
import time
import torch
from torch.utils.data import *
from PIL import Image
from scipy import ndimage, misc
import torchvision.transforms as T
import pandas as pd
import numpy as np
import torchvision
import torch.nn.functional as F
from torch import nn, Tensor
import pytorch_lightning as pl

from robust_detection.utils import str2bool
from robust_detection.utils import DATA_DIR

from rdkit import Chem

class SMILES_CountAtoms_Dataset(Dataset):

    def __init__(self, data_dir, num_classes = 5):
        self.smiles_file = f"{data_dir}/smiles"
        self.smiles_df   = pd.read_csv(self.smiles_file)
        self.len_df      = self.smiles_df.shape[0]
        self.imagefolder = f"{data_dir}/images/"

    def __len__(self):
        return self.len_df

    def __getitem__(self, idx):
        #self.dict_atom = {'C': 1, 'H': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'Se': 10, 'P': 11, 'B': 12, 'Si': 13}
        atomnumber_dict = {6: 0, 1: 1, 7: 2, 8: 3, 16: 4, 9: 5, 17: 6, 35: 7, 53: 8, 34: 9, 15: 10, 5: 11}
        count_atoms = np.zeros(self.num_classes)
        df_row = self.smiles_df.iloc[idx]
        imagename=self.imagefolder+"/"+str(idx)+'.png'
        #import ipdb; ipdb.set_trace()
        smiles   = df_row['smiles'] 
        img      = Image.open(f"{imagename}").convert("L")
        #img      = Ftrans.to_tensor(img)
        m        = Chem.MolFromSmiles(smiles)
        for atom in m.GetAtoms():
            count_atoms[atomnumber_dict[atom.GetAtomicNum()]] +=1
            #print(atom.GetAtomicNum())
        #values, counts = np.unique(count_atoms, return_counts=True)
        return img, count_atoms


def collate_with_boxes(batch):
    """
    X : images
    counts : counts of objects in the images
    aggs : aggregated statistic for each image
    targets : boxes
    og_labels : True if the sample comes from the og data set (not the target data set)
    """
    X = torch.stack([b[0] for b in batch])
    counts = torch.stack([torch.Tensor(b[1]) for b in batch])
    aggs = torch.stack([torch.Tensor(b[2]) for b in batch])
    targets = [b[3] for b in batch]
    og_labels = torch.stack([torch.Tensor([b[4]]) for b in batch])
    return X, counts, aggs, targets, og_labels
    
class Objects_Count_Dataset(Dataset):

    def __init__(self, data_dir, num_classes = 10, return_boxes = False, og_label = False):
        self.data_dir = os.path.join(DATA_DIR,data_dir)
        self.transforms = T.ToTensor()
        self.num_classes = num_classes

        self.return_boxes = return_boxes
        self.og_label = og_label


    def __len__(self):
        return len([name for name in os.listdir(f"{self.data_dir}/images/")])

    def __getitem__(self, idx):
        imagename=f"{self.data_dir}/images/{str(idx)}.png"
        img = Image.open(imagename).convert("L")
        labels_df = pd.read_csv(f"{self.data_dir}/labels/{str(idx)}.txt")
        digits = labels_df.groupby("label").size().index.values
        counts = labels_df.groupby("label").size().values
        
        counts_vec = np.zeros(self.num_classes)
        counts_vec[digits] = counts

        agg_vec = np.array([(digits * counts).sum()])

        if self.transforms is not None:
            img = self.transforms(img)
            # print(target["image_id"])
        
        if not self.return_boxes:
            return img, counts_vec, agg_vec, None, self.og_label
        else:
            labels = []
            boxes = []
            obj_idx = 1
            for index, row in labels_df.iterrows():
                labels.append(int(row['label']))
                xmin = int(row['xmin'])
                xmax = int(row['xmax'])
                ymin = int(row['ymin'])
                ymax = int(row['ymax'])
                boxes.append([xmin, ymin, xmax, ymax])
                start = (xmin,ymin)
                end = (xmax,ymax)



            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.as_tensor(labels, dtype=torch.int64)
          
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            
            return img, counts_vec, agg_vec, target, self.og_label


class ObjectsCountDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, seed, num_classes, data_dir, og_data_dir, fold, num_workers = 4, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True
    
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.og_data_dir = og_data_dir
        self.fold = fold


    def prepare_data(self):

        dataset = Objects_Count_Dataset(data_dir = self.data_dir + "/train", num_classes = self.num_classes , return_boxes = True, og_label = False)
        self.test = Objects_Count_Dataset(data_dir = self.data_dir + "/test", num_classes = self.num_classes, return_boxes = True, og_label = False) 
        self.test_ood = Objects_Count_Dataset(data_dir = self.data_dir + "/test_ood", num_classes = self.num_classes, return_boxes = True, og_label = False) 
        
        train_idx = np.load(os.path.join(DATA_DIR,self.data_dir,"folds",str(self.fold),"train_idx.npy"))
        val_idx = np.load(os.path.join(DATA_DIR,self.data_dir,"folds",str(self.fold),"val_idx.npy"))
        
        if self.og_data_dir is not None:
            train_ds = Subset(dataset,train_idx)

            ds_og =  Objects_Count_Dataset(data_dir = self.og_data_dir + "/train", num_classes = self.num_classes, og_label = True)
            train_idx_og = np.load(os.path.join(DATA_DIR,self.og_data_dir,"folds",str(self.fold),"train_idx.npy"))
            og_train_ds = Subset(ds_og,train_idx_og)

            self.train = torch.utils.data.ConcatDataset([train_ds,og_train_ds])
        else:
            self.train = Subset(dataset,train_idx)
        
        self.val = Subset(dataset,val_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn = collate_with_boxes
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn = collate_with_boxes
        )

    def test_dataloader(self, shuffle = False):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn = collate_with_boxes
            )

    def test_ood_dataloader(self, shuffle = False):
        return DataLoader(
            self.test_ood,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn = collate_with_boxes
            )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--num_classes', type=int, default=10)
        parser.add_argument('--data_dir', type=str, default="data_skip9")
        parser.add_argument('--og_data_dir', type=str, default=None)
        parser.add_argument('--paired_og', type=str2bool, default=False)
        return parser


if __name__ =="__main__":
    dataset = Objects_Count_Dataset(data_dir = "mnist/alldigits/train")
    dataset[10]
    import ipdb; ipdb.set_trace()

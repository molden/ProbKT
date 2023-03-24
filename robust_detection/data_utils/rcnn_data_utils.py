from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from PIL import Image
from scipy import ndimage, misc
import skimage.draw as draw
import robust_detection.transforms as T
import robust_detection.data_utils.detr_transforms as detr_T
import numpy as np
import pandas as pd
import torch
import os
from robust_detection.utils import str2bool

from robust_detection.utils import DATA_DIR

import torchvision.datasets as dset
import torchvision.transforms as transforms



def collate_tuple(batch):
    return tuple(zip(*batch))


def get_transform(normalize=False):
    transforms = []
    transforms.append(T.ToTensor())
    if normalize:
        transf = T.Compose([
            T.ToTensor(),
            detr_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transf
    else:
        return T.Compose(transforms)


class Objects_RCNN_Predictor(pl.LightningDataModule):
    def __init__(self, batch_size, data_path, fold, filtered=False, num_workers=4, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.fold = fold
        self.filtered = filtered

    def prepare_data(self):
        dataset = Objects_Predictor_Dataset(
            os.path.join(DATA_DIR, self.data_path, "train"))
        dataset_test = Objects_Predictor_Dataset(
            os.path.join(DATA_DIR, self.data_path, "test"))
        dataset_ood = Objects_Predictor_Dataset(
            os.path.join(DATA_DIR, self.data_path, "test_ood"))

        train_idx = np.load(os.path.join(
            DATA_DIR, f"{self.data_path}", "folds", str(self.fold), "train_idx.npy"))
        if self.filtered:
            #           import ipdb; ipdb.set_trace()
            train_idx = np.load(os.path.join(DATA_DIR, f"{self.data_path}", "folds", str(
                self.fold), f"filtered_train_idx.npy"))
        self.train = Subset(dataset, train_idx)

        val_idx = np.load(os.path.join(
            DATA_DIR, f"{self.data_path}", "folds", str(self.fold), "val_idx.npy"))

        self.val = Subset(dataset, val_idx)
        self.test = dataset_test
        self.test_ood = dataset_ood

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def test_ood_dataloader(self, shuffle=False):
        return DataLoader(
            self.test_ood,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--batch_size', type=int, default=15)
        parser.add_argument('--fold', type=int, default=0)
        parser.add_argument('--data_path', type=str,
                            default="mnist/alldigits/")
        parser.add_argument('--filtered', type=str2bool, default=False)
        parser.add_argument('--num_workers', type=int, default=4)
        return parser


class Objects_RCNN(pl.LightningDataModule):
    def __init__(self, batch_size, data_path, fold, num_workers=4, filtered=False, box_loss_mask=False, skip_data_path=None, rgb=False, re_train=False, og_data_dir=None, og_data_path=None, in_memory=False, target_data_cls=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.skip_data_path = skip_data_path
        self.og_data_dir = og_data_dir
        self.box_loss_mask = box_loss_mask
        self.og_data_path = og_data_path
        if rgb:
            normalize = True
        else:
            normalize = False
        self.transforms = get_transform(normalize)
        self.fold = fold
        self.re_train = re_train
        self.in_memory = in_memory
        self.rgb = rgb
        self.filtered = filtered
        self.target_data_cls = target_data_cls
        self.base_class = Objects_Detection_Dataset

    def prepare_data(self):
        dataset = self.base_class(os.path.join(DATA_DIR, self.data_path, "train"),
                                            self.transforms, box_loss_mask=self.box_loss_mask, rgb=self.rgb, in_memory=self.in_memory)
        dataset_test = self.base_class(os.path.join(
            DATA_DIR, self.data_path, "test"), self.transforms, rgb=self.rgb, in_memory=False)
        dataset_ood = self.base_class(os.path.join(
            DATA_DIR, self.data_path, "test_ood"), self.transforms, rgb=self.rgb, in_memory=False)

        if self.og_data_dir is not None:
            dataset_train1 = self.base_class(os.path.join(
                DATA_DIR, self.data_path, "train"), self.transforms, box_loss_mask=True, rgb=self.rgb, in_memory=self.in_memory)
            train_idx1 = np.load(os.path.join(
                DATA_DIR, f"{self.data_path}", "folds", str(self.fold), "train_idx.npy"))
            train1 = Subset(dataset_train1, train_idx1)

            dataset_train2 = self.base_class(os.path.join(
                DATA_DIR, self.og_data_dir, "train"), self.transforms, box_loss_mask=False, rgb=self.rgb, in_memory=self.in_memory)
            train_idx2 = np.load(os.path.join(
                DATA_DIR, f"{self.og_data_dir}", "folds", str(self.fold), "train_idx.npy"))
            train2 = Subset(dataset_train2, train_idx2)
            self.train = torch.utils.data.ConcatDataset([train1, train2])
        elif self.re_train:
            dataset_train = self.base_class(os.path.join(
                DATA_DIR, self.data_path, "train"), self.transforms, rgb=self.rgb, re_train=True, in_memory=self.in_memory)
            re_train_idx = np.load(os.path.join(
                DATA_DIR, f"{self.data_path}", "folds", str(self.fold), "re_train_idx.npy"))
            re_train_ds = Subset(dataset_train, re_train_idx)

            # This is for training on both og and relabeled dataset
            dataset_og = self.base_class(os.path.join(
                DATA_DIR, self.og_data_path, "train"), self.transforms, rgb=self.rgb, in_memory=self.in_memory)
            train_idx = np.load(os.path.join(
                DATA_DIR, f"{self.og_data_path}", "folds", str(self.fold), "train_idx.npy"))
            og_train_ds = Subset(dataset_og, train_idx)
            self.train = torch.utils.data.ConcatDataset(
                [re_train_ds, og_train_ds])
            #self.train = re_train_ds
        elif self.skip_data_path is not None:
            dataset_train1 = self.base_class(os.path.join(
                DATA_DIR, self.data_path, "train"), self.transforms, box_loss_mask=True, rgb=self.rgb, in_memory=self.in_memory)
            train_idx1 = np.load(os.path.join(
                DATA_DIR, f"{self.data_path}", "folds", str(self.fold), "train_idx.npy"))
            train1 = Subset(dataset_train1, train_idx1)

            dataset_train2 = self.base_class(os.path.join(
                DATA_DIR, self.skip_data_path, "train"), self.transforms, box_loss_mask=False, rgb=self.rgb, in_memory=self.in_memory)
            train_idx2 = np.load(os.path.join(
                DATA_DIR, f"{self.skip_data_path}", "folds", str(self.fold), "train_idx.npy"))
            train2 = Subset(dataset_train2, train_idx2)
            self.train = torch.utils.data.ConcatDataset([train1, train2])
            # import ipdb; ipdb.set_trace()

        else:
            train_idx = np.load(os.path.join(
                DATA_DIR, f"{self.data_path}", "folds", str(self.fold), "train_idx.npy"))
            if self.filtered:
                import ipdb
                ipdb.set_trace()
                train_idx = np.load(os.path.join(DATA_DIR, f"{self.data_path}", "folds", str(
                    self.fold), f"filtered_train_idx.npy"))
            self.train = Subset(dataset, train_idx)
        
        val_idx = np.load(os.path.join(
            DATA_DIR, f"{self.data_path}", "folds", str(self.fold), "val_idx.npy"))
        if self.target_data_cls is not None:
           dataset = self.base_class(os.path.join(DATA_DIR, self.data_path, "train"),
                                      self.transforms, box_loss_mask=self.box_loss_mask, rgb=self.rgb, in_memory=self.in_memory, target_data_cls=self.target_data_cls)
        self.val = Subset(dataset, val_idx)
        self.test = dataset_test
        self.test_ood = dataset_ood

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def test_ood_dataloader(self, shuffle=False):
        return DataLoader(
            self.test_ood,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--batch_size', type=int, default=15)
        parser.add_argument('--fold', type=int, default=0)
        parser.add_argument('--data_path', type=str,
                            default="mnist/alldigits/")
        parser.add_argument('--filtered', type=str2bool, default=False)
        parser.add_argument('--box_loss_mask', type=str2bool, default=False)
        parser.add_argument('--skip_data_path', type=str, default=None)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--re_train', type=str2bool, default=False)
        parser.add_argument('--in_memory', type=str2bool, default=False)
        parser.add_argument('--rgb', type=str2bool, default=False)
        parser.add_argument('--og_data_dir', type=str, default=None)
        parser.add_argument('--paired_og', type=str2bool, default=False)
        return parser


class Objects_RCNN_Pred(Objects_RCNN):
    """_summary_

    Same as Objects_RCNN but without processing of the labels, we just return the tensor version of the dataframe.
    """
    def __init__(self, batch_size, data_path, fold, num_workers=4, filtered=False, box_loss_mask=False, skip_data_path=None, rgb=False, re_train=False, og_data_dir=None, og_data_path=None, in_memory=False, target_data_cls=None, **kwargs):
        super().__init__(batch_size, data_path, fold, num_workers, filtered, box_loss_mask, skip_data_path, rgb, re_train, og_data_dir, og_data_path, in_memory, **kwargs)
        self.base_class = Objects_Detection_Predictor_Dataset
    

class Objects_Predictor_Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if "molecules" in data_dir:
            self.label_shift = 1
        # shifting the label index by 1 in the mnist case (0 is background in RCNN)
        elif "mnist" in data_dir:
            self.label_shift = 1
        # shifting the label index by 1 in the mnist case (0 is background in RCNN)
        elif "clevr" in data_dir:
            self.label_shift = 1

    def __len__(self):
        return len([name for name in os.listdir(f"{self.data_dir}/tensors/")])

    def __getitem__(self, idx):
        tensor_file = f"{self.data_dir}/tensors/{str(idx)}.pt"
        tensor_item = torch.load(tensor_file)
        labels_df = pd.read_csv(f"{self.data_dir}/labels/{str(idx)}.txt")
        labels = []
        boxes = []
        for index, row in labels_df.iterrows():
            labels.append(int(row['label'])+self.label_shift)
            boxes.append([0, 0, 0, 0])
        target = {}
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target["boxes"] = boxes
        target["labels"] = labels
        return tensor_item, target, idx




class Objects_Detection_Dataset(Dataset):

    def __init__(self, data_dir, transforms, box_loss_mask=False, rgb=False, re_train=False, in_memory=False, target_data_cls=None):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.re_train = re_train
        self.box_loss_mask = box_loss_mask

        self.rgb = rgb
        if in_memory:  # images are stored in memory rather than having to IO. Currently not much speed up observed there
            self.images_dict = {f: Image.open(os.path.join(self.data_dir, "images", f)).convert(
                "L") for f in os.listdir(f"{self.data_dir}/images/")}
            if self.re_train:
                self.labels_dict = {f: pd.read_csv(
                    f"{self.data_dir}/re_labels/{f}") for f in os.listdir(f"{self.data_dir}/re_labels/")}
            else:
                self.labels_dict = {f: pd.read_csv(
                    f"{self.data_dir}/labels/{f}") for f in os.listdir(f"{self.data_dir}/labels/")}

        self.in_memory = in_memory
        self.target_data_cls = target_data_cls

        if "molecules" in data_dir:
            self.label_shift = 1
        # shifting the label index by 1 in the mnist case (0 is background in RCNN)
        elif "mnist" in data_dir:
            self.label_shift = 1
        # shifting the label index by 1 in the mnist case (0 is background in RCNN)
        elif "clevr" in data_dir:
            self.label_shift = 1

    def __len__(self):
        return len([name for name in os.listdir(f"{self.data_dir}/images/")])

    def __getitem__(self, idx):
        imagename = f"{self.data_dir}/images/{str(idx)}.png"
        if self.in_memory:
            img = self.images_dict[f"{str(idx)}.png"]
            labels_df = self.labels_dict[f"{str(idx)}.txt"]
        else:
            img = Image.open(imagename).convert("L")
            if self.rgb:
                img = img.convert('RGB')
            if self.re_train:
                labels_df = pd.read_csv(
                    f"{self.data_dir}/re_labels/{str(idx)}.txt")
            else:
                labels_df = pd.read_csv(
                    f"{self.data_dir}/labels/{str(idx)}.txt")

        labels = []
        boxes = []
        mask_objs = np.zeros(img.size)
        w, h = img.size
        obj_idx = 1
        for index, row in labels_df.iterrows():
            ###TODO this needs to be a target_data_cls method if this is set
            if self.target_data_cls is None:
                labels.append(int(row['label'])+self.label_shift)
                xmin = int(row['xmin'])
                xmax = int(row['xmax'])
                ymin = int(row['ymin'])
                ymax = int(row['ymax'])
            else:
                labels.append(self.target_data_cls.read_labelrow(row)+self.label_shift)
                xmin = 0
                xmax = 0
                ymin = 0
                ymax = 0
            if self.box_loss_mask:
                boxes.append([0, 0, 0, 0])
            else:
                boxes.append([xmin, ymin, xmax, ymax])
            start = (xmin, ymin)
            end = (xmax, ymax)
            #import ipdb; ipdb.set_trace()
            rr, cc = draw.rectangle(start, end=end, shape=img.size)
            mask_objs[rr, cc] = obj_idx
            obj_idx += 1

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
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        if self.box_loss_mask:
            target["box_loss_mask"] = 1
        else:
            target["box_loss_mask"] = 0
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # print(target["image_id"])
        return img, target, idx


class Objects_Detection_Predictor_Dataset(Objects_Detection_Dataset):
    def __init__(self, data_dir, transforms, box_loss_mask=False, rgb=False, re_train=False, in_memory=False): 
        super().__init__(data_dir, transforms, box_loss_mask, rgb, re_train, in_memory)

    def __getitem__(self, idx):
        imagename = f"{self.data_dir}/images/{str(idx)}.png"
        if self.in_memory:
            img = self.images_dict[f"{str(idx)}.png"]
            labels_df = self.labels_dict[f"{str(idx)}.txt"]
        else:
            img = Image.open(imagename).convert("L")
            if self.rgb:
                img = img.convert('RGB')
            if self.re_train:
                labels_df = pd.read_csv(
                    f"{self.data_dir}/re_labels/{str(idx)}.txt")
            else:
                labels_df = pd.read_csv(
                    f"{self.data_dir}/labels/{str(idx)}.txt")

        labels = []
        boxes = []
        mask_objs = np.zeros(img.size)
        w, h = img.size
        obj_idx = 1

        labels = torch.Tensor(labels_df.values)

        target = {}
        target["labels"] = labels
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target, idx

class CocoDetection(dset.VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform=None,
        target_transform=None,
        transforms=None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.len = 10000000

    def _load_image(self, id: int):
        #path = self.coco.loadImgs(id)[0]["file_name"]
        path = "".join(["0"] * (12 - len(str(id)))) + str(id) + ".jpg"
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def process_target(self, target):
        boxes = []
        labels = []
        iscrowd = []
        target_dict = {}
        for i in range(len(target)):
            if target[i]["category_id"] == 17:  # cat
                target[i]["category_id"] = 2
            if target[i]["category_id"] == 18:  # dog
                target[i]["category_id"] = 3
            if target[i]["category_id"] > 3:  # other
                target[i]["category_id"] = 4
            boxes.append(torch.Tensor(target[i]["bbox"]))
            labels.append(torch.LongTensor([target[i]["category_id"]]))
            iscrowd.append(torch.LongTensor([target[i]["iscrowd"]]))
        # bbox is (x,y,width,height)
        if len(target) == 0:
            import ipdb
            ipdb.set_trace()
        target_dict["boxes"] = torch.stack(boxes, 0)
        target_dict["boxes"][:, 2] = target_dict["boxes"][:, 0] + \
            target_dict["boxes"][:, 2]
        target_dict["boxes"][:, 3] = target_dict["boxes"][:, 1] + \
            target_dict["boxes"][:, 3]
        # labels
        target_dict["labels"] = torch.cat(labels)
        target_dict["area"] = (target_dict["boxes"][:, 3] - target_dict["boxes"]
                               [:, 1]) * (target_dict["boxes"][:, 2] - target_dict["boxes"][:, 0])
        target_dict["image_id"] = target[0]["image_id"]
        target_dict["iscrowd"] = torch.cat(iscrowd)
        return target_dict

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)
        if len(target) == 0:
            import ipdb
            ipdb.set_trace()
        target = self.process_target(target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target, None

    def __len__(self) -> int:
        return len(self.ids)


class COCO_RCNN(pl.LightningDataModule):
    def __init__(self, batch_size, data_path, fold, num_workers=4, re_train=False, og_data_path=None, in_memory=False, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.og_data_path = og_data_path
        self.transforms = get_transform()
        self.fold = fold
        self.re_train = re_train
        self.in_memory = in_memory

    def prepare_data(self):
        if self.re_train:
            train_idx = np.load(os.path.join(
                DATA_DIR, "coco", "folds", "train", "all.npy"))
            val_idx = np.load(os.path.join(
                DATA_DIR, "coco", "folds", "val", "all.npy"))
        else:
            train_idx = np.load(os.path.join(
                DATA_DIR, "coco", "folds", "train", "restricted.npy"))
            val_idx = np.load(os.path.join(DATA_DIR, "coco",
                              "folds", "val", "restricted.npy"))

        dataset = CocoDetection(root=os.path.join(DATA_DIR, "coco", "train2017"),
                                annFile=os.path.join(
                                    DATA_DIR, "coco", "annotations", "instances_train2017.json"),
                                transform=transforms.ToTensor())
        self.train = Subset(dataset, train_idx)
        dataset_val = CocoDetection(root=os.path.join(DATA_DIR, "coco", "val2017"),
                                    annFile=os.path.join(
                                        DATA_DIR, "coco", "annotations", "instances_val2017.json"),
                                    transform=transforms.ToTensor())
        self.val = Subset(dataset_val, val_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def test_ood_dataloader(self, shuffle=False):
        return DataLoader(
            self.test_ood,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--batch_size', type=int, default=15)
        parser.add_argument('--fold', type=int, default=0)
        parser.add_argument('--data_path', type=str,
                            default="mnist/alldigits/")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--re_train', type=str2bool, default=False)
        parser.add_argument('--in_memory', type=str2bool, default=False)
        return parser


if __name__ == "__main__":
    cap = CocoDetection(root=os.path.join(DATA_DIR, "coco", "val2017"),
                        annFile=os.path.join(
                            DATA_DIR, "coco", "annotations", "instances_val2017.json"),
                        transform=transforms.ToTensor())

    val_idx = np.load(os.path.join(DATA_DIR, "coco",
                      "folds", "val", "restricted.npy"))
    cap_ = Subset(cap, val_idx)
    print('Number of samples: ', len(cap_))
    img, target, _ = cap_[56]  # load 4th sample

    print("Image Size: ", img.size())
    print(target)

    mnist = Objects_Detection_Dataset(data_dir=os.path.join(
        DATA_DIR, "mnist/alldigits/train"), transforms=get_transform())
    train_idx = np.load(os.path.join(
        DATA_DIR, f"mnist/alldigits/", "folds", "0", "train_idx.npy"))
    mnist_data = Subset(mnist, train_idx)
    img_mnist, target_mnist, _ = mnist_data[56]

    dl = DataLoader(cap_, collate_fn=collate_tuple)
    for i, b in enumerate(dl):
        print(i)
    import ipdb
    ipdb.set_trace()

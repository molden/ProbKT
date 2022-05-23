import os
import pathlib
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from scipy import ndimage, misc
import skimage.draw as draw
from sklearn.model_selection import train_test_split


class CompoundGraphDataset(Dataset):

      def __init__(self, csv_file, image_dir):
          self.compounds = pd.read_csv(csv_file,index_col='molid')
          self.image_dir = image_dir
          self.molids = np.unique(self.compounds.index.values)
          self.dict_atom = {'C': 1, 'H': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'Se': 10, 'P': 11, 'B': 12, 'Si': 13}
          self.atom_list = []

      def __len__(self):
          return len(np.unique(self.compounds.index.values))

      def __getitem__(self, idx):
          unique_len = len(np.unique(self.compounds.index.values))
          idx = self.molids[idx%unique_len]
          folderindex = idx // 1000
          imageindex=idx
          imagename=self.image_dir+str(folderindex)+"/"+str(imageindex)+'.png'
          img = Image.open(imagename).convert("L")
          mask_atom=np.zeros(img.size)
          mask_objs=np.zeros(img.size)
          edgesdf = self.compounds.loc[idx]
          atoms1df=edgesdf[['atom1','atom1coord1','atom1coord2']]
          atoms1df.columns=atoms1df.columns.str.replace('atom1','atom')

          atoms2df=edgesdf[['atom2','atom2coord1','atom2coord2']]
          atoms2df.columns=atoms2df.columns.str.replace('atom2','atom')

          atomframes=[atoms1df,atoms2df]
          result=pd.concat(atomframes)
          atoms=result.drop_duplicates().values
          x = atoms[:,2:].astype(np.float32)
          num_atoms=np.size(atoms,0)
          labels = []
          edgesdf = edgesdf[edgesdf.bondtype != "nobond"]
          obj_idx = 1

          for i in range(num_atoms):
                labels.append(int(self.dict_atom.setdefault(atoms[i,0], 14)))
                mask_atom[int(atoms[i,2]),int(atoms[i,1])]=i+1
                mask_objs[int(atoms[i,2]),int(atoms[i,1])]=obj_idx+i
          mask_atom = ndimage.maximum_filter(mask_atom,size=10)
          mask_objs = ndimage.maximum_filter(mask_objs,size=10)
          # instances are encoded as different colors
          obj_ids = np.unique(mask_objs)
          # first id is the background, so remove it
          obj_ids = obj_ids[1:]

          # split the color-encoded mask into a set
          # of binary masks
          obj_masks = mask_objs == obj_ids[:, None, None]
          #masks = [np.zeros(img.size)]
          num_objs = len(obj_ids)
          boxes = []
          for i in range(num_objs):
            pos = np.where(obj_masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

          boxes = torch.as_tensor(boxes, dtype=torch.float32)
          # there is only one class
          labels = torch.as_tensor(labels, dtype=torch.int64)
         # labels = torch.ones((num_objs,), dtype=torch.int64)
          masks = torch.as_tensor(obj_masks, dtype=torch.uint8)

          image_id = torch.tensor([idx])
    #      import ipdb; ipdb.set_trace()
          if len(list(boxes.size())) == 1:
              import ipdb; ipdb.set_trace()
          area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
          # suppose all instances are not crowd
          iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

          target = {}
          target["boxes"] = boxes
          target["labels"] = labels
          target["masks"] = masks
          target["image_id"] = image_id
          target["area"] = area
          target["iscrowd"] = iscrowd
          return imagename, target

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse the rdkit labels")

    parser.add_argument("--inputfile", help="Input file with rdkit labels", type=str, default="labels_rdkit_with_S")
    parser.add_argument("--imagefolder", help="Folder where images are saved", type=str, default="images_with_S/")
    parser.add_argument("--outputfolder", help="Outputfolder where labels will be saved.", type=str, default="mol_labels")
    parser.add_argument(
        "--num-train-images", default=2000, type=int
    )
    parser.add_argument(
        "--num-test-images", default=1000, type=int
    )
    parser.add_argument(
        "--folds", default=5, type=int
    )
    args = parser.parse_args()
    dataset = CompoundGraphDataset(args.inputfile, args.imagefolder)
    #import ipdb; ipdb.set_trace()
    dirpath = pathlib.Path(args.outputfolder,"train")
    dirpath_test = pathlib.Path(args.outputfolder,"test")
    image_dir = dirpath.joinpath("images")
    label_dir = dirpath.joinpath("labels")
    image_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True, parents=True)
    image_dir_test = dirpath_test.joinpath("images")
    label_dir_test = dirpath_test.joinpath("labels")
    image_dir_test.mkdir(exist_ok=True, parents=True)
    label_dir_test.mkdir(exist_ok=True, parents=True)
    test_index = 0
    for image_id,data_item in enumerate(dataset):
        img_id=image_id
        if image_id >= args.num_train_images:
           img_id=test_index
           image_dir=image_dir_test
           label_dir=label_dir_test
           test_index+=1
        if image_id >= args.num_train_images+args.num_test_images:
           break
        image_target_path = image_dir.joinpath(f"{img_id}.png")
        label_target_path = label_dir.joinpath(f"{img_id}.txt")
        import shutil
        shutil.copyfile(data_item[0], image_target_path)
        with open(label_target_path, "w") as fp:
            fp.write("label,xmin,ymin,xmax,ymax\n")
            for l, bbox in zip(data_item[1]['labels'], data_item[1]['boxes']):
                bbox = [str(int(_)) for _ in bbox]
                to_write = f"{l}," + ",".join(bbox) + "\n"
                fp.write(to_write)        
        
    
    for fold in range(args.folds):
        idx_train, idx_val = train_test_split(np.arange(args.num_train_images),test_size = 0.3, random_state = 40 + fold)
        os.makedirs(os.path.join("./",args.outputfolder,"folds",str(fold)),exist_ok=True)
        np.save(os.path.join("./",args.outputfolder,"folds",str(fold),"train_idx.npy"),idx_train)
        np.save(os.path.join("./",args.outputfolder,"folds",str(fold),"val_idx.npy"), idx_val)

if __name__ == "__main__":
    main()


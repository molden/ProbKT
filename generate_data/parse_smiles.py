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
from rdkit import Chem


class SMILES_CountAtoms_Dataset(Dataset):

    def __init__(self, smiles_file, imagefolder):
        self.smiles_file = smiles_file
        self.smiles_df   = pd.read_csv(self.smiles_file)
        self.len_df      = self.smiles_df.shape[0]
        self.imagefolder = imagefolder

    def __len__(self):
        return self.len_df

    def __getitem__(self, idx):
        #self.dict_atom = {'C': 1, 'H': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'Se': 10, 'P': 11, 'B': 12, 'Si': 13}
        atomnumber_dict = {6: 1, 1: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7, 35: 8, 53: 9, 34: 10, 15: 11, 5: 12, 14: 13, 19: 14}
        count_atoms = np.zeros(15)
        df_row = self.smiles_df.iloc[idx]
        folderindex = idx // 1000
        imageindex=idx
        imagename=self.imagefolder+str(folderindex)+"/"+str(imageindex)+'.png'
        #import ipdb; ipdb.set_trace()
        smiles   = df_row['smiles']
        img      = Image.open(f"{imagename}").convert("L")
        #img      = Ftrans.to_tensor(img)
        m        = Chem.MolFromSmiles(smiles)
        for atom in m.GetAtoms():
            count_atoms[atomnumber_dict[atom.GetAtomicNum()]] +=1
            #print(atom.GetAtomicNum())
        #values, counts = np.unique(count_atoms, return_counts=True)
        return img, count_atoms, smiles

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse the SMILES labels")

    parser.add_argument("--inputfile", help="Input file with SMILES labels", type=str, default="smiles")
    parser.add_argument("--imagefolder", help="Folder where images are saved", type=str, default="images/")
    parser.add_argument("--outputfolder", help="Outputfolder where labels will be saved.", type=str, default="outputfolder/")
    parser.add_argument(
        "--num-test-images", default=1000, type=int
    )
    parser.add_argument(
        "--filter", default=6, type=int
    )
    args = parser.parse_args()
    dataset = SMILES_CountAtoms_Dataset(args.inputfile, args.imagefolder)
    len_data = len(dataset)
    new_i = 0
    with open(f"{args.outputfolder}/smiles", 'w') as f:
         f.write("smiles\n")
    for i in range(len_data):
        img, counts, smiles = dataset[i]
        if (not np.any(counts[args.filter:])) and (new_i<args.num_test_images):
           labels = counts[:args.filter]
           with open(f"{args.outputfolder}/labels/{new_i}.txt", "w") as fp:
                fp.write("label,xmin,ymin,xmax,ymax\n")
                for i_atom, label_count in enumerate(labels):
                    for i in range(int(label_count)):
                        to_write=f"{i_atom-1},0,0,0,0\n"
                        fp.write(to_write)
           img = img.crop((300, 300, 800, 800))
           img.save(f"{args.outputfolder}/images/{new_i}.png")
           with open(f"{args.outputfolder}/smiles", 'a') as f:
                f.write(f"{smiles}\n")
           new_i+=1
        if new_i == args.num_test_images:
            break

if __name__ == "__main__":
    main()


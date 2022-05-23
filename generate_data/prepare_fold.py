import numpy as np
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description="Prepare fold")
parser.add_argument('--data-path', default='mnist/alldigits_2/train/', help='dataset')
parser.add_argument('--fold', default=0, type=int, help='fold')
parser.add_argument('--output-path', default='mnist/alldigits_2/train_fold_0/', help='output')
args = parser.parse_args()

path_imgs = os.path.join(args.output_path, "images")
path_labels = os.path.join(args.output_path, "labels")
if not os.path.exists(path_imgs):
   os.makedirs(path_imgs)
if not os.path.exists(path_labels):
   os.makedirs(path_labels)

train_idx = np.load(os.path.join(f"{args.data_path}/../","folds",str(args.fold),"train_idx.npy"))
for i,j in enumerate(train_idx):
    shutil.copyfile(f"{args.data_path}/images/{j}.png", f"{path_imgs}/{i}.png")
    shutil.copyfile(f"{args.data_path}/labels/{j}.txt", f"{path_labels}/{i}.txt")
#import ipdb; ipdb.set_trace()

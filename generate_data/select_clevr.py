import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import pathlib

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Select CLEVR sample")

    parser.add_argument("--inputfile", help="Input file with scene labels", type=str, default="../CLEVR_v1.0/scenes/CLEVR_train_scenes.json")
    parser.add_argument("--imagefolder", help="Folder where images are saved", type=str, default="../CLEVR_v1.0/images/train/")
    parser.add_argument("--outputfolder", help="Outputfolder where labels will be saved.", type=str, default="clevr_labels/")
    parser.add_argument("--skip_cube", help="set to 1 to skip cube", type=int, default=1)
    parser.add_argument("--num_min_objects", help="set the number of min objects per image", type=int, default=5)
    parser.add_argument("--num_max_objects", help="set the number of max objects per image", type=int, default=6)
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
    count_samples = 0
    count_tr_samples = 0
    count_te_samples = 0
    train_dir=pathlib.Path(args.outputfolder,"train")
    test_dir=pathlib.Path(args.outputfolder,"test")
    image_dir_train = train_dir.joinpath("images")
    label_dir_train = train_dir.joinpath("labels")
    image_dir_test = test_dir.joinpath("images")
    label_dir_test = test_dir.joinpath("labels")
    image_dir_train.mkdir(exist_ok=True, parents=True)
    label_dir_train.mkdir(exist_ok=True, parents=True)
    image_dir_test.mkdir(exist_ok=True, parents=True)
    label_dir_test.mkdir(exist_ok=True, parents=True)
    label_dict={"cube_metal_small": 0, "cube_metal_large": 1, "cube_rubber_small": 2, "cube_rubber_large": 3, "sphere_metal_small": 4, "sphere_metal_large": 5, "sphere_rubber_small": 6, "sphere_rubber_large": 7, "cylinder_metal_small": 8, "cylinder_metal_large": 9, "cylinder_rubber_small": 10, "cylinder_rubber_large": 11}
    f = open(f"{args.inputfile}")
    data = json.load(f)
    for scene in data["scenes"]:
        cube = False
        len_scene = len(scene["objects"])
        for obj in scene["objects"]:
            if obj["shape"] == "cube":
               cube = True
        if len_scene>=args.num_min_objects and len_scene<=args.num_max_objects:
            if cube and args.skip_cube==0:
               #good sample
               if count_samples < args.num_train_images:
                   #good train sample
                   filename = f"{args.imagefolder}/{scene['image_filename']}"
                   import shutil

                   shutil.copyfile(filename, f"{image_dir_train}/{count_tr_samples}.png")
                   label_train_path = label_dir_train.joinpath(f"{count_tr_samples}.txt")
                   with open(label_train_path, "w") as fp:
                        fp.write("label,xmin,ymin,xmax,ymax\n")
                        for obj in scene["objects"]:
                            pixel_coord = obj["pixel_coords"]
                            shape    = obj["shape"]
                            material = obj["material"]
                            size     = obj["size"]
                            label    = label_dict[f"{shape}_{material}_{size}"]
                            xmin     = int(pixel_coord[0]-25)
                            xmax     = int(pixel_coord[0]+25)
                            ymin     = int(pixel_coord[1]-25)
                            ymax     = int(pixel_coord[1]+25)
                            to_write = f"{label},{xmin},{ymin},{xmax},{ymax}\n"
                            fp.write(to_write)
                   count_tr_samples+=1
               elif (count_samples < args.num_train_images + args.num_test_images):
                   #good test sample
                   filename = f"{args.imagefolder}/{scene['image_filename']}"
                   import shutil

                   shutil.copyfile(filename, f"{image_dir_test}/{count_te_samples}.png")
                   label_test_path = label_dir_test.joinpath(f"{count_te_samples}.txt")
                   with open(label_test_path, "w") as fp:
                        fp.write("label,xmin,ymin,xmax,ymax\n")
                        for obj in scene["objects"]:
                            pixel_coord = obj["pixel_coords"]
                            shape    = obj["shape"]
                            material = obj["material"]
                            size     = obj["size"]
                            label    = label_dict[f"{shape}_{material}_{size}"]
                            xmin     = int(pixel_coord[0]-25)
                            xmax     = int(pixel_coord[0]+25)
                            ymin     = int(pixel_coord[1]-25)
                            ymax     = int(pixel_coord[1]+25)
                            to_write = f"{label},{xmin},{ymin},{xmax},{ymax}\n"
                            fp.write(to_write)
                   count_te_samples+=1
               else:
                   #end loop
                   break
               count_samples+=1
            if not cube and args.skip_cube==1:
               #good sample
               if count_samples < args.num_train_images:
                   #good train sample
                   filename = f"{args.imagefolder}/{scene['image_filename']}"
                   import shutil

                   shutil.copyfile(filename, f"{image_dir_train}/{count_tr_samples}.png")
                   label_train_path = label_dir_train.joinpath(f"{count_tr_samples}.txt")
                   with open(label_train_path, "w") as fp:
                        fp.write("label,xmin,ymin,xmax,ymax\n")
                        for obj in scene["objects"]:
                            pixel_coord = obj["pixel_coords"]
                            shape    = obj["shape"]
                            material = obj["material"]
                            size     = obj["size"]
                            label    = label_dict[f"{shape}_{material}_{size}"]
                            xmin     = int(pixel_coord[0]-25)
                            xmax     = int(pixel_coord[0]+25)
                            ymin     = int(pixel_coord[1]-25)
                            ymax     = int(pixel_coord[1]+25)
                            to_write = f"{label},{xmin},{ymin},{xmax},{ymax}\n"
                            fp.write(to_write)
                   count_tr_samples+=1
               elif count_samples < args.num_train_images + args.num_test_images:
                   #good test sample
                   filename = f"{args.imagefolder}/{scene['image_filename']}"
                   import shutil

                   shutil.copyfile(filename, f"{image_dir_test}/{count_te_samples}.png")
                   label_test_path = label_dir_test.joinpath(f"{count_te_samples}.txt")
                   with open(label_test_path, "w") as fp:
                        fp.write("label,xmin,ymin,xmax,ymax\n")
                        for obj in scene["objects"]:
                            pixel_coord = obj["pixel_coords"]
                            shape    = obj["shape"]
                            material = obj["material"]
                            size     = obj["size"]
                            label    = label_dict[f"{shape}_{material}_{size}"]
                            xmin     = int(pixel_coord[0]-25)
                            xmax     = int(pixel_coord[0]+25)
                            ymin     = int(pixel_coord[1]-25)
                            ymax     = int(pixel_coord[1]+25)
                            to_write = f"{label},{xmin},{ymin},{xmax},{ymax}\n"
                            fp.write(to_write)
                   count_te_samples+=1
               else:
                   #end loop
                   break
               count_samples+=1

    for fold in range(args.folds):
        idx_train, idx_val = train_test_split(np.arange(args.num_train_images),test_size = 0.3, random_state = 40 + fold)
        os.makedirs(os.path.join("./",args.outputfolder,"folds",str(fold)),exist_ok=True)
        np.save(os.path.join("./",args.outputfolder,"folds",str(fold),"train_idx.npy"),idx_train)
        np.save(os.path.join("./",args.outputfolder,"folds",str(fold),"val_idx.npy"), idx_val)           

if __name__ == "__main__":
    main()

import os
import itertools
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import argparse

parser = argparse.ArgumentParser(description="Process the SMILES of a given file")

parser.add_argument("--inputfile", help="Input file with SMILES", type=str, required=True)
parser.add_argument("--imagefolder", help="Folder where images are saved", type=str, default="molimages/")
parser.add_argument("--filter", help="id of atoms to filter above", type=int, default=4)
parser.add_argument("--filter_num", help="number of atoms to filter above", type=int, default=35)
parser.add_argument("--outputfile", help="Outputfile where labels will be saved. If no outputfile is provided older RDKit code will be used.", type=str, default="mol_labels")
args = parser.parse_args()

inputfile   = args.inputfile
file = open(inputfile,"r")
imagefolder = args.imagefolder
index = 0 

fontsize = [18,25]
dots = [40]
width = [8,5,2]
#face = ["times", "sans", "arial"]
face = ["serif", "sans", "Helvetica"]
offset = [0.15, 0.2, 0.25]
bold = ["true", "false"]

inputdata = [
    fontsize,
    dots,
    width,
    face,
    offset,
    bold
]
result = list(itertools.product(*inputdata))
length_result = len(result)

if args.outputfile is not None:
   os.environ["RDKIT_LABELPATH"] = args.outputfile
   with open(args.outputfile, 'w') as f:
        f.write("index,molid,bondtype,atom1,charge1,atom2,charge2,atom1coord1,atom1coord2,atom2coord1,atom2coord2\n")
line = file.readline()
atomnumber_dict = {6: 1, 1: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7, 35: 8, 53: 9, 34: 10, 15: 11, 5: 12, 14: 13, 19: 14}
#atomnumber_dict[atom.GetAtomicNum()]
while line:
      m = Chem.MolFromSmiles(line)
      numatoms = m.GetNumAtoms()
      if numatoms>args.filter_num:
         line = file.readline()
         continue
      atoms_ok=True
      for atom in m.GetAtoms():
          if atom.GetAtomicNum() not in atomnumber_dict:
             atoms_ok=False
             break
          if atomnumber_dict[atom.GetAtomicNum()]>args.filter:
             atoms_ok=False
             break
      if not atoms_ok:
#         import ipdb; ipdb.set_trace()
         line = file.readline()
         continue
#      import ipdb; ipdb.set_trace()
      folderindex = index // 1000
      dirName=imagefolder+str(folderindex)
      if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
      else:    
        print("Directory " , dirName ,  " already exists")   
        print("at image ",str(index))
      
      
      option_item = result[index%length_result]
      if args.outputfile is not None:
         if option_item[5] == "true":
            os.environ["RDKIT_FONTBOLD"] = "true"
         else:
            if os.environ.get("RDKIT_FONTBOLD") is not None:
               del os.environ["RDKIT_FONTBOLD"]
         if option_item[3] == "Helvetica":
            print(f"Helvetica {index}")
         os.environ["RDKIT_FONTFACE"] = option_item[3]
         os.environ["RDKIT_LABELID"] = str(index)
         d = rdMolDraw2D.MolDraw2DCairo(300,300)
         #d.drawOptions().fixedBondLength=40 
         #d.SetFontSize(option_item[0]/20)
         d.drawOptions().bondLineWidth=option_item[2]
         d.drawOptions().multipleBondOffset=option_item[4]                     
         d.DrawMolecule(m)
         d.FinishDrawing()
         image_filename=f"{imagefolder}{folderindex}/{index}.png"
         with open(image_filename, 'wb') as f:   
              f.write(d.GetDrawingText())
      index += 1
      line = file.readline()

file.close()


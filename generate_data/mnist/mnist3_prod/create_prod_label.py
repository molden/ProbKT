import pandas as pd
import os
import numpy as np

# Read the data
for file in os.listdir('train/labels'):
    if file.endswith('.txt'):
        df = pd.read_csv('train/labels/' + file)
        df_new = pd.DataFrame()
        df_new["label"] = [np.prod(df["label"].values)]
        df_new.to_csv("train/labels/" + file, index=False)
import pandas as pd
import numpy as np
import os 

path_to_data = ""
out_path = ""

if not os.path.exists(out_path):
    os.mkdir(out_path)

np.random.seed(42)

noise = True

drop = not noise

if noise:
    for file in os.listdir(path_to_data):
        df = pd.read_excel(os.path.join(path_to_data,file))
        noisy_df = df.copy()
        for col in df.columns:
            std = noisy_df[col].std()
            noise = np.random.normal(0, 0.02 * std, size=len(noisy_df))
            noisy_df[col] += noise
        noisy_df.to_excel(os.path.join(out_path,file),index=False)
        
if drop:
    for file in os.listdir(path_to_data):
        df = pd.read_excel(os.path.join(path_to_data,file))
        drop_df = df.iloc[0::2].reset_index(drop=True)
        drop_df.to_excel(os.path.join(out_path,file),index=False)
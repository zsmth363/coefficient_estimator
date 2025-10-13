# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset
# from features import extract_features
# import pandas as pd
# import numpy as np
# import os

# class LoadExcelRegressionDataset(Dataset):
#     def __init__(self, folder, target_csv, fs):
#         """
#         folder     : path with all XXX_outputs.xlsx files
#         target_csv : CSV with columns [file, target]
#         fs         : sampling frequency
#         """
#         self.folder = folder
#         self.input_files = sorted([file for file in os.listdir(folder) if file.endswith(".xlsx")],key=lambda x:int(x.split("_")[0]))
#         for file in self.input_files:
#             df = pd.read_excel(os.path.join(folder,file))
#             if df[df.columns[2]].sum() == 0:
#                 print(file)
#                 self.input_files.remove(file)
#         param_ind = [int(_.split("_")[0]) for _ in self.input_files]
#         self.targets_arr = np.loadtxt(target_csv, delimiter=',')
#         self.targets_arr = self.targets_arr[param_ind]
#         self.targets_df = pd.DataFrame(self.targets_arr,columns=["H","Ra","Xa"])
#         self.targets_df["file"] = self.input_files
#         self.fs = fs

#     def __len__(self):
#         return len(self.targets_df)

#     def __getitem__(self, idx):
#         row = self.targets_df.iloc[idx]
#         file = row["file"]
#         target = row[["H","Ra","Xa"]].values.astype(np.float32)

#         filepath = os.path.join(self.folder, file)

#         # Load Excel → numpy array [channels, timesteps]
#         df = pd.read_excel(filepath)
#         cols = [df.columns.to_list()[i] for i in [2,5,6]]

#         df = df[cols]
#         window = df.values.T.astype(np.float32)  # [channels, timesteps]

#         # Feature engineering
#         feats = extract_features(window, self.fs)
#         feat_vector = np.array(list(feats.values()), dtype=np.float32)

#         if np.any(np.isnan(feat_vector)):
#             print(f"nan in feats: {file}")


#         return (
#             torch.tensor(window, dtype=torch.float32),   # sequence input
#             torch.tensor(feat_vector, dtype=torch.float32),  # engineered features
#             torch.tensor(target, dtype=torch.float32)    # regression target
#         )
    
import os
import numpy as np
import pandas as pd
import torch
from features import extract_features
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LoadExcelRegressionDataset(Dataset):
    def __init__(self, folder, target_csv, fs, split='train',
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 seed=42, return_sequence=True):
        """
        folder     : path with all XXX_outputs.xlsx files
        target_csv : CSV with columns [file, target]
        fs         : sampling frequency
        split      : one of ['train', 'val', 'test']
        return_sequence : if True, also returns [channels, timesteps] tensor
        """
        self.folder = folder
        self.fs = fs
        self.split = split.lower()
        self.return_sequence = return_sequence
        self.seed = seed

        # ---------- 1. Load valid input files ----------
        input_files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".xlsx")],
            key=lambda x: int(x.split("_")[0])
        )

        valid_files = []
        for file in input_files:
            df = pd.read_excel(os.path.join(folder, file))
            if df[df.columns[2]].sum() == 0:
                print(f"Skipping all-zero file: {file}")
            else:
                valid_files.append(file)

        # ---------- 2. Load target data ----------
        param_ind = [int(f.split("_")[0]) for f in valid_files]
        targets_arr = np.loadtxt(target_csv, delimiter=',')
        targets_arr = targets_arr[param_ind]
        self.targets_arr = targets_arr.astype(np.float32)
        self.files = valid_files

        # ---------- 3. Extract features (and optionally sequences) ----------
        self.features, self.windows = self._extract_all_features(valid_files)

        # ---------- 4. Split train / val / test ----------
        X_train, X_temp, y_train, y_temp, f_train, f_temp, w_train, w_temp = train_test_split(
            self.features, self.targets_arr, self.files, self.windows,
            test_size=(1 - train_ratio), random_state=seed
        )
        val_size = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test, f_val, f_test, w_val, w_test = train_test_split(
            X_temp, y_temp, f_temp, w_temp, test_size=val_size, random_state=seed
        )

        self.splits = {
            'train': (X_train, y_train, f_train, w_train),
            'val'  : (X_val,   y_val,   f_val,   w_val),
            'test' : (X_test,  y_test,  f_test,  w_test)
        }

        # ---------- 5. Fit scaler on train and apply ----------
        if self.split == 'train':
            self.scaler = StandardScaler().fit(X_train)
        else:
            self.scaler = None  # set later

        # Prepare scaled features
        self.X, self.y, self.file_list, self.windows_split = self.splits[self.split]
        if self.scaler is not None:
            self.X = self.scaler.transform(self.X)
        else:
            self.X = np.array(self.X)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.windows_split = [torch.tensor(w, dtype=torch.float32) for w in self.windows_split]

    # ---------- helper for scaling ----------
    def set_scaler(self, scaler):
        """Assign pre-fitted scaler (from train) and rescale features."""
        self.scaler = scaler
        self.X = torch.tensor(self.scaler.transform(self.X.numpy()), dtype=torch.float32)

    # ---------- feature extraction ----------
    def _extract_all_features(self, files):
        all_feats = []
        all_windows = []
        for file in files:
            path = os.path.join(self.folder, file)
            df = pd.read_excel(path)
            cols = [df.columns.to_list()[i] for i in [2, 5, 6]]  # voltage, freq, power
            df = df[cols]
            window = df.values.T.astype(np.float32)  # [channels, timesteps]

            feats = extract_features(window, self.fs)
            feat_vec = np.array(list(feats.values()), dtype=np.float32)
            if np.any(np.isnan(feat_vec)):
                print(f"NaN in features: {file}")

            all_feats.append(feat_vec)
            all_windows.append(window)
        return np.vstack(all_feats), all_windows

    # ---------- dataset API ----------
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_feat = self.X[idx]
        y = self.y[idx]
        if self.return_sequence:
            window = self.windows_split[idx]
            return window, x_feat, y
        else:
            return x_feat, y



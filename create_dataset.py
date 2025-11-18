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
from features import extract_features, extract_poly_features
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LoadExcelRegressionDataset(Dataset):
    def __init__(self, folder, target_csv, fs, split='train',
                 train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                 seed=42, return_sequence=True, use_polyfit=True):
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
        self.use_polyfit = use_polyfit
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
            # df = df.loc[df["Time(s)"] > 2]
            df["Pagg"] = df.iloc[:,1] + df.iloc[:,2]
            cols = [df.columns.to_list()[i] for i in [-1,5,6]]  # voltage, freq, power TODO: CHANGING THIS CHANGES WHICH CHANNELS ARE USED AS INPUTS
            df = df[cols]
            # df = df[df[cols[2]] > 0.9]
            # df["dPdV"] = np.gradient(df.values.T.astype(np.float32)[0],df.values.T.astype(np.float32)[2])
            window = df.values.T.astype(np.float32) # [channels, timesteps]

            feats = extract_features(window, self.fs)
            corrVP = np.corrcoef(df.values.T.astype(np.float32)[2],df.values.T.astype(np.float32)[0])[0][1]
            corrV2P = np.corrcoef(df.values.T.astype(np.float32)[2]**2,df.values.T.astype(np.float32)[0])[0][1]
            df1 = df.loc[df[cols[2]] >= 0.95].copy()

            # Create bins and group by them
            num_bins = 25
            df1['V_bin'] = pd.cut(df1[cols[2]], bins=np.linspace(df1[cols[2]].min(), df1[cols[2]].max(), num_bins + 1))

            # Compute mean per bin
            grouped = df1.groupby('V_bin',observed=True).agg({cols[2]: 'mean', cols[0]: 'mean'}).dropna()

            V_means = grouped[cols[2]].values
            P_means = grouped[cols[0]].values
            poly_coef = np.array(list(extract_poly_features(V_means,P_means/7.67).values()))
            poly_coef = np.clip(poly_coef,0,1)
            poly_coef = poly_coef/sum(poly_coef)
            
            if self.use_polyfit:
                feat_vec = np.array(list(feats.values()) + [corrVP,corrV2P]+ list(poly_coef), dtype=np.float32)
            else:
                feat_vec = np.array(list(feats.values()) + [corrVP,corrV2P], dtype=np.float32)
            if np.any(np.isnan(feat_vec)):
                print(f"NaN in features: {file}")

            all_feats.append(feat_vec)
            df[cols[0]] = df[cols[0]]/7.67 # This is hardcoded. Needs to be generalized
            window = torch.tensor(df.values, dtype=torch.float32)  # shape (timesteps, channels)
            all_windows.append(window)
            all_equal = len(set(len(sublist) for sublist in all_windows)) == 1
            if not all_equal:
                print(file)
        return np.vstack(all_feats), all_windows
    
    def rotate_data(self,V, P, degrees, around_mean=True):
        # Convert to radians
        theta = np.deg2rad(degrees)
        
        # Optionally rotate around the mean instead of the origin
        if around_mean:
            V0, P0 = np.median(V), np.median(P)
            Vc, Pc = V - V0, P - P0
        else:
            Vc, Pc = V, P
        
        # Rotation matrix
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        
        # Apply rotation
        rotated = rotation_matrix @ np.vstack([Vc, Pc])
        V_rot, P_rot = rotated[0, :], rotated[1, :]
        
        # Shift back if rotated around mean
        if around_mean:
            V_rot += V0
            P_rot += P0

        return V_rot, P_rot

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



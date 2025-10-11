import torch
import torch.nn as nn
from torch.utils.data import Dataset
from features import extract_features
import pandas as pd
import numpy as np
import os

class HybridModel(nn.Module):
    def __init__(self, seq_input_channels, seq_len, feat_dim):
        super(HybridModel, self).__init__()

        # # Sequence branch (CNN + LSTM)
        self.cnn = nn.Sequential(
            nn.Conv1d(seq_input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)

        # Feature branch (MLP)
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Fusion + regression head
        self.regressor = nn.Sequential(
            nn.Linear(64+32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)   # <-- output a single value
        )

    def forward(self, seq, feats):
        # seq: [batch, channels, seq_len]
        x = self.cnn(seq)                        # [batch, 32, seq_len/2]
        x = x.permute(0, 2, 1)                   # [batch, seq_len/2, 32]
        _, (h, _) = self.lstm(x)                 # take final hidden state
        seq_embed = h[-1]                        # [batch, 64]

        feat_embed = self.feat_mlp(feats)        # [batch, 32]

        combined = torch.cat([seq_embed, feat_embed], dim=1)  # [batch, 96]
        out = self.regressor(combined)           # [batch, 1]
        return out.squeeze(-1)                   # [batch]
    
    # def forward(self, seq, feats):
    #     # seq: [batch, channels, seq_len]
    #     # x = self.cnn(seq)                        # [batch, 32, seq_len/2]
    #     # x = x.permute(0, 2, 1)                   # [batch, seq_len/2, 32]
    #     # _, (h, _) = self.lstm(x)                 # take final hidden state
    #     # seq_embed = h[-1]                        # [batch, 64]

    #     feat_embed = self.feat_mlp(feats)        # [batch, 32]

    #     # combined = torch.cat([seq_embed, feat_embed], dim=1)  # [batch, 96]
    #     out = self.regressor(feat_embed)           # [batch, 1]
    #     return out.squeeze(-1)    


class LoadExcelRegressionDataset(Dataset):
    def __init__(self, folder, target_csv, fs):
        """
        folder     : path with all XXX_outputs.xlsx files
        target_csv : CSV with columns [file, target]
        fs         : sampling frequency
        """
        self.folder = folder
        self.input_files = sorted([file for file in os.listdir(folder) if file.endswith(".xlsx")],key=lambda x:int(x.split("_")[0]))
        for file in self.input_files:
            df = pd.read_excel(os.path.join(folder,file))
            if df[df.columns[2]].sum() == 0:
                print(file)
                self.input_files.remove(file)
        param_ind = [int(_.split("_")[0]) for _ in self.input_files]
        self.targets_arr = np.loadtxt(target_csv, delimiter=',')
        self.targets_arr = self.targets_arr[param_ind]
        self.targets_df = pd.DataFrame(self.targets_arr,columns=["H","Ra","Xa"])
        self.targets_df["file"] = self.input_files
        self.fs = fs

    def __len__(self):
        return len(self.targets_df)

    def __getitem__(self, idx):
        row = self.targets_df.iloc[idx]
        file = row["file"]
        target = row[["H","Ra","Xa"]].values.astype(np.float32)

        filepath = os.path.join(self.folder, file)

        # Load Excel → numpy array [channels, timesteps]
        df = pd.read_excel(filepath)
        cols = [df.columns.to_list()[i] for i in [2,5,6]]

        df = df[cols]
        window = df.values.T.astype(np.float32)  # [channels, timesteps]

        # Feature engineering
        feats = extract_features(window, self.fs)
        feat_vector = np.array(list(feats.values()), dtype=np.float32)

        if np.any(np.isnan(feat_vector)):
            print(f"nan in feats: {file}")


        return (
            torch.tensor(window, dtype=torch.float32),   # sequence input
            torch.tensor(feat_vector, dtype=torch.float32),  # engineered features
            torch.tensor(target, dtype=torch.float32)    # regression target
        )

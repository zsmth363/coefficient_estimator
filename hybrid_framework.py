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




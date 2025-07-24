import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def main():
    ce = Coefficient_Estimator('/home/zachsmith/Desktop/ML/data/morteza_data/ZIP_500_samples_smaller_TimeStep', 
                                '/home/zachsmith/Desktop/ML/data/morteza_data/',
                                coef_index=0)
    ce.load_data()
    ce.fit_coefficients()
    ce.train_model()
    ce.save_model()
    ce.load_model()
    ce.evaluate_unseen(idx_range=[len(ce.sequences)-5,len(ce.sequences)])


class InputsDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class CNN(nn.Module):
    def __init__(self, out_chnl1, out_chnl2, out_chnl3, ks_1, ks_2, ks_3, num_steps,dropout_rate=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=out_chnl1, kernel_size=ks_1, padding=ks_1 // 2)
        self.bn1 = nn.BatchNorm1d(out_chnl1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(out_chnl1, out_chnl2, kernel_size=ks_2, padding=ks_2 // 2)
        self.bn2 = nn.BatchNorm1d(out_chnl2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv1d(out_chnl2, out_chnl3, kernel_size=ks_3, padding=ks_3 // 2)
        self.bn3 = nn.BatchNorm1d(out_chnl3)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout_rate)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_chnl3, int(num_steps))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        x = self.drop3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class Coefficient_Estimator:
    def __init__(self, input_path, target_path,**kwargs):
        """
        Parameters
        ----------
        coef_index: int
            Integer representing the index of the coefficient to be estimated in the zip target arrays. 
            {Z:0,I:1,P:2}
        """
        self.input_path = input_path
        self.target_path = target_path

        # Hyperparameters
        self.out_chnl1 = 64
        self.out_chnl2 = 64
        self.out_chnl3 = 32
        self.ks_1 = 5
        self.ks_2 = 5
        self.ks_3 = 3
        self.batch_size = 16
        self.num_epochs = 150
        self.learning_rate = 0.001
        self.coef_index = kwargs.get('coef_index',2)
        self.test_train_split_factor = 0.8
        self.target_quantization_step = 0.2
        num_steps = 1/self.target_quantization_step + 1

        self.model = CNN(self.out_chnl1, self.out_chnl2, self.out_chnl3, self.ks_1, self.ks_2, self.ks_3,num_steps)
    
    def fit_coefficients(self,V, P):
        """
        Estimate a, b, c in P = a*V^2 + b*V + c for one sequence.
        V, P: 1D arrays of same length
        """
        X = np.vstack([V**2, V, np.ones_like(V)]).T  # shape (T, 3)
        coeffs, *_ = np.linalg.lstsq(X, P, rcond=None)
        return coeffs  # [a, b, c]
    
    @staticmethod
    def quantize_targets(targets, step=0.2):
        """
        Quantize float targets to nearest level in [0, 1] and map to class index 0-4.
        """
        quantized = np.round(targets / step).astype(int)
        return quantized  # Values in {0, 1, 2, 3, 4}

    def load_data(self):
        dyn_results = sorted(
            [f for f in os.listdir(self.input_path) if f.endswith('.csv')],
            key=lambda x: x.split('_')[1]
        )
        zip_files = [f for f in os.listdir(self.target_path) if 'zip_' in f and '500' in f]

        # Load targets
        target_data = []
        for file in zip_files:
            df = pd.read_csv(os.path.join(self.target_path, file))
            cols = df.columns
            target_data.append(df[[cols[0], cols[1], cols[2]]].values)
        zip_array = np.vstack(target_data)
        self.zip_array = zip_array #TODO: Remove or modify this later
        targets = zip_array.T[self.coef_index].round()
        targets = Coefficient_Estimator.quantize_targets(zip_array.T[self.coef_index],self.target_quantization_step)
        # targets = zip_array.T[self.coef_index]

        # Load inputs
        input_array = []
        for file in dyn_results:
            df = pd.read_csv(os.path.join(self.input_path, file))
            cols = df.columns
            input_data = np.column_stack((df[cols[1]], df[cols[3]], df[cols[4]]))
            input_array.append(input_data)

        # self.sequences = [x[5:50] for x in input_array]
        # self.sequences = [x[2:12] for x in input_array]

        self.sequences = input_array
        for seq_array,zip in zip(input_array,zip_array):
            V = seq_array.T[0]
            P = seq_array.T[2]/1575
            coef_est = self.fit_coefficients(V,P)
            print(f"Estimated coefficients: {coef_est}")
            print(f"Actual coefficients: {zip}")
            
        self.targets = targets
        test_train_split_index = round(len(input_array)*self.test_train_split_factor) 
        self.train_seq = self.scale_sequences(input_array[0:test_train_split_index])
        self.test_seq = self.scale_sequences(input_array[test_train_split_index:-5])

        self.train_targets = targets[0:test_train_split_index]
        self.test_targets = targets[test_train_split_index:-5]

    def scale_sequences(self, sequences):
        """
        Scales each sequence (2D array) using MinMaxScaler per column.
        If a column is constant, it is scaled to 1.

        Parameters
        ----------
        sequences : list of np.ndarray
            Each ndarray is (seq_len, features)

        Returns
        -------
        list of np.ndarray
            Scaled sequences with same shape
        """
        scaled_sequences = []

        for seq in sequences:
            seq_scaled = np.zeros_like(seq)
            for i in range(seq.shape[1]):
                column = seq[:, i].reshape(-1, 1)
                if np.all(column == column[0]):
                    seq_scaled[:, i] = 0  # Constant column, set all values to 0
                else:
                    scaler = MinMaxScaler(feature_range=(-1,1))
                    seq_scaled[:, i] = scaler.fit_transform(column).flatten()
            scaled_sequences.append(seq_scaled)

        return scaled_sequences

if __name__ == '__main__':
    main()
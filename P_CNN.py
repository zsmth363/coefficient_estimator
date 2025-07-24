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
    ce = Coefficient_Estimation('/home/zachsmith/Desktop/ML/data/morteza_data/ZIP_500_samples_smaller_TimeStep', 
                                '/home/zachsmith/Desktop/ML/data/morteza_data/',
                                coef_index=2)
    ce.load_data()
    ce.train_model()
    ce.save_model()
    ce.load_model()
    ce.evaluate_unseen(idx_range=[len(ce.sequences)-5,len(ce.sequences)])


class InputsDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class CNNBinaryClassifier(nn.Module):
    def __init__(self, out_chnl1, out_chnl2, ks_1, ks_2, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=out_chnl1, kernel_size=ks_1, padding=ks_1 // 2)
        self.bn1 = nn.BatchNorm1d(out_chnl1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(out_chnl1, out_chnl2, kernel_size=ks_2, padding=ks_2 // 2)
        self.bn2 = nn.BatchNorm1d(out_chnl2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_chnl2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class Coefficient_Estimation:
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
        self.ks_1 = 5
        self.ks_2 = 5
        self.batch_size = 16
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.coef_index = kwargs.get('coef_index',2)
        self.test_train_split_factor = 0.8

        self.model = CNNBinaryClassifier(self.out_chnl1, self.out_chnl2, self.ks_1, self.ks_2)

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
        self.targets = targets
        test_train_split_index = round(len(input_array)*self.test_train_split_factor) 
        # self.train_seq = input_array[0:test_train_split_index]
        # self.test_seq = input_array[test_train_split_index:-5]
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

    def train_model(self):
        train_dataset = InputsDataset(self.train_seq, self.train_targets)
        test_dataset = InputsDataset(self.test_seq, self.test_targets)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        train_losses, val_losses, val_accuracies = [], [], []

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for sequences, targets in train_loader:
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            train_loss = total_loss / len(train_loader)
            train_losses.append(train_loss)

            val_loss, correct, total = 0, 0, 0
            self.model.eval()
            with torch.no_grad():
                for val_seq, val_target in test_loader:
                    val_output = self.model(val_seq)
                    val_loss += criterion(val_output, val_target).item()

                    preds = (val_output > 0.5).float()
                    correct += (preds == val_target).sum().item()
                    total += val_target.size(0)

            val_losses.append(val_loss / len(test_loader))
            val_accuracies.append(correct / total)

            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_losses[-1]:.4f} | Accuracy: {val_accuracies[-1]:.2%}")

        self.plot_metrics(train_losses, val_losses, val_accuracies)

    def plot_metrics(self, train_losses, val_losses, val_accuracies):
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(val_accuracies, label='Validation Accuracy', color='green')
        plt.title("Validation Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_model(self, path="Pcnn_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="Pcnn_model.pth"):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict_sequence(self, sequence):
        seq_tensor = torch.tensor(sequence.astype('float32')).unsqueeze(0)
        output = self.model(seq_tensor)
        prob = torch.sigmoid(output)
        pred_label = int(prob.item() > 0.5)
        return prob.item(), pred_label

    def predict_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        seq = df[['volt', 'freq', 'curr']].values.astype('float32')
        return self.predict_sequence(seq)

    def evaluate_unseen(self, idx_range):
        """
        Parameters
        ----------
        idx_range: list[int,int]
            List containing the index range of the sequences and zip targets not used in testing and training.
        """
        sequences = self.scale_sequences(self.sequences[idx_range[0]:idx_range[1]])
        actual_zip = self.zip_array[idx_range[0]:idx_range[1]]
        targets = self.targets[idx_range[0]:idx_range[1]]
        for i in range(len(sequences)):
            prob, pred_label = self.predict_sequence(np.array(sequences[i]))
            print(f"Predicted coefficient: {prob:.4f}")
            print(f"Actual Coeffcient: {actual_zip[i]}")
            print(f"Predicted label: {pred_label}")
            print(f"Actual label: {targets[i]}")

if __name__ == '__main__':
    main()

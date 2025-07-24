import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler

path_to_input_data = '/home/zachsmith/Desktop/ML/data/morteza_data//out_files_N_500'
path_to_target_data = '/home/zachsmith/Desktop/ML/data/morteza_data/'

# dyn_results = [f for f in os.listdir(path_to_input_data) if os.path.isfile(os.path.join(path_to_input_data,f)) and '.xlsx' in f and f.split('_')[1] == '0']
# zip_files = [f for f in os.listdir(path_to_target_data) if os.path.isfile(os.path.join(path_to_target_data,f)) and 'zip_data_sim-0' in f]

dyn_results = [f for f in os.listdir(path_to_input_data) if os.path.isfile(os.path.join(path_to_input_data,f)) and '.csv' in f]
zip_files = [f for f in os.listdir(path_to_target_data) if os.path.isfile(os.path.join(path_to_target_data,f)) and 'zip_' in f and '500' in f]

dyn_results.sort(key=lambda x: x.split('_')[1])

combined_data = []  # This will store all rows from all files 
for file_path in zip_files:
    # Read the Excel file into a DataFrame
    df = pd.read_csv(os.path.join(path_to_target_data,file_path),header=0)
    cols = df.columns.values
    combined_data.append(df[[cols[0], cols[1], cols[2]]].values)


# Stack all the data into a single 2D numpy array
zip_array = np.vstack(combined_data)
targets = zip_array.T[2].round()
input_array = []  # This will store the (seq_len, 2) arrays for each file

constant_count = 0
constant_idx = []
for file_path in dyn_results:
    # Read the Excel file into a DataFrame
    df = pd.read_csv(os.path.join(path_to_input_data,file_path))
    cols = df.columns.values
    
    # Extract the 'Freq' and 'Volt' columns and stack them into a (seq_len, 2) array
    input_data = np.column_stack((df[cols[1]],df[cols[3]],df[cols[4]])) # Array with [V,I,MW]
    # scale featuress
    # plt.plot(df.current_pu)
    input_array.append(input_data)

plt.show()
sequences = [_[5:50] for _ in input_array]
targets = targets
train_seq = input_array[0:144]
test_seq = input_array[144:175]

train_targets = targets[0:144]
test_targets = targets[144:175]

# unk_seq_1 = sequences[95].astype('float32')
# unk_zip_1 = targets[95]

class InputsDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)  # shape: (N, 1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# Hyperparameters
out_chnl1 = 64
out_chnl2 = 64
ks_1 = 5
ks_2 = 5
batch_size = 16
num_epochs = 50
learning_rate = 0.001

# Dataset and loader
train_dataset = InputsDataset(train_seq, train_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = InputsDataset(test_seq, test_targets)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


def train(model, train_loader, val_loader, epochs, lr):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sequences, targets in train_loader:
            outputs = model(sequences)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_seq, val_target in val_loader:
                val_output = model(val_seq)
                val_loss += criterion(val_output, val_target).item()

                preds = (val_output > 0.5).float()
                correct += (preds == val_target).sum().item()
                total += val_target.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.2%}")

    return train_losses, val_losses, val_accuracies

class CNNBinaryClassifier(nn.Module):
    def __init__(self,out_chnl1,out_chnl2,ks_1,ks_2,dropout_rate=0.3):
        """
        Parameters
        ----------
        out_chnl1: int
            number of channels between Conv layers
        out_chnl2: int
            number of channels between second Conv layer and Linear function
        ks_1: int
            kernal size in first Conv layer
        ks_2: int
            kernal size in second Conv layer
        """
        super(CNNBinaryClassifier, self).__init__()
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
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(x).squeeze(-1)  # (batch, out_chnl2)
        x = self.fc(x)
        return x  # raw logits (used with BCEWithLogitsLoss)


model = CNNBinaryClassifier(out_chnl1,out_chnl2,ks_1,ks_2)
# Train the model and get metrics
train_losses, val_losses, val_accuracies = train(model, train_loader, test_loader, epochs=num_epochs, lr=learning_rate)

# Plot loss
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

# Plot accuracy
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

torch.save(model.state_dict(), "Pcnn_model.pth")

# Recreate the model with same structure
model = CNNBinaryClassifier(out_chnl1,out_chnl2,ks_1,ks_2)

# Load the saved weights
model.load_state_dict(torch.load("Pcnn_model.pth"))
model.eval()  # Set model to evaluation mode


# seq_tensor = torch.tensor(unk_seq_1).unsqueeze(0)  # shape: (1, seq_len, 2)
# output = model(seq_tensor)
for i in range(175,180):
    seq_tensor = torch.tensor(sequences[i].astype('float32')).unsqueeze(0)
    output = model(seq_tensor)
    prob = torch.sigmoid(output)
    pred_label = int(prob.item() > 0.5)

    print(f"Predicted probability: {prob.item():.4f}")
    print(f"Predicted label: {pred_label}")
    print(f"Actual label: {targets[i]}")

    print(f"Check against: {targets[i]}")

def predict(model, csv_path):
    df = pd.read_csv(csv_path)
    seq = df[['volt', 'freq', 'curr']].values.astype('float32')  # adjust if needed
    seq_tensor = torch.tensor(seq).unsqueeze(0)  # shape: (1, seq_len, 3)
    output = model(seq_tensor)
    prob = torch.sigmoid(output)
    pred_label = int(prob.item() > 0.5)
    return prob.item(), pred_label


"""
Inputs: P, V, I

Targets: Rounded P coefficients either 1 or 0 if Pcoef >= 0.5, 1 else 0

If the CNN outputs 1, then Pcoef > 0.5 therefore Icoef + Zcoef <= 0.5

From here we can either try to guess all three, by providing the P,V,I + CNN guess as inputs OR
we can do the same process but with I, but included CNN guess as input. 

Then if we can find P,I coefficients with relatively high accuracy, Z coefficient can be extrapolated.

"""
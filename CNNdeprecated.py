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

# matplotlib.use('qtagg')
# Sample dataset class
# NOTE (6/23): Truncate input data to start at fault clearing time
# NOTE (6/23): Need to generate new data with system from Mo 


# path_to_input_data = '/home/zachsmith/Desktop/ML/data/inputs'
# path_to_target_data = '/home/zachsmith/Desktop/ML/data/targets'

path_to_input_data = '/home/zachsmith/Desktop/ML/data/morteza_data/out_files_N_500'
path_to_target_data = '/home/zachsmith/Desktop/ML/data/morteza_data'

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

input_array = []  # This will store the (seq_len, 2) arrays for each file

for file_path in dyn_results:
    # Read the Excel file into a DataFrame
    df = pd.read_csv(os.path.join(path_to_input_data,file_path))
    cols = df.columns.values
    
    # Extract the 'Freq' and 'Volt' columns and stack them into a (seq_len, 2) array
    input_data = np.column_stack((df[cols[1]],df[cols[3]],df[cols[4]])) # Array with [V,f,I,MW]
    # scale featuress

    last_two = input_data[:, -2:]

    # Check if any of the columns are constant
    is_constant = np.all(last_two == last_two[0, :], axis=0)

    # Initialize output with same shape
    scaled_last_two = np.copy(last_two)

    # Apply MinMaxScaler only to non-constant columns
    if not np.all(is_constant):  # If not all columns are constant
        scaler = MinMaxScaler()
        # Only scale non-constant columns
        non_constant_cols = ~is_constant
        scaled_last_two[:, non_constant_cols] = scaler.fit_transform(last_two[:, non_constant_cols])

    # Replace the last two columns in the original data with the scaled version
    input_data[:, -2:] = scaled_last_two

    input_array.append(input_data)

class InputsDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)  # output shape (batch, 64, 1)
        self.fc = nn.Linear(64, 3)
        self.softmax = nn.Softmax(dim=1)  # To ensure output sums to 1

    def forward(self, x):
        # Input x: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)  # → (batch, channels=3, seq_len)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # → (batch, 64)
        x = self.fc(x)
        return self.softmax(x)  # output: (batch, 3)


# Hyperparameters
hidden_size = 216
batch_size = 24
num_epochs = 50
learning_rate = 0.0001

sequences = [_[5:50] for _ in input_array]
targets = zip_array
train_seq = input_array[0:400]
test_seq = input_array[400:495]

train_targets = zip_array[0:400]
test_targets = zip_array[400:495]

unk_seq_1 = sequences[95].astype('float32')
unk_zip_1 = targets[95]

# Dataset and loader
train_dataset = InputsDataset(train_seq, train_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_dataset = InputsDataset(test_seq, test_targets)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train(model, train_loader, val_loader, epochs=20, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_seq, val_target in val_loader:
                val_output = model(val_seq)
                val_loss += criterion(val_output, val_target).item()
            

        # # Convert predictions and targets to NumPy arrays
        # preds = val_output.numpy()
        # targets_np = val_loss.numpy()

        # # Plot true vs predicted for each class
        # labels = ['a', 'b', 'c']
        # for i in range(3):
        #     plt.figure(figsize=(10, 4))
        #     plt.plot(targets_np[:, i], label='True', marker='o')
        #     plt.plot(preds[:, i], label='Predicted', marker='x')
        #     plt.title(f'Component {labels[i]}: True vs Predicted')
        #     plt.xlabel('Sample Index')
        #     plt.ylabel('Value')
        #     plt.legend()
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.show()

        
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}")


model = CNNModel()
train(model, train_loader, test_loader, epochs=num_epochs,lr=learning_rate)
torch.save(model.state_dict(), "cnn_model.pth")

# Recreate the model with same structure
model = CNNModel()

# Load the saved weights
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()  # Set model to evaluation mode


# seq_tensor = torch.tensor(unk_seq_1).unsqueeze(0)  # shape: (1, seq_len, 2)
# output = model(seq_tensor)
for i in range(495,500):
    seq_tensor = torch.tensor(sequences[i].astype('float32')).unsqueeze(0)
    output = model(seq_tensor)


    print(f"Predicted output: {output}")
    print(f"Check against: {targets[i]}")

def predict(model, csv_path):
    df = pd.read_csv(csv_path)
    seq = df[['volt', 'freq']].values.astype('float32')
    seq_tensor = torch.tensor(sequences[i].astype('float32')).unsqueeze(0)  # shape: (1, seq_len, 2)
    
    output = model(seq_tensor)
    return output.squeeze(0).numpy()  # shape: (3,)
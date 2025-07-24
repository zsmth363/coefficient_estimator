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

# Example RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1,output_size=3):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        # Take output from the last timestep
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    

# Hyperparameters
hidden_size = 216
batch_size = 24
num_epochs = 50
learning_rate = 0.001

sequences = [_[10:45] for _ in input_array]
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
        
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}")


model = RNNModel(hidden_size=hidden_size)
train(model, train_loader, test_loader, epochs=num_epochs,lr=learning_rate)
torch.save(model.state_dict(), "rnn_model.pth")

# Recreate the model with same structure
model = RNNModel(input_size=3, hidden_size=hidden_size, num_layers=1, output_size=3)

# Load the saved weights
model.load_state_dict(torch.load("rnn_model.pth"))
model.eval()  # Set model to evaluation mode

# seq_tensor = torch.tensor(unk_seq_1).unsqueeze(0)  # shape: (1, seq_len, 2)
# output = model(seq_tensor)
for i in range(495,500):
    seq_tensor = torch.tensor(sequences[i].astype('float32')).unsqueeze(0)  # shape: (1, seq_len, 2)
    output = model(seq_tensor)

    print(f"Predicted output: {output}")
    print(f"Check against: {targets[i]}")

def predict(model, csv_path):
    df = pd.read_csv(csv_path)
    seq = df[['volt', 'freq']].values.astype('float32')
    seq_tensor = torch.tensor(seq).unsqueeze(0)  # shape: (1, seq_len, 2)
    
    output = model(seq_tensor)
    return output.squeeze(0).numpy()  # shape: (3,)
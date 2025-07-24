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


# Sample dataset class

path_to_input_data = '/home/zachsmith/Desktop/ML/data/inputs'
path_to_target_data = '/home/zachsmith/Desktop/ML/data/targets'

dyn_results = [f for f in os.listdir(path_to_input_data) if os.path.isfile(os.path.join(path_to_input_data,f)) and '.xlsx' in f and f.split('_')[1] == '0']
zip_files = [f for f in os.listdir(path_to_target_data) if os.path.isfile(os.path.join(path_to_target_data,f)) and 'zip_data_sim-0' in f]

combined_data = []  # This will store all rows from all files 
for file_path in zip_files:
    # Read the Excel file into a DataFrame
    df = pd.read_csv(os.path.join(path_to_target_data,file_path),header=None)
    cols = df.columns.values
    combined_data.append(df[[cols[0], cols[1], cols[2]]].values)


# Stack all the data into a single 2D numpy array
zip_array = np.vstack(combined_data)

freq_volt_array = []  # This will store the (seq_len, 2) arrays for each file

for file_path in dyn_results:
    # Read the Excel file into a DataFrame
    df = pd.read_excel(os.path.join(path_to_input_data,file_path))
    cols = df.columns.values
    
    # Extract the 'Freq' and 'Volt' columns and stack them into a (seq_len, 2) array
    freq_volt_data = np.column_stack((df[cols[1]], df[cols[2]]))
    
    # Append this (seq_len, 2) array to the parent list
    freq_volt_array.append(freq_volt_data)



# freq_volt_df = pd.DataFrame(freq_volt_array)
# zip_df = pd.DataFrame(combined_data_array)


class VoltageFrequencyDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# Example RNN model
class RNNRegressor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1,output_size=3):
        super(RNNRegressor, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        # Take output from the last timestep
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    


# Hyperparameters
hidden_size = 64
batch_size = 64
num_epochs = 20
learning_rate = 0.001

# Dummy data shapes: (10000, seq_len, 2), (10000, 3)
# Replace with your real data
# sequences = torch.randn(10000, 50, 2)  # Example: 50 timesteps
# targets = torch.randn(10000, 3)

sequences = freq_volt_array
targets = zip_array

# Dataset and loader
dataset = VoltageFrequencyDataset(sequences, targets)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Model, loss, optimizer
model = RNNRegressor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

def train(model, dataloader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for sequences, targets in dataloader:
            outputs = model(sequences)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")



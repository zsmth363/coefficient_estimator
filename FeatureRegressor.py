import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureRegressor(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=128, num_layers=1, dropout=0.1):
        """
        input_dim  : number of features per sample
        output_dim : number of regression targets (e.g., 3 for H, Ra, Xa)
        hidden_dim : neurons per hidden layer
        num_layers : number of hidden layers
        dropout    : dropout probability
        """
        super().__init__()

        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

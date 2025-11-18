from create_dataset import LoadExcelRegressionDataset
from FeatureRegressor import FeatureRegressor
import numpy as np
import pandas as pd
import torch 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import itertools


# -------------------------------------------------------
# Split dataset into train/val/test
# -------------------------------------------------------

path = os.getcwdb()

path_to_inputs = ""
path_to_targets = ""
path_to_output = ""

if not os.path.exists(path_to_output):
    os.mkdir(path_to_output)

fs = 120.3 # Sampling rate from PSS/E

use_hybrid = True
use_polyfit = False

def main(**kwargs):

    train_set = LoadExcelRegressionDataset(
        path_to_inputs, path_to_targets, fs=fs, split='train', return_sequence=use_hybrid,seed=kwargs.get("seed"),use_polyfit=use_polyfit
    )

    # --- Validation / test sets (reuse train scaler) ---
    val_set = LoadExcelRegressionDataset(
        path_to_inputs, path_to_targets, fs=fs, split='val', return_sequence=use_hybrid,seed=kwargs.get("seed"),use_polyfit=use_polyfit
    )
    val_set.set_scaler(train_set.scaler)

    test_set = LoadExcelRegressionDataset(
        path_to_inputs, path_to_targets, fs=fs, split='test', return_sequence=use_hybrid,seed=kwargs.get("seed"),use_polyfit=use_polyfit
    )
    test_set.set_scaler(train_set.scaler)          

    X_train = train_set.X.numpy()
    y_train = train_set.y.numpy()
    X_val = val_set.X.numpy()
    y_val = val_set.y.numpy()    
    X_test = test_set.X.numpy()
    y_test = test_set.y.numpy()    

    # Scale targets
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    y_test_scaled = target_scaler.transform(y_test)

    # Convert back to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

    # Loaders
    train_loader = DataLoader(list(zip(train_set.windows_split,X_train_t, y_train_t)), batch_size=32, shuffle=True)
    val_loader   = DataLoader(list(zip(val_set.windows_split,X_val_t, y_val_t)), batch_size=32, shuffle=False)
    test_loader  = DataLoader(list(zip(test_set.windows_split,X_test_t, y_test_t)), batch_size=32, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # param_grid = {
    #     "num_layers": [1,2, 3],
    #     "hidden_dim": [64,128,256],
    #     "dropout": [0.1, 0.2, 0.3],
    #     "lr": [1e-3, 5e-4],
    #     "lstm_hidden": [128],
    #     "lstm_layers": [1],
    # }

    param_grid = {
        "num_layers": [2],
        "hidden_dim": [256],
        "dropout": [0.2],
        "lr": [1e-3],
        "lstm_hidden": [128],
        "lstm_layers": [1],
    }

    all_combinations = list(itertools.product(*param_grid.values()))

    i = 0
    hyp_param_results = [[] for _ in range(len(all_combinations))]
    for combo in all_combinations:

        num_layers, hidden_dim, dropout, lr, lstm_hidden, lstm_layers = combo
        hyp_param_results[i].append(combo)
        model = FeatureRegressor(train_set.X.shape[1],output_dim=6,hidden_dim=hidden_dim,num_layers=num_layers,dropout=dropout,
                                        use_lstm=use_hybrid,lstm_hidden=lstm_hidden,lstm_layers=lstm_layers,lstm_input_dim=3,bidirectional=False)
        model = model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        early_stopper = EarlyStopping(patience=15, min_delta=1e-4, mode='min')

        # Train
        model,train_losses,val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, device, early_stopper,epochs=300,target_scaler=target_scaler)

        # Test
        stats,results = test(model, test_loader, device,target_scaler=target_scaler,save_path=None)

        if stats["R2"] > 0.8:
            os.mkdir(os.path.join(path_to_output,f"param_index_{i}"))
            params_df = pd.DataFrame(combo).T
            params_df.columns = ["num_layers","hidden_dim","dropout","lr","lstm_hidden","lstm_layers"]
            params_df.to_csv(os.path.join(path_to_output,f"param_index_{i}","hyperparameters.csv"),index=False)
            results_df = pd.DataFrame(np.concat([results[0],results[1]],axis=1),columns=["a_pred","b_pred","c_pred","H_pred","R1_pred","X1_pred","a","b","c","H","R1","X1"])
            results_df.to_excel(os.path.join(path_to_output,f"param_index_{i}","results.xlsx"),index=False)
        print("---------------------- \n")
        print(combo)
        print(stats)
        print("---------------------- \n")
        hyp_param_results[i].append(stats)
        i+=1
    r_arr = [list(_[0])+list(_[1].values()) for _ in hyp_param_results]
    df = pd.DataFrame(r_arr,columns = ["num_layers","hidden_dim","dropout","lr","lstm_hidden","lstm_layers","MAE","RMSE","R^2"])
    print(df)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total = 0.0, 0

    for seq, feats, labels in loader:
        seq, feats, labels = seq.to(device),feats.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(feats,seq)  
        loss = criterion(outputs, labels)    
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)

    avg_loss = total_loss / total
    return avg_loss

def validate(model, loader, criterion, device, target_scaler=None):
    model.eval()
    total_loss, total = 0.0, 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for seq, feats, labels in loader:
            seq, feats, labels = seq.to(device),feats.to(device), labels.to(device)
            outputs = model(feats,seq)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            preds_all.append(outputs.cpu())
            labels_all.append(labels.cpu())

    avg_loss = total_loss / total

    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    # Apply inverse transform if scaler is provided
    if target_scaler is not None:
        preds_all = target_scaler.inverse_transform(preds_all)
        labels_all = target_scaler.inverse_transform(labels_all)

    mae = mean_absolute_error(labels_all, preds_all)
    rmse = mean_squared_error(labels_all, preds_all)
    r2 = r2_score(labels_all, preds_all)
    return avg_loss, mae, rmse, r2

def train_model(model, train_loader, val_loader, criterion, optimizer, device, early_stopper, epochs=20,target_scaler=None):
    best_val_loss = float("inf")
    best_model_state = None
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, mae, rmse, r2 = validate(model, val_loader, criterion, device, target_scaler)
        print(f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}, "
            f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        early_stopper(val_loss, model, epoch)
        if early_stopper.early_stop:
            break   

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model,train_losses,val_losses

def test(model, loader, device, target_scaler=None, save_path=None):
    """
    Evaluate model on test data with proper inverse transform.

    Args:
        model          : PyTorch model
        loader         : DataLoader for test set
        device         : 'cpu' or 'cuda'
        target_scaler  : sklearn StandardScaler fitted on training targets
        save_path      : optional path to save prediction vs true plot

    Returns:
        metrics        : dict with MAE, RMSE, R2
        [y_pred, y_true] : arrays of predictions and true values (original scale)
    """
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for seq, feats, labels in loader:
            seq, feats = seq.to(device), feats.to(device)
            labels_all.append(labels.cpu())
            outputs = model(feats, seq)
            preds_all.append(outputs.cpu())

    # Concatenate batches
    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    # Apply inverse transform if scaler provided
    if target_scaler is not None:
        preds_all = target_scaler.inverse_transform(preds_all)
        labels_all = target_scaler.inverse_transform(labels_all)

    # Compute metrics
    mae = mean_absolute_error(labels_all, preds_all)
    rmse = mean_squared_error(labels_all, preds_all)
    r2 = r2_score(labels_all, preds_all)

    # Optional plotting
    if save_path is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(labels_all, label='y_true', marker='o')
        plt.plot(preds_all, label='y_pred', marker='x')
        plt.title("Test Set: y_true vs y_pred")
        plt.xlabel("Sample Index")
        plt.ylabel("Target Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    return {"MAE": mae, "RMSE": rmse, "R2": r2}, [preds_all, labels_all]

def get_features_targets(dataset):
    """Convert PyTorch dataset into numpy arrays for sklearn models."""
    X, y = [], []
    for i in range(len(dataset)):
        _, feats, target = dataset[i]   # ignore seq, only use features
        X.append(feats.numpy())
        y.append(target.numpy())
    return np.array(X), np.array(y)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min', restore_best_weights=True, verbose=False):
        """
        patience : number of epochs to wait after last improvement before stopping
        min_delta: minimum change in monitored value to qualify as improvement
        mode     : 'min' for loss, 'max' for metrics (e.g., accuracy, R²)
        restore_best_weights : whether to load the best model state at the end
        verbose  : print messages when stopping or improving
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_score = None
        self.best_state = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

        # define comparison operator
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best_score = np.inf
        elif mode == 'max':
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best_score = -np.inf
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, current_value, model, epoch):
        """
        current_value : current monitored value (e.g., validation loss)
        model         : PyTorch model
        epoch         : current epoch number
        """
        if self.monitor_op(current_value, self.best_score):
            self.best_score = current_value
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"Improvement at epoch {epoch+1}: best {self.mode} = {self.best_score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                if self.restore_best_weights and self.best_state is not None:
                    model.load_state_dict(self.best_state)
                    if self.verbose:
                        print(f"🔁 Restored model weights from epoch {self.best_epoch+1}")

    def reset(self):
        """Reset the early stopping state (for a new training run)."""
        self.best_score = np.inf if self.mode == 'min' else -np.inf
        self.best_state = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

if '__main__' == __name__:
    seed = 0
    main(seed=seed)


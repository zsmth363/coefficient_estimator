from hybrid_framework import HybridModel
from create_dataset import LoadExcelRegressionDataset
from features import extract_features
import numpy as np
import pandas as pd
import torch 
import os
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Split dataset into train/val/test
# -------------------------------------------------------


path_to_inputs = "C:\\Users\\Zach\\OneDrive - Clemson University\\IM_runs\\data"
path_to_targets = "C:\\Users\\Zach\\OneDrive - Clemson University\\IM_runs\\H_cons\\H_Ra_Xa_constants.csv"

input_files = [file for file in os.listdir(path_to_inputs) if file.endswith(".xlsx")]
for file in input_files:
    df = pd.read_excel(os.path.join(path_to_inputs,file))
    if df[df.columns[2]].sum() == 0:
        input_files.remove(file)
fs = 120.3

rfr = False

def main():
    # Example workflow
    dataset = LoadExcelRegressionDataset(path_to_inputs, path_to_targets, fs=fs)

    if not rfr:
        train_set = LoadExcelRegressionDataset(
            path_to_inputs, path_to_targets, fs=fs, split='train', return_sequence=True
        )

        # --- Validation / test sets (reuse train scaler) ---
        val_set = LoadExcelRegressionDataset(
            path_to_inputs, path_to_targets, fs=fs, split='val', return_sequence=True
        )
        val_set.set_scaler(train_set.scaler)

        test_set = LoadExcelRegressionDataset(
            path_to_inputs, path_to_targets, fs=fs, split='test', return_sequence=True
        )
        test_set.set_scaler(train_set.scaler)
        # X_val_scaled = scaler_X.transform(val_set)

        # # Scale targets
        # scaler_y = StandardScaler()
        # y_train_scaled = scaler_y.fit_transform(test_set.reshape(-1,1))
        # y_val_scaled = scaler_y.transform(test_set.reshape(-1,1))

        # # Convert to tensors
        # X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        # y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        # X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

        # Loaders
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_set, batch_size=32)
        test_loader  = DataLoader(test_set, batch_size=32)

        # Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridModel(seq_input_channels=3, seq_len=1000, feat_dim=len(extract_features(dataset[0][0].numpy(), fs)))
        model = model.to(device)

        # Loss + optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Train
        model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50)

        # Test
        stats,results = test(model, test_loader, device)
        print(stats)
        print(results)

    else:
        train_set, val_set, test_set = make_splits(dataset)
        rf_model = train_random_forest(train_set, val_set)
        rf_results = test_random_forest(rf_model, test_set) 
        print(rf_results)

def make_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    torch.manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test])

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total = 0.0, 0

    for seq, feats, labels in loader:
        seq, feats, labels = seq.to(device), feats.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(seq, feats)              # [batch]
        loss = criterion(outputs, labels)        # regression loss (MSE, etc.)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)

    avg_loss = total_loss / total
    return avg_loss

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for seq, feats, labels in loader:
            seq, feats, labels = seq.to(device), feats.to(device), labels.to(device)
            outputs = model(seq, feats)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            preds_all.append(outputs.cpu())
            labels_all.append(labels.cpu())

    avg_loss = total_loss / total

    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    mae = mean_absolute_error(labels_all, preds_all)
    rmse = mean_squared_error(labels_all, preds_all)
    r2 = r2_score(labels_all, preds_all)
    return avg_loss, mae, rmse, r2

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20):
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, mae, rmse, r2 = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}, "
              f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        if val_loss < best_val_loss:   # lower loss is better
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def test(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for seq, feats, labels in loader:
            seq, feats = seq.to(device), feats.to(device)
            outputs = model(seq, feats)

            preds_all.append(outputs.cpu())
            labels_all.append(labels.cpu())

    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    mae = mean_absolute_error(labels_all, preds_all)
    rmse = mean_squared_error(labels_all, preds_all)
    r2 = r2_score(labels_all, preds_all)
    plt.figure(figsize=(10, 6))
    plt.title("Testing y_test and y_pred")
    plt.plot(labels_all,label='y_test')
    plt.plot(preds_all,label='y_pred')
    plt.ylabel("H")
    plt.xlabel("Test Set Index")
    plt.legend()
    plt.savefig("C:\\Users\\Zach\\Downloads\\testing_HM.png")
    return {"MAE": mae, "RMSE": rmse, "R2": r2},[preds_all,labels_all]

def get_features_targets(dataset):
    """Convert PyTorch dataset into numpy arrays for sklearn models."""
    X, y = [], []
    for i in range(len(dataset)):
        _, feats, target = dataset[i]   # ignore seq, only use features
        X.append(feats.numpy())
        y.append(target.numpy())
    return np.array(X), np.array(y)

def train_random_forest(train_set, val_set, n_estimators=200, max_depth=None, random_state=42):
    # Prepare numpy arrays
    X_train, y_train = get_features_targets(train_set)
    X_val, y_val     = get_features_targets(val_set)

    # Fit RF model
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Evaluate on validation set
    preds = rf.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = mean_squared_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    print(f"[RandomForest] Val MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    plt.figure(figsize=(10, 6))
    plt.title("Training y_val and y_pred")
    plt.plot(y_val,label='y_val')
    plt.plot(preds,label='y_pred')
    plt.legend()
    plt.ylabel("H")
    plt.xlabel("Training Set Index")
    plt.savefig("C:\\Users\\Zach\\Downloads\\training_RF.png")
    return rf

def test_random_forest(rf_model, test_set):
    X_test, y_test = get_features_targets(test_set)
    preds = rf_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    plt.figure(figsize=(10, 6))
    plt.title("Testing y_test and y_pred")
    plt.plot(y_test,label='y_test')
    plt.plot(preds,label='y_pred')
    plt.ylabel("H")
    plt.xlabel("Test Set Index")
    plt.legend()
    plt.savefig("C:\\Users\\Zach\\Downloads\\testing_RF.png")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


if '__main__' == __name__:
    main()


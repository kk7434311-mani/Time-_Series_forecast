#!/usr/bin/env python3
"""
project.py
Single-file project for:
"Advanced Time Series Forecasting with Neural State Space Models (NSSMs)"

Features:
- Synthetic multi-seasonal multivariate dataset generation
- TimeSeries dataset/loader
- NSSM (GRU-based) model
- LSTM baseline model
- Training loop (train/val/test)
- Save best models and results.json
- Plot predictions vs actual
- CLI arguments for easy control

Run:
    python project.py --epochs 10
"""

import argparse
import json
from pathlib import Path
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
import sys # Import sys for checking interactive mode

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm

# -----------------------
# Utilities / Metrics
# -----------------------
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def safe_mape(y_true, y_pred):
    try:
        return float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        y_true = np.asarray(y_true)
        denom = np.where(y_true == 0, 1e-8, y_true)
        return float(np.mean(np.abs((y_true - np.asarray(y_pred)) / denom)))

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# -----------------------
# Data generation
# -----------------------
def generate_synthetic(n=3000, seed=42):
    """
    Generate synthetic time series with 3 seasonal components, trend, noise,
    and two exogenous features (feat1, feat2).
    """
    np.random.seed(seed)
    t = np.arange(n).astype(float)

    seasonal_daily = 10.0 * np.sin(2 * np.pi * t / 24.0)
    seasonal_weekly = 5.0 * np.sin(2 * np.pi * t / 168.0)
    seasonal_monthly = 2.0 * np.sin(2 * np.pi * t / 720.0)
    trend = 0.0008 * t
    noise = np.random.normal(0, 1.0, size=n)

    value = seasonal_daily + seasonal_weekly + seasonal_monthly + trend + noise
    feat1 = np.cos(t / 50.0)
    feat2 = np.sin(t / 100.0)

    df = pd.DataFrame({
        "time": pd.date_range("2000-01-01", periods=n, freq="H"),
        "value": value,
        "feat1": feat1,
        "feat2": feat2
    })
    return df

# -----------------------
# Dataset class
# -----------------------
class TimeSeriesDataset(Dataset):
    """
    Sliding window dataset. data is numpy array shape (T, features), where column 0 is target.
    __getitem__ returns (X_seq, y_scalar)
    """
    def __init__(self, data: np.ndarray, seq_len: int = 48):
        self.seq_len = seq_len
        self.data = data.astype("float32")
        self.X = []
        self.y = []
        n = len(self.data)
        for i in range(n - seq_len):
            self.X.append(self.data[i:i+seq_len])
            self.y.append(self.data[i+seq_len, 0])
        self.X = np.array(self.X)
        self.y = np.array(self.y).astype("float32")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# -----------------------
# Models
# -----------------------
class NSSM(nn.Module):
    """
    Simple NSSM: GRU transition + observation head
    """
    def __init__(self, input_dim=3, latent_dim=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=latent_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout)
        mid = max(4, latent_dim // 2)
        self.obs = nn.Sequential(
            nn.Linear(latent_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, 1)
        )

    def forward(self, x):
        # x: B x seq_len x input_dim
        _, h = self.gru(x)
        h_last = h[-1]
        out = self.obs(h_last).squeeze(-1)
        return out

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        mid = max(4, hidden_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, 1)
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_last = h[-1]
        return self.head(h_last).squeeze(-1)

# -----------------------
# Train / Eval helpers
# -----------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n += X.size(0)
    return total_loss / max(1, n)

def evaluate_model(model, loader, device):
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            p = model(X)
            ys.append(y.cpu().numpy())
            ps.append(p.cpu().numpy())
    if len(ys) == 0:
        return np.array([]), np.array([])
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    return ys, ps

# -----------------------
# Prepare data splits
# -----------------------
def prepare_splits(df, seq_len=48, test_ratio=0.15, val_ratio=0.1):
    values = df[['value', 'feat1', 'feat2']].values.astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    n = len(scaled)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_end = n - test_size - val_size

    train = scaled[:train_end]
    val = scaled[train_end:train_end+val_size]
    test = scaled[train_end+val_size:]

    train_ds = TimeSeriesDataset(train, seq_len=seq_len)
    # validation dataset needs seq_len history, so combine tail of train with val
    val_context = np.vstack([train[-seq_len:], val]) if len(val) > 0 else train
    val_ds = TimeSeriesDataset(val_context, seq_len=seq_len)
    test_context = np.vstack([train[-seq_len:], val, test]) if len(val) > 0 else np.vstack([train[-seq_len:], test])
    test_ds = TimeSeriesDataset(test_context, seq_len=seq_len)
    return train_ds, val_ds, test_ds, scaler

# -----------------------
# Main
# -----------------------
def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", default=True, help="Generate synthetic data (default True)")
    parser.add_argument("--n", type=int, default=3000, help="Length of synthetic series")
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")

    # Check if running in an interactive environment (like Colab/Jupyter)
    # This prevents argparse from trying to parse kernel-specific arguments.
    if 'ipykernel' in sys.modules:
        args = parser.parse_args([]) # Pass an empty list to parse_args
    else:
        args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # generate or load
    if args.generate:
        df = generate_synthetic(n=args.n, seed=args.seed)
        df.to_csv(outdir / "generated_dataset.csv", index=False)
    else:
        # fallback: try to load from outputs/generated_dataset.csv
        df = pd.read_csv(outdir / "generated_dataset.csv")

    train_ds, val_ds, test_ds, scaler = prepare_splits(df, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = train_ds.X.shape[2]
    nssm = NSSM(input_dim=input_dim, latent_dim=args.latent).to(device)
    lstm = LSTMForecaster(input_dim=input_dim, hidden_dim=args.lstm_hidden).to(device)

    crit = nn.MSELoss()
    nssm_opt = torch.optim.Adam(nssm.parameters(), lr=args.lr)
    lstm_opt = torch.optim.Adam(lstm.parameters(), lr=args.lr)

    best_val_rmse = float("inf")
    results = {"nssm": {}, "lstm": {}}

    print("Starting training on device:", device)
    for epoch in range(1, args.epochs + 1):
        nssm_train_loss = train_epoch(nssm, train_loader, nssm_opt, crit, device)
        lstm_train_loss = train_epoch(lstm, train_loader, lstm_opt, crit, device)

        yv_n, pv_n = evaluate_model(nssm, val_loader, device)
        yv_l, pv_l = evaluate_model(lstm, val_loader, device)

        # compute metrics if available
        if len(yv_n) > 0:
            nssm_rmse = rmse(yv_n, pv_n)
            nssm_mape = safe_mape(yv_n, pv_n)
        else:
            nssm_rmse = nssm_mape = None

        if len(yv_l) > 0:
            lstm_rmse = rmse(yv_l, pv_l)
            lstm_mape = safe_mape(yv_l, pv_l)
        else:
            lstm_rmse = lstm_mape = None

        print(f"Epoch {epoch}/{args.epochs} | NSSM train loss {nssm_train_loss:.4f} | LSTM train loss {lstm_train_loss:.4f}")
        if nssm_rmse is not None:
            print(f"  >> Validation: NSSM RMSE {nssm_rmse:.4f} MAPE {nssm_mape:.4f} | LSTM RMSE {lstm_rmse:.4f} MAPE {lstm_mape:.4f}")

        # save best by NSSM val rmse
        if nssm_rmse is not None and nssm_rmse < best_val_rmse:
            best_val_rmse = nssm_rmse
            torch.save(nssm.state_dict(), str(outdir / "nssm_best.pth"))
            torch.save(lstm.state_dict(), str(outdir / "lstm_best.pth"))
            results["nssm"]["val_rmse"] = float(nssm_rmse)
            results["nssm"]["val_mape"] = float(nssm_mape)
            results["lstm"]["val_rmse"] = float(lstm_rmse)
            results["lstm"]["val_mape"] = float(lstm_mape)
            print("Saved best models to", outdir)

    # Load best and evaluate on test
    if (outdir / "nssm_best.pth").exists():
        nssm_best = NSSM(input_dim=input_dim, latent_dim=args.latent)
        lstm_best = LSTMForecaster(input_dim=input_dim, hidden_dim=args.lstm_hidden)
        nssm_best.load_state_dict(torch.load(outdir / "nssm_best.pth", map_location=device))
        lstm_best.load_state_dict(torch.load(outdir / "lstm_best.pth", map_location=device))
        nssm_best.to(device)
        lstm_best.to(device)

        yt_n, pt_n = evaluate_model(nssm_best, test_loader, device)
        yt_l, pt_l = evaluate_model(lstm_best, test_loader, device)

        if len(yt_n) > 0:
            results["nssm"]["test_rmse"] = rmse(yt_n, pt_n)
            results["nssm"]["test_mape"] = safe_mape(yt_n, pt_n)
        if len(yt_l) > 0:
            results["lstm"]["test_rmse"] = rmse(yt_l, pt_l)
            results["lstm"]["test_mape"] = safe_mape(yt_l, pt_l)

        print("Test results:")
        print("NSSM:", results["nssm"].get("test_rmse"), results["nssm"].get("test_mape"))
        print("LSTM:", results["lstm"].get("test_rmse"), results["lstm"].get("test_mape"))

        # Save results JSON
        save_json(results, outdir / "results.json")

        # Plot first N points of test predictions vs actual
        try:
            idx = slice(0, 400)
            plt.figure(figsize=(12, 4))
            if len(yt_n) > 0:
                plt.plot(yt_n[idx], label="Actual (test)", linewidth=1)
                plt.plot(pt_n[idx], label="NSSM Pred", linewidth=1)
            if len(yt_l) > 0:
                plt.plot(pt_l[idx], label="LSTM Pred", linewidth=1, alpha=0.9)
            plt.legend()
            plt.title("Actual vs Predictions (test windows)")
            plt.tight_layout()
            plot_path = outdir / "pred_vs_actual.png"
            plt.savefig(plot_path)
            plt.close()
            print("Saved plot to", plot_path)
        except Exception as e:
            print("Plotting failed:", e)
    else:
        print("No best model saved (maybe validation not available). Check training logs.")

    # Additionally create a simple SARIMAX baseline result on raw target (not scaled)
    try:
        # Fit SARIMAX on the original value column for a quick baseline
        sarimax_model = sm.tsa.statespace.SARIMAX(df['value'], order=(3,1,2), seasonal_order=(1,0,1,24))
        sar_res = sarimax_model.fit(disp=False)
        # forecast the last len(test) steps
        test_len = len(test_ds.y) if hasattr(test_ds, "y") else 0
        if test_len > 0:
            fc = sar_res.get_prediction(start=len(df)-test_len, end=len(df)-1)
            fc_mean = fc.predicted_mean.values
            # Need to compare with test true values (unscaled)
            # extract the actual values in the last test_len points
            actual = df['value'].values[-test_len:]
            sar_rmse = rmse(actual, fc_mean)
            sar_mape = safe_mape(actual, fc_mean)
            results["sarimax"] = {"test_rmse": float(sar_rmse), "test_mape": float(sar_mape)}
            save_json(results, outdir / "results.json")
            print("SARIMAX baseline:", sar_rmse, sar_mape)
    except Exception as e:
        print("SARIMAX baseline failed (ok):", e)

    # Save README-like summary for submission convenience
    readme_text = f"""
Project: Advanced Time Series Forecasting with NSSM
Files: project.py (single-file)
Outputs: outputs/results.json, outputs/nssm_best.pth, outputs/lstm_best.pth, outputs/pred_vs_actual.png
Uploaded screenshot path: /mnt/data/Screenshot_20251122_145227.jpg
"""
    with open(outdir / "submission_note.txt", "w") as f:
        f.write(readme_text.strip())
    print("Wrote submission_note.txt with helpful summary.")

if __name__ == "__main__":
    main_cli()

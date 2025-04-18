import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn

print("Starting traffic prediction with PyTorch GRU with Attention...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ========== CONFIG ==========
csv_folder = "/nas/eclairnas01/users/shefalit/working/code/csv_files"
train_dates = ["20210108", "20210109", "20210110", "20210111", "20210112", "20210113"]
test_date = "20210114"
window_size = 12
feature_cols = ['packet_count', 'flow_count', 'tcp_count']
hidden_dim = 128
epochs = 50
learning_rate = 0.0005
batch_size = 128
dropout = 0.3

# ========== DATA LOADING ==========
def load_data(dates):
    all_data = []
    for date in dates:
        fname = f"{date[-2:]}_preprocessed_data.csv.gz"
        path = os.path.join(csv_folder, fname)
        print(f"Loading file: {path}")
        df = pd.read_csv(path, compression='gzip')
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y%m%d-%H%M%S")
        df = df.sort_values("timestamp")
        print(f"  Loaded {len(df)} rows. First timestamp: {df['timestamp'].iloc[0]}")
        all_data.append(df)
    return pd.concat(all_data)

print("Loading training data...")
train_df = load_data(train_dates)
print("Loading test data...")
test_df = load_data([test_date])

all_df = pd.concat([train_df, test_df])
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(12, 4))
plt.plot(all_df['timestamp'], all_df['packet_count'], color='tab:gray')
plt.title("Traffic Trend from 08 to 14 Jan")
plt.xlabel("Time")
plt.ylabel("packet_count")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/traffic_trend_full.png")
plt.close()

# ========== NORMALIZATION ==========
scalers = {}
print("Normalizing features...")
for col in feature_cols:
    mean = train_df[col].mean()
    std = train_df[col].std()
    std = std if std > 1e-6 else 1.0
    scalers[col] = (mean, std)
    train_df[col] = (train_df[col] - mean) / std
    test_df[col] = (test_df[col] - mean) / std
    print(f"  {col} -> mean: {mean:.2f}, std: {std:.2f}")

# ========== UPSAMPLING PEAK TRAFFIC REGIONS ==========
def upsample_peaks(df, column, threshold_factor=1.5):
    threshold = df[column].mean() + threshold_factor * df[column].std()
    peak_rows = df[df[column] > threshold]
    df_upsampled = pd.concat([df, peak_rows] * 2, ignore_index=True)
    return df_upsampled.sort_values("timestamp")

train_df = upsample_peaks(train_df, 'packet_count')

# ========== SLIDING WINDOW ==========
def sliding_window(arr, size):
    X, y = [], []
    for i in range(len(arr) - size):
        X.append(arr[i:i+size])
        y.append(arr[i+size][0])  # only packet_count target
    return np.array(X), np.array(y)

print("Creating sliding windows...")
train_X, train_y = sliding_window(train_df[feature_cols].values, window_size)
test_X, test_y = sliding_window(test_df[feature_cols].values, window_size)
print(f"Train samples: {len(train_X)}, Test samples: {len(test_X)}")

train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1).to(device)
test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(-1).to(device)

# ========== GRU MODEL with ATTENTION ==========
class AttentionGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_weights = torch.softmax(self.attn(gru_out).squeeze(-1), dim=1).unsqueeze(-1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        out = self.fc(context)
        return out

print("Initializing Attention GRU model...")
model = AttentionGRU(len(feature_cols), hidden_dim, 1).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# ========== TRAINING ==========
dataset = TensorDataset(train_X, train_y)
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

train_losses, val_losses = [], []
best_val_loss = float('inf')

print("Training model...")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1:03d}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")

# ========== EVALUATION ==========
print("Evaluating on test set...")
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
preds = []
with torch.no_grad():
    for i in range(len(test_X)):
        X = test_X[i].unsqueeze(0)
        y_pred = model(X)
        preds.append(y_pred.squeeze().cpu().item())

preds = np.array(preds).reshape(-1, 1)
actuals = test_y.cpu().numpy()

mean, std = scalers['packet_count']
preds = preds * std + mean
actuals = actuals * std + mean

def compute_accuracy(preds, actuals, threshold=0.10):
    relative_error = np.abs(preds - actuals) / np.maximum(actuals, 1)
    correct = (relative_error <= threshold).astype(int)
    accuracy = correct.mean()
    return accuracy

acc = compute_accuracy(preds, actuals)
print(f"Test Accuracy (within 10% tolerance): {acc * 100:.2f}%")

# ========== PLOTTING ==========
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/loss_curve.png")
plt.close()

timestamps = test_df['timestamp'].iloc[window_size:].reset_index(drop=True)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 6), sharex=True)
ymin = min(np.min(actuals), np.min(preds))
ymax = max(np.max(actuals), np.max(preds))

axes[0].plot(timestamps, actuals, label='Actual', color='tab:orange')
axes[0].set_title("packet_count - Actual")
axes[0].set_ylabel("packet_count")
axes[0].grid(True)
axes[0].set_ylim([ymin, ymax])
axes[0].xaxis.set_major_locator(mdates.HourLocator(interval=2))
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

axes[1].plot(timestamps, preds, label='Predicted', color='tab:blue')
axes[1].set_title("packet_count - Predicted")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("packet_count")
axes[1].grid(True)
axes[1].set_ylim([ymin, ymax])
axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/packet_count_prediction.png")
plt.close()

print("Done! All results saved in the 'results/' directory.")

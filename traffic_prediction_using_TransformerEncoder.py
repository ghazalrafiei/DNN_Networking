import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import math
import torch.nn.functional as F

print("Starting traffic prediction with PyTorch GRU with Attention...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ========== CONFIG ==========
csv_folder = "/scratch1/zhilihu/566project/DNN_Networking/csv_files"
train_dates = ["20210108", "20210109", "20210110", "20210111", "20210112", "20210113"]
test_date = "20210114"
window_size = 24
feature_cols = ['packet_count', 'flow_count', 'tcp_count']
hidden_dim = 64
epochs = 50
learning_rate = 0.0001
batch_size = 128
dropout = 0.4
nhead = 4
num_layers = 2




print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

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
plt.savefig("results/traffic_trend_full_transformer.png")
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
def upsample_peaks(df, column, threshold_factor=1.0):
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

# ========== TRANSFORMER-BASED MODEL ==========

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预先计算位置编码矩阵
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 d_model=128,
                 nhead=nhead,
                 num_layers=num_layers,
                 dim_feedforward=256,
                 dropout=0.1,
                 output_dim=1):
        super().__init__()
        # 1) 输入映射
        self.input_proj = nn.Linear(input_dim, d_model)
        # 2) 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # 3) Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 使输入维度为 (batch, seq, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # 4) 输出映射
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        returns: (batch_size, output_dim)
        """
        # 投射到 d_model 维度
        x = self.input_proj(x)                    # (B, S, d_model)
        # 加上位置编码
        x = self.pos_encoder(x)                   # (B, S, d_model)
        # Transformer 编码
        x = self.transformer_encoder(x)           # (B, S, d_model)
        # 取最后一步特征，映射到 output_dim
        out = self.fc_out(x[:, -1, :])            # (B, output_dim)
        return out

print("Initializing Transformer model...")
model = TimeSeriesTransformer(
    input_dim=len(feature_cols),
    d_model=hidden_dim,         # 可复用原来的 hidden_dim=128
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=hidden_dim*2,
    dropout=dropout,
    output_dim=1
).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    verbose=True
)
# ========== END OF TRANSFORMER-BASED MODEL ==========

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
plt.savefig("results/loss_curve_transformer.png")
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
plt.savefig("results/packet_count_prediction_transformer.png")
plt.close()

print("Done! All results saved in the 'results/' directory.")

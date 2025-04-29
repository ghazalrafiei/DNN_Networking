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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

print("Starting traffic prediction with PyTorch GRU with Attention...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ========== CONFIG ==========

delta_seconds   = 60
horizon_seconds = 140
csv_folder = "/scratch1/zhilihu/566project/DNN_Networking/csv_files"
train_dates = ["20210108", "20210109", "20210110", "20210111", "20210112", "20210113"]
test_date = "20210114"
window_size = 24
feature_cols = ['packet_count', 'flow_count', 'tcp_count']
hidden_dim = 256
epochs = 50
learning_rate = 5e-5
batch_size = 64
dropout = 0.5

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


sampling_interval = (train_df['timestamp'].iloc[1] - train_df['timestamp'].iloc[0]).total_seconds()
delta_steps = int(delta_seconds / sampling_interval)
num_steps   = math.ceil(horizon_seconds / delta_seconds)


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
def sliding_window_multi(arr, window_size, delta_steps, num_steps):
    X, Y = [], []
    max_i = len(arr) - window_size - delta_steps * num_steps
    for i in range(max_i):
        X.append(arr[i:i+window_size])
        y_steps = [ arr[i + window_size + k*delta_steps][0]
                    for k in range(1, num_steps+1) ]
        Y.append(y_steps)
    return np.array(X), np.array(Y)

print("Creating multi-step sliding windows...")
train_X, train_y = sliding_window_multi(
    train_df[feature_cols].values,
    window_size, delta_steps, num_steps
)
test_X, test_y = sliding_window_multi(
    test_df[feature_cols].values,
    window_size, delta_steps, num_steps
)
print(f"Train samples: {len(train_X)}, each y has shape ({num_steps},)")
print(f"Test  samples: {len(test_X)}, each y has shape ({num_steps},)")

train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
test_X  = torch.tensor(test_X,  dtype=torch.float32).to(device)
test_y  = torch.tensor(test_y,  dtype=torch.float32).to(device)

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
    def __init__(self, input_dim, d_model, nhead, num_layers,
                 dim_feedforward, dropout, output_steps):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers)
        # 这里 output_steps 替换原来单一 output_dim
        self.fc_out = nn.Linear(d_model, output_steps)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # 直接输出 shape (B, num_steps)
        return self.fc_out(x[:, -1, :])

print("Initializing Transformer model...")
model = TimeSeriesTransformer(
    input_dim=len(feature_cols),
    d_model=hidden_dim,
    nhead=16,
    num_layers=4,
    dim_feedforward=hidden_dim*2,
    dropout=dropout,
    output_steps=num_steps
).to(device)

criterion = nn.SmoothL1Loss(beta=0.5)
#criterion = nn.L1Loss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer,
#    mode='min',
#    factor=0.5,
#    patience=2,
#    verbose=True
#)

optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-5
)



# ========== END OF TRANSFORMER-BASED MODEL ==========

# ========== TRAINING ==========
patience_counter = 0
early_stop_patience = 50
dataset = TensorDataset(train_X, train_y)
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)



total_steps = epochs * len(train_loader)
warmup_steps = int(0.1 * total_steps)
def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    # 余弦退火
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)

train_losses, val_losses = [], []
best_val_loss = float('inf')
scaler = GradScaler()

print("Training model...")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        with autocast():
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
#        y_pred = model(X_batch)
#        loss = criterion(y_pred, y_batch)
#        loss.backward()
#        optimizer.step()
        scheduler.step()
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
#    scheduler.step(val_loss)

    print(f"Epoch {epoch+1:03d}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f},"f" LR: {optimizer.param_groups[0]['lr']:.2e}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model_time_transformer.pt")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping.")
            break

# ========== EVALUATION ==========
print("Evaluating on test set...")
model.load_state_dict(torch.load("best_model_time_transformer.pt"))
model.eval()

test_X_cpu, test_y_cpu = test_X.detach().cpu(), test_y.detach().cpu()

test_dataset  = TensorDataset(test_X_cpu, test_y_cpu)
test_loader   = DataLoader(
    test_dataset,
    batch_size=batch_size,    # 同训练时的 batch_size，或根据显存调小
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

all_preds = []
with torch.no_grad():
    for Xb, _ in test_loader:
        Xb = Xb.to(device)
        preds_batch = model(Xb)              # (B, num_steps) 或 (B,1)
        all_preds.append(preds_batch.detach().cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)  # (N, num_steps) 或 (N,1)

mean, std = scalers['packet_count']
preds   = all_preds * std + mean             # shape (N, num_steps) 或 (N,1)
all_actuals = test_y.cpu().numpy() * std + mean


pred60  = preds[:, 0]
pred120 = preds[:, 1]
pred180 = preds[:, 2]
actual   = all_actuals[:, 0]

preds_steps  = np.stack([pred60, pred120, pred180], axis=1)
actuals_steps = all_actuals
last_actuals = test_X.detach().cpu().numpy()[:, -1, 0] * std + mean
alpha     = 1.5
threshold = mean + alpha * std
N, S = preds_steps.shape
dir_correct   = np.zeros((N, S), dtype=bool)
spike_correct = np.zeros((N, S), dtype=bool)

for k in range(S):
    y_pred_k = preds_steps[:, k]
    y_true_k = actuals_steps[:, k]
    prev_pred = last_actuals if k == 0 else preds_steps[:, k-1]
    prev_true = last_actuals if k == 0 else actuals_steps[:, k-1]

    # 方向一致性
    dir_pred = np.sign(y_pred_k - prev_pred)
    dir_true = np.sign(y_true_k - prev_true)
    dir_correct[:, k] = (dir_pred == dir_true)

    # 峰值事件一致性
    spike_pred = (y_pred_k > threshold)
    spike_true = (y_true_k > threshold)
    spike_correct[:, k] = (spike_pred == spike_true)

# —— 3) 统计 per-step 和 overall 准确率 —— 
dir_acc_step   = dir_correct.mean(axis=0)
spike_acc_step = spike_correct.mean(axis=0)
dir_acc_overall   = dir_correct.mean()
spike_acc_overall = spike_correct.mean()

# —— 4) 打印结果 —— 
print(f"Overall Direction Accuracy: {dir_acc_overall*100:.2f}%")
print(f"Overall Spike     Accuracy: {spike_acc_overall*100:.2f}% (α={alpha})\n")

for k in range(S):
    t_sec = (k+1) * delta_seconds
    print(f"Step {k+1:02d} @ t+{t_sec}s | "
          f"DirAcc = {dir_acc_step[k]*100:.2f}%  | "
          f"SpikeAcc = {spike_acc_step[k]*100:.2f}%")
          

def compute_accuracy_multi(preds, actuals, threshold=0.10):
    """
    preds: np.array, shape (N, S)
    actuals: np.array, shape (N, S)
    返回 overall_acc, acc_per_step
    overall_acc: 所有 N×S 个预测的平均正确率
    acc_per_step: 长度 S 的数组，每一步的正确率
    """
    # 计算相对误差
    rel_err = np.abs(preds - actuals) / np.maximum(actuals, 1)
    # 小于等于阈值即视为“正确”
    correct = (rel_err <= threshold)
    
    # 每个 step 上的准确率
    acc_per_step = correct.mean(axis=0)        # shape (S,)
    # 所有预测点的整体准确率
    overall_acc   = correct.mean()             # scalar
    
    return overall_acc, acc_per_step

overall_acc, acc_steps = compute_accuracy_multi(preds, all_actuals)
print(f"Test Accuracy (overall within 10% tolerance): {overall_acc * 100:.2f}%")
print("Accuracy per step:")
for i, acc in enumerate(acc_steps):
    print(f"  Step {i+1}: {acc * 100:.2f}%")
    
mae_overall  = np.mean(np.abs(preds - all_actuals))
rmse_overall = np.sqrt(np.mean((preds - all_actuals)**2))

print(f"Overall MAE:  {mae_overall:.2f}")
print(f"Overall RMSE: {rmse_overall:.2f}")

# 如果想看每个 step（checkpoint）单独的 MAE/RMSE：
mae_per_step  = np.mean(np.abs(preds - all_actuals), axis=0)    # shape (S,)
rmse_per_step = np.sqrt(np.mean((preds - all_actuals)**2, axis=0))

for k, (m, r) in enumerate(zip(mae_per_step, rmse_per_step), start=1):
    print(f" Step {k:02d}: MAE = {m:.2f}, RMSE = {r:.2f}")


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
plt.savefig("results/timestamp_loss_curve.png")
plt.close()

timestamps = (
    test_df['timestamp']
      .iloc[window_size : window_size + len(actual)]
      .reset_index(drop=True)
)
fig, axes = plt.subplots(
    nrows=4, ncols=1,
    figsize=(14, 10),
    sharex=True
)



# 第 1 行：实际值
axes[0].plot(timestamps, actual, color='gray', linewidth=1)
axes[0].set_title("Actual 14 Jan")
axes[0].set_ylabel("packet_count")
axes[0].grid(True)

# 第 2 行：t + 60s 预测 vs 实际
axes[1].plot(timestamps, actual, color='lightgray', linewidth=1, label='Actual')
axes[1].plot(timestamps, pred60, color='tab:blue', linewidth=1, label='Pred @ t+60s')
axes[1].set_title("Prediction at t+60s vs Actual")
axes[1].set_ylabel("packet_count")
axes[1].grid(True)
axes[1].legend(loc='upper left')

# 第 3 行：t + 120s 预测 vs 实际
axes[2].plot(timestamps, actual, color='lightgray', linewidth=1, label='Actual')
axes[2].plot(timestamps, pred120, color='tab:green', linewidth=1, label='Pred @ t+120s')
axes[2].set_title("Prediction at t+120s vs Actual")
axes[2].set_ylabel("packet_count")
axes[2].grid(True)
axes[2].legend(loc='upper left')

# 第 4 行：t + 180s 预测 vs 实际
axes[3].plot(timestamps, actual, color='lightgray', linewidth=1, label='Actual')
axes[3].plot(timestamps, pred180, color='tab:red', linewidth=1, label='Pred @ t+180s')
axes[3].set_title("Prediction at t+180s vs Actual")
axes[3].set_ylabel("packet_count")
axes[3].grid(True)
axes[3].legend(loc='upper left')

# X 轴格式化为小时:分钟
axes[3].xaxis.set_major_locator(mdates.HourLocator(interval=2))
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)

plt.xlabel("Time")
plt.tight_layout()
plt.savefig("results/multi_horizon_plot.png")

print("Done! All results saved in the 'results/' directory.")
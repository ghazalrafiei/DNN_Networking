import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_curve, auc,
    precision_recall_curve
)

print("Starting traffic prediction with PyTorch GRU+Attention (60s history → 60/120/180s ahead)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ========== CONFIG ==========
csv_folder   = "/nas/eclairnas01/users/shefalit/working/code/csv_files"
train_dates  = ["20210108","20210109","20210110","20210111","20210112","20210113"]
val_date     = "20210113"
test_date    = "20210114"
window_size  = 60    # 60 s history
horizons     = [60, 120, 180]
feature_cols = ['packet_count','flow_count','tcp_count']
hidden_dim   = 128
epochs       = 100
lr           = 5e-4
batch_size   = 128
n_lag        = 6    # for RF hourly

os.makedirs('results', exist_ok=True)

# ========== HELPERS ==========
def load_data(dates):
    dfs = []
    for d in dates:
        path = os.path.join(csv_folder, f"{d[-2:]}_preprocessed_data.csv.gz")
        df   = pd.read_csv(path, compression='gzip')
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y%m%d-%H%M%S")
        dfs.append(df.sort_values('timestamp').reset_index(drop=True))
    return pd.concat(dfs, ignore_index=True)

def sliding_window_multi(arr, window, horizons):
    X, y = [], []
    mh = max(horizons)
    for i in range(len(arr) - window - mh):
        X.append(arr[i:i+window])
        y.append([arr[i+window+h, 0] for h in horizons])
    return np.array(X), np.array(y)

def load_hourly_peaks(date):
    path = os.path.join(csv_folder, f"{date[-2:]}_preprocessed_data.csv.gz")
    df = pd.read_csv(path, compression='gzip')
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d-%H%M%S")
    df = df.set_index("timestamp").sort_index()
    start = pd.to_datetime(date)
    end   = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df.loc[start:end]
    s = df["packet_count"].resample("H").max()
    if len(s) != 24:
        raise RuntimeError(f"{date}: expected 24 hourly points, got {len(s)}")
    return s

def make_lag_df(series, n_lag):
    df = pd.DataFrame({"peak": series})
    for lag in range(1, n_lag+1):
        df[f"lag_{lag}"] = df["peak"].shift(lag)
    return df.dropna()

# ========== 1) PREPARE TRAIN+VAL FOR GRU ==========
df_full = load_data(train_dates)
scalers  = {}
for c in feature_cols:
    μ,σ = df_full[c].mean(), df_full[c].std().clip(min=1e-6)
    scalers[c] = (μ,σ)
    df_full[c] = (df_full[c] - μ) / σ

# upsample peaks
thr   = df_full['packet_count'].mean() + 1.5 * df_full['packet_count'].std()
peaks = df_full[df_full['packet_count'] > thr]
df_full = pd.concat([df_full, peaks]*2, ignore_index=True).sort_values('timestamp')

arr = df_full[feature_cols].values.astype(np.float32)
X_np, y_np = sliding_window_multi(arr, window_size, horizons)
print("Train+Val samples:", X_np.shape[0])
X = torch.tensor(X_np, dtype=torch.float32).to(device)
y = torch.tensor(y_np, dtype=torch.float32).to(device)

# ========== 2) DEFINE MODEL ==========
class AttentionGRU(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.gru  = nn.GRU(in_dim, hid_dim, batch_first=True)
        self.attn = nn.Linear(hid_dim, 1)
        self.fc   = nn.Linear(hid_dim, out_dim)
    def forward(self, x):
        g,_ = self.gru(x)
        w   = torch.softmax(self.attn(g).squeeze(-1), dim=1).unsqueeze(-1)
        c   = (w * g).sum(1)
        return self.fc(c)

model = AttentionGRU(len(feature_cols), hidden_dim, len(horizons)).to(device)

# ========== 3) TRAIN & VALIDATE ==========
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True)

dataset    = TensorDataset(X, y)
val_size   = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
t_ds, v_ds = random_split(dataset, [train_size, val_size])
train_ld   = DataLoader(t_ds, batch_size=batch_size, shuffle=True)
val_ld     = DataLoader(v_ds, batch_size=batch_size)

train_losses, val_losses = [], []
best_val = float('inf')

for ep in range(1, epochs+1):
    model.train()
    tot_tr = 0
    for xb, yb in train_ld:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        tot_tr += loss.item() * xb.size(0)
    tr_loss = tot_tr / len(train_ld.dataset)
    train_losses.append(tr_loss)

    model.eval()
    tot_vl = 0
    with torch.no_grad():
        for xb, yb in val_ld:
            tot_vl += criterion(model(xb), yb).item() * xb.size(0)
    vl_loss = tot_vl / len(val_ld.dataset)
    val_losses.append(vl_loss)

    scheduler.step(vl_loss)
    print(f"Epoch {ep:02d} | Train L1 {tr_loss:.4f} | Val L1 {vl_loss:.4f}")

    if vl_loss < best_val:
        best_val = vl_loss
        torch.save(model.state_dict(), 'results/best_attn.pt')

print(f"\n→ Best Train L1: {min(train_losses):.4f}")
print(f"→ Best Val   L1: {best_val:.4f}")

# ========== 3b) PLOT EPOCH vs LOSS ==========
plt.figure(figsize=(8,4))
plt.plot(range(1, epochs+1), train_losses, label='Train L1')
plt.plot(range(1, epochs+1), val_losses,   label='Val   L1')
plt.title('Training & Validation Loss vs Epoch')
plt.xlabel('Epoch'); plt.ylabel('L1 Loss')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig('results/loss_vs_epochs.png')
plt.close()

# ========== 4) TEST MULTI‐HORIZON FORECAST ==========
print("\nForecasting multi-horizon on 14 Jan…")
test_df_full = load_data([test_date])
test_df      = test_df_full.copy()
for c in feature_cols:
    μ,σ = scalers[c]
    test_df[c] = (test_df[c] - μ) / σ

t_arr, ty_np = sliding_window_multi(
    test_df[feature_cols].values.astype(np.float32),
    window_size, horizons
)
tX = torch.tensor(t_arr, dtype=torch.float32).to(device)

model.load_state_dict(torch.load('results/best_attn.pt'))
model.eval()
with torch.no_grad():
    preds = model(tX).cpu().numpy()

μ,σ = scalers['packet_count']
preds   = preds * σ + μ
actuals = ty_np * σ + μ

for i,h in enumerate(horizons):
    mae  = mean_absolute_error(actuals[:,i], preds[:,i])
    rmse = mean_squared_error(actuals[:,i], preds[:,i], squared=False)
    print(f"Test @ t+{h:3d}s → MAE = {mae:7.2f}, RMSE = {rmse:7.2f}")

start         = window_size + max(horizons)
times_act     = test_df_full['timestamp'].iloc[start:].reset_index(drop=True)
actual_series = test_df_full['packet_count'].iloc[start:].reset_index(drop=True)

# ========== 5) TRAIN & PREDICT RF HOURLY SPIKES ==========
s_tr = pd.concat([load_hourly_peaks(d) for d in train_dates[:-1]])
s_va = load_hourly_peaks(val_date)
s_te = load_hourly_peaks(test_date)

df_tr = make_lag_df(s_tr, n_lag)
df_va = make_lag_df(pd.concat([s_tr.tail(n_lag), s_va]), n_lag)
df_te = make_lag_df(pd.concat([pd.concat([s_tr, s_va]).tail(n_lag), s_te]), n_lag)

X_tr, y_tr = df_tr.drop("peak",1).values, df_tr["peak"].values
X_va, y_va = df_va.drop("peak",1).values, df_va["peak"].values
X_te, y_te = df_te.drop("peak",1).values, df_te["peak"].values
times_te   = df_te.index

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_tr, y_tr)
y_te_pred = rf.predict(X_te)

# ========== 6) PLOT MULTI‐HORIZON WITH STACKED SPIKES ==========
times_act_idx = pd.DatetimeIndex(times_act)
pos = times_act_idx.get_indexer(times_te)

fig, axs = plt.subplots(len(horizons)+1, 1, figsize=(14,10), sharex=True)
axs[0].plot(times_act, actual_series, color='tab:gray')
axs[0].set_title('Actual 14 Jan (1 Hz)'); axs[0].grid(False)

colors = ['tab:blue','tab:green','tab:red']
for i,h in enumerate(horizons):
    ax = axs[i+1]
    ax.plot(times_act, actual_series, color='tab:gray', alpha=0.3)
    ax.plot(times_act, preds[:,i],    color=colors[i], label=f'GRU Pred @ t+{h}s')
    bottom_ys = preds[pos, i]
    ax.vlines(times_te,
              ymin=bottom_ys,
              ymax=y_te_pred,
              color=colors[i], linestyle='-', linewidth=1,
              label='RF Hourly Spike' if i==0 else None)
    ax.set_title(f'Forecast t+{h}s vs Actual')
    ax.legend(loc='upper right'); ax.grid(False)

for ax in axs:
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axs[-1].set_xlabel('Time (HH:MM)')

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/multi_horizon_with_spikes.png')
plt.close()
print("✅ Saved results/multi_horizon_with_spikes.png")

# ===== ADDITIONAL ANALYSIS PLOTS =====

# 1) Pred vs Actual scatter
plt.figure(figsize=(6,6))
for i,h in enumerate(horizons):
    plt.scatter(actuals[:,i], preds[:,i], alpha=0.3, s=10, label=f't+{h}s')
maxv = max(actuals.max(), preds.max())
plt.plot([0, maxv], [0, maxv], 'k--', label='Ideal')
plt.xlabel('Actual packet_count'); plt.ylabel('Predicted packet_count')
plt.title('Predicted vs Actual'); plt.legend(); plt.grid(False)
plt.tight_layout(); plt.savefig('results/scatter_pred_vs_act.png'); plt.close()

# 2) Error distribution (KDE)
errors = preds - actuals
plt.figure(figsize=(8,4))
for i,h in enumerate(horizons):
    sns.kdeplot(errors[:,i], label=f't+{h}s', bw_adjust=0.5)
plt.axvline(0, color='k', linestyle='--')
plt.title('Error Distribution by Horizon'); plt.xlabel('Prediction Error (pkts)')
plt.legend(); plt.grid(False); plt.tight_layout()
plt.savefig('results/error_kde.png'); plt.close()

# 3) Boxplot of absolute % error
ape = np.abs((preds - actuals) / np.where(actuals == 0,1,actuals)) * 100
plt.figure(figsize=(6,4))
plt.boxplot([ape[:,i] for i in range(len(horizons))],
            labels=[f't+{h}s' for h in horizons])
plt.ylabel('Absolute % Error'); plt.title('Boxplot of % Error by Horizon')
plt.grid(False); plt.tight_layout()
plt.savefig('results/percent_error_box.png'); plt.close()

# 4) Mean Absolute Error by hour of day (t+60s)
df_err = pd.DataFrame({'time': times_act, 'err': np.abs(preds[:,0] - actuals[:,0])})
df_err['hour'] = df_err['time'].dt.hour
hour_mae = df_err.groupby('hour')['err'].mean()
plt.figure(figsize=(8,2))
plt.bar(hour_mae.index, hour_mae.values, color='tab:blue')
plt.xlabel('Hour of Day'); plt.ylabel('MAE (t+60s)')
plt.title('Mean Absolute Error by Hour'); plt.grid(False)
plt.tight_layout(); plt.savefig('results/mae_by_hour.png'); plt.close()

# 5) RF feature importances
importances = rf.feature_importances_
lags = [f'lag_{i}' for i in range(1, n_lag+1)]
plt.figure(figsize=(5,3))
plt.barh(lags, importances, color='tab:green')
plt.xlabel('Importance'); plt.title('RF Hourly‐Spike Feature Importances')
plt.grid(False); plt.tight_layout()
plt.savefig('results/rf_importance.png'); plt.close()

# 6) Attention weights
xb_sample = X[:1]
with torch.no_grad():
    g, _ = model.gru(xb_sample)
    attn_scores  = model.attn(g).squeeze(-1)
    attn_weights = torch.softmax(attn_scores, dim=1).cpu().numpy()[0]
plt.figure(figsize=(8,2))
plt.bar(np.arange(window_size), attn_weights, width=1.0)
plt.xlabel('Second in window'); plt.ylabel('Attention weight')
plt.title('Attention Weights (sample #1)'); plt.grid(False)
plt.tight_layout(); plt.savefig('results/attention_weights.png'); plt.close()

# 7) ROC Curve
thr = y_te.mean()
y_true  = (y_te > thr).astype(int)
y_score = y_te_pred
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc     = auc(fpr, tpr)
plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve (Hourly Spike)')
plt.legend(); plt.grid(False)
plt.tight_layout(); plt.savefig('results/roc_curve.png'); plt.close()

# 8) Precision‐Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_score)
plt.figure(figsize=(4,4))
plt.plot(recall, precision)
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision‐Recall (Hourly Spike)')
plt.grid(False); plt.tight_layout()
plt.savefig('results/pr_curve.png'); plt.close()

print("✅ Saved all plots under results/")  

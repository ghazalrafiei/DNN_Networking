#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import argparse
import random
import joblib
from sklearn.preprocessing import StandardScaler

def extract_features(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, parse_dates=['timestamp'])

    # Normalize domain names: lowercase and strip trailing dot
    df['domain'] = df['domain'].str.lower().str.rstrip('.')

    grouped = df.groupby('domain')
    rows = []

    for domain, group in grouped:
        total_queries = len(group)
        a_queries = (group['qtype'] == 1).sum()
        aaaa_queries = (group['qtype'] == 28).sum()
        unique_ips = group['src_ip'].nunique()

        # Compute burstiness: standard deviation of inter-arrival times (in seconds)
        timestamps = group['timestamp'].astype(np.int64) // 1_000_000_000
        intervals = np.diff(timestamps) if len(timestamps) > 1 else np.array([0])
        burstiness = np.std(intervals)

        # NXDOMAIN ratio: proportion of queries that returned NXDOMAIN (rcode == 3)
        nxdomain_rate = (group['rcode'] == 3).mean()

        rows.append({
            'domain': domain,
            'total_queries': total_queries,
            'A_queries': a_queries,
            'AAAA_queries': aaaa_queries,
            'unique_ips': unique_ips,
            'burstiness': burstiness,
            'nxdomain_rate': nxdomain_rate
        })

    return pd.DataFrame(rows)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_model(model, train_loader, val_loader, device):
    train_loss, val_loss = [], []
    best_val_loss = float('inf')
    patience, no_improve = 50, 0
    num_epochs = 300
    best_model_state = None
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_loss.append(epoch_train_loss / len(train_loader))

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                epoch_val_loss += loss.item()
        val_loss.append(epoch_val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss[-1]:.6f} | Val Loss: {val_loss[-1]:.6f}")

        if val_loss[-1] < best_val_loss - 1e-4:
            best_val_loss = val_loss[-1]
            best_model_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), "ae_model.pth")
        print("Best model saved to ae_model.pth")
        
    plot_loss(train_loss, val_loss)
    return model

def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = []

    with torch.no_grad():
        for (batch,) in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = torch.mean((batch - output) ** 2, dim=1)
            total_loss.extend(loss.cpu().numpy())

    plt.figure(figsize=(8, 5))
    sns.histplot(total_loss, bins=50, kde=True)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Reconstruction Error Distribution")
    plt.savefig("reconstruction_error.png")
    print("Saved reconstruction error histogram to reconstruction_error.png")


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.show()

def analyze_reconstruction_errors(model, X, features_df, device):
    # X = features_df.drop(columns=["domain"]).to_numpy(dtype=np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon = model(X_tensor).cpu().numpy()
        latent_vectors = model.encoder(X_tensor).cpu().numpy()

    reconstruction_errors = np.mean((X - recon) ** 2, axis=1)

    features_df = features_df.reset_index(drop=True).copy()
    features_df['recon_error'] = reconstruction_errors

    top_n = 20
    high_error_domains = features_df.sort_values(by='recon_error', ascending=False).head(top_n)
    low_error_domains = features_df.sort_values(by='recon_error', ascending=True).head(top_n)

    print("\nDomains with High Reconstruction Error:")
    print(high_error_domains[['domain', 'recon_error']])

    print("\nDomains with Low Reconstruction Error:")
    print(low_error_domains[['domain', 'recon_error']])

    # PCA visualization

    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        latent_2d[:, 0], latent_2d[:, 1],
        c=reconstruction_errors,
        cmap='coolwarm',
        s=20
    )
    plt.colorbar(scatter, label="Reconstruction Error")
    plt.title("Latent Space (PCA projection, colored by reconstruction error)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file with DNS queries")
    args = parser.parse_args()
    # "dns_queries_20210118_mia.csv"
    csv_path = args.csv
    features_df = extract_features(csv_path)
    filtered_df = features_df.copy()
    filtered_df = filtered_df[filtered_df['total_queries'] > 1]
    filtered_df = filtered_df[filtered_df['unique_ips'] > 1]
    filtered_df = filtered_df[filtered_df['burstiness'] > 0]
    print(f"Filtered dataset shape: {filtered_df.shape}")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Normalize
    feature_cols = [
        'total_queries', 'A_queries', 'AAAA_queries',
        'unique_ips', 'burstiness', 'nxdomain_rate'
    ]

    scaler = StandardScaler()
    X = scaler.fit_transform(filtered_df[feature_cols])

    joblib.dump(scaler, 'scaler.pkl')
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n = len(X)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    
    train_tensor = torch.tensor(X[:train_end], dtype=torch.float32)
    val_tensor = torch.tensor(X[train_end:val_end], dtype=torch.float32)
    test_tensor = torch.tensor(X[val_end:], dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=64)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=64)
    # Define model and optimizer
    model = Autoencoder(input_dim=X.shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, device)
    
    evaluate(model, test_loader, device)
    analyze_reconstruction_errors(model, X,filtered_df, device)
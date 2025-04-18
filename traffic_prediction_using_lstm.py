import os
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import logging
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


def extract_traffic(file_path):
    """Extract DNS traffic from .lax.pcap.xz file using xzcat and tcpdump."""
    logging.info(f"Extracting traffic from {file_path}")
    command = f"xzcat {file_path} | tcpdump -r - -nn -v -q"
    try:
        result = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT
        )
        output = result.decode('utf-8').splitlines()
        logging.info(f"Extracted {len(output)} lines from {file_path}")
        logging.debug(f"Sample output: {output[:3]}")
        return output
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Error extracting traffic from {file_path}: {e.output.decode()}"
        )
        return []


def preprocess_traffic_data(traffic_data):
    """
    Convert traffic data to time-based features and packet lengths.
    Returns timestamp information and packet lengths.
    Modified to use 1-hour intervals instead of 15-minute intervals.
    """
    logging.info("Preprocessing traffic data...")

    # Store packet data by time intervals (every hour)
    interval_data = defaultdict(list)
    packet_count = defaultdict(int)
    timestamps = []
    packet_lengths = []

    time_pattern = re.compile(r'(\d{2}):(\d{2}):(\d{2})\.(\d+)')
    length_pattern = re.compile(r'length (\d+)')

    for line in traffic_data:
        if 'IP' in line or 'IP6' in line:
            try:
                # Extract timestamp
                time_match = time_pattern.search(line)
                if time_match:
                    hour, minute, second, msec = time_match.groups()
                    # Group by hour intervals (instead of 15-minute intervals)
                    interval_key = f"{hour}:00"

                    # Extract packet length
                    length_match = length_pattern.search(line)
                    if length_match:
                        packet_length = int(length_match.group(1))

                        # Store data
                        interval_data[interval_key].append(packet_length)
                        packet_count[interval_key] += 1

                        # Also keep all data points for detailed analysis
                        timestamps.append(f"{hour}:{minute}:{second}")
                        packet_lengths.append(packet_length)
            except Exception as e:
                logging.warning(f"Error parsing line: {line}. {e}")

    logging.info(f"Processed {len(packet_lengths)} packets")
    logging.info(f"Found {len(interval_data)} unique 1-hour intervals")

    # Calculate features per interval
    interval_summary = []
    for interval_key in sorted(interval_data.keys()):
        avg_length = sum(interval_data[interval_key]) / len(interval_data[interval_key])
        count = packet_count[interval_key]
        # Add standard deviation as a feature
        std_dev = np.std(interval_data[interval_key])
        # Calculate packet rate (packets per minute)
        packet_rate = count / 60.0  # Changed from 15.0 to 60.0 for hourly rate
        
        interval_summary.append((interval_key, avg_length, count, std_dev, packet_rate))
    
    # Log the intervals found for debugging
    logging.info(f"Intervals found in data: {[item[0] for item in interval_summary]}")

    return timestamps, packet_lengths, interval_summary


def create_interval_time_series(interval_summary):
    """Convert interval summary data into a time series for LSTM model."""
    logging.info("Creating time series data from intervals...")

    intervals = [item[0] for item in interval_summary]
    avg_lengths = [item[1] for item in interval_summary]
    counts = [item[2] for item in interval_summary]
    std_devs = [item[3] for item in interval_summary]
    packet_rates = [item[4] for item in interval_summary]

    # Normalize features to handle different magnitudes
    count_scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_counts = count_scaler.fit_transform(
        np.array(counts).reshape(-1, 1)
    )

    length_scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_lengths = length_scaler.fit_transform(
        np.array(avg_lengths).reshape(-1, 1)
    )
    
    std_scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_stds = std_scaler.fit_transform(
        np.array(std_devs).reshape(-1, 1)
    )
    
    rate_scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_rates = rate_scaler.fit_transform(
        np.array(packet_rates).reshape(-1, 1)
    )

    # Combine features
    X = np.hstack((normalized_counts, normalized_lengths, normalized_stds, normalized_rates))

    logging.info(
        f"Created time series with {len(X)} data points and {X.shape[1]} features"
    )

    return intervals, X, (count_scaler, length_scaler, std_scaler, rate_scaler)


class TrafficLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(TrafficLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(
            x.device
        )
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(
            x.device
        )

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Get the output from the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out


def sequential_prediction_training(X, intervals, val_X=None, val_intervals=None, num_epochs=50, batch_size=1, hidden_dim=64):
    """
    Train model sequentially, predicting each interval using previous interval data,
    updating model after each prediction. Includes validation if validation data is provided.
    """
    logging.info("Starting sequential interval-by-interval prediction and learning...")
    
    # Need at least 3 intervals for meaningful prediction
    if len(X) < 3:
        logging.error(f"Not enough interval data points for sequential training. Found {len(X)} intervals.")
        return None, None, []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Initialize model
    input_dim = X.shape[1]
    model = TrafficLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2).to(device)
    criterion = nn.MSELoss()
    # Using AdamW instead of Adam as recommended
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # Lists to store predictions and metrics
    predictions = []
    actuals = []
    mses = []
    maes = []
    
    # Lists to store training and validation losses per interval
    train_losses_per_interval = []
    val_losses_per_interval = []
    
    # Track epoch-by-epoch losses for the last interval (for plotting epoch vs loss)
    last_interval_epoch_train_losses = []
    last_interval_epoch_val_losses = []
    
    # Initialize the first window for training (minimum 1 interval)
    min_train_size = 1
    
    for interval_idx in range(min_train_size, len(X) - 1):
        # Current training data up to interval_idx
        X_train = X[:interval_idx]
        y_train = X[1:interval_idx+1, 0]  # Predict the packet count (first feature)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        
        # Create DataLoader for efficient training
        train_dataset = TensorDataset(X_train_tensor.unsqueeze(1), y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Train on current data
        model.train()
        epoch_train_losses = []
        epoch_val_losses = []
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss/len(train_loader)
            epoch_train_losses.append(avg_epoch_loss)
            
            # Validation phase
            if val_X is not None and len(val_X) > 1:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_idx in range(len(val_X) - 1):
                        val_input = torch.tensor(val_X[val_idx].reshape(1, 1, -1), dtype=torch.float32).to(device)
                        val_target = torch.tensor(val_X[val_idx + 1, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                        val_pred = model(val_input)
                        val_loss += criterion(val_pred, val_target).item()
                    
                    avg_val_loss = val_loss / (len(val_X) - 1)
                    epoch_val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                val_info = f", Val Loss: {epoch_val_losses[-1]:.4f}" if epoch_val_losses else ""
                logging.info(f'Interval {interval_idx}, Epoch [{epoch+1}/{num_epochs}], '
                           f'Train Loss: {avg_epoch_loss:.4f}{val_info}')
        
        # Store the average training and validation loss for this interval
        train_losses_per_interval.append(np.mean(epoch_train_losses))
        if epoch_val_losses:
            val_losses_per_interval.append(np.mean(epoch_val_losses))
        
        # Store the full epoch history for the last interval (for epoch vs loss plot)
        if interval_idx == len(X) - 2:  # Last interval
            last_interval_epoch_train_losses = epoch_train_losses
            last_interval_epoch_val_losses = epoch_val_losses
        
        # Make prediction for next interval
        model.eval()
        with torch.no_grad():
            # Use the current interval to predict the next interval
            current_interval_data = torch.tensor(X[interval_idx].reshape(1, 1, -1), dtype=torch.float32).to(device)
            next_interval_pred = model(current_interval_data).item()
            next_interval_actual = X[interval_idx + 1, 0]
            
            # Store predictions
            predictions.append(next_interval_pred)
            actuals.append(next_interval_actual)
            
            # Compute metrics
            interval_mse = mean_squared_error([next_interval_actual], [next_interval_pred])
            interval_mae = mean_absolute_error([next_interval_actual], [next_interval_pred])
            
            mses.append(interval_mse)
            maes.append(interval_mae)
            
            logging.info(f"Interval {intervals[interval_idx]} → {intervals[interval_idx+1]}: "
                        f"Predicted={next_interval_pred:.4f}, Actual={next_interval_actual:.4f}, "
                        f"MSE={interval_mse:.4f}")
    
    # Calculate overall metrics
    overall_mse = mean_squared_error(actuals, predictions)
    overall_mae = mean_absolute_error(actuals, predictions)
    
    logging.info(f"Overall MSE: {overall_mse:.4f}, MAE: {overall_mae:.4f}")
    
    # Plot results
    plt.figure(figsize=(18, 14))
    
    # Plot individual predictions vs actuals
    interval_labels = intervals[min_train_size+1:]
    plt.subplot(3, 1, 1)
    plt.plot(interval_labels, actuals, 'b-', label='Actual')
    plt.plot(interval_labels, predictions, 'r--', label='Predicted')
    plt.title('Interval-by-Interval Sequential Prediction')
    plt.xlabel('Time Interval')
    plt.ylabel('Normalized Traffic')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Plot training and validation loss per interval
    plt.subplot(3, 1, 2)
    plt.plot(interval_labels, train_losses_per_interval, 'g-', label='Training Loss')
    if val_X is not None and len(val_losses_per_interval) > 0:
        # Ensure we plot only as many validation points as we have intervals
        valid_intervals = interval_labels[:len(val_losses_per_interval)]
        plt.plot(valid_intervals, val_losses_per_interval, 'c--', label='Validation Loss')
    plt.title('Training and Validation Loss per Interval')
    plt.xlabel('Time Interval')
    plt.ylabel('Loss')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Plot MSE and MAE over time
    plt.subplot(3, 1, 3)
    plt.plot(interval_labels, mses, 'g-', label='MSE')
    plt.plot(interval_labels, maes, 'm--', label='MAE')
    plt.title('Error Metrics Over Time')
    plt.xlabel('Time Interval')
    plt.ylabel('Error')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sequential_prediction_results.png')
    
    # Create a separate figure for epoch vs loss for the last training interval
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, num_epochs + 1))
    plt.plot(epochs, last_interval_epoch_train_losses, 'b-', label='Training Loss')
    if last_interval_epoch_val_losses:
        plt.plot(epochs, last_interval_epoch_val_losses, 'r--', label='Validation Loss')
    plt.title(f'Training and Validation Loss vs. Epochs (Last Interval: {intervals[-2]} → {intervals[-1]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('epoch_vs_loss.png')
    
    return (overall_mse, overall_mae), model, (predictions, actuals, train_losses_per_interval, val_losses_per_interval)


def verify_sequential_model(model, verify_X, verify_intervals, device=None):
    """
    Verify the model on new data using the same sequential prediction approach.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"Verifying model on {len(verify_X)} intervals of new data...")
    
    model.eval()
    predictions = []
    actuals = []
    
    # Need at least 2 intervals of data (1 for context, 1 for validation)
    if len(verify_X) < 2:
        logging.error("Not enough verification data")
        return None
    
    for interval_idx in range(len(verify_X) - 1):
        with torch.no_grad():
            # Use current interval to predict next interval
            current_interval_data = torch.tensor(verify_X[interval_idx].reshape(1, 1, -1), dtype=torch.float32).to(device)
            next_interval_pred = model(current_interval_data).item()
            next_interval_actual = verify_X[interval_idx + 1, 0]
            
            predictions.append(next_interval_pred)
            actuals.append(next_interval_actual)
    
    # Calculate metrics
    verify_mse = mean_squared_error(actuals, predictions)
    verify_mae = mean_absolute_error(actuals, predictions)
    
    logging.info(f"Verification MSE: {verify_mse:.4f}, MAE: {verify_mae:.4f}")
    
    # Plot verification results
    plt.figure(figsize=(12, 6))
    interval_labels = verify_intervals[1:]
    plt.plot(interval_labels, actuals, 'b-', label='Actual Traffic')
    plt.plot(interval_labels, predictions, 'r--', label='Predicted Traffic')
    plt.title('DNS Traffic Verification (Sequential Prediction)')
    plt.xlabel('Time Interval')
    plt.ylabel('Normalized Traffic Volume')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig('sequential_verification.png')
    
    return verify_mse, verify_mae


def process_multiple_days(base_dir, days, files_per_day=25):
    """
    Process traffic data from multiple days.
    
    Args:
        base_dir (str): Base directory path for traces
        days (list): List of days to process (e.g., ["08", "09"])
        files_per_day (int): Number of files to process per day
        
    Returns:
        dict: Dictionary containing processed data for each day
    """
    all_data = {}
    
    for day in days:
        day_dir = os.path.join(base_dir, day)
        logging.info(f"Processing files from day {day} in directory: {day_dir}")
        
        try:
            # Get all .lax.pcap.xz files for this day with "lax" in the filename
            all_files = sorted([f for f in os.listdir(day_dir) if f.endswith('.lax.pcap.xz') and 'lax' in f])
            if not all_files:
                logging.error(f"No .lax.pcap.xz files found in {day_dir}")
                continue
                
            logging.info(f"Found {len(all_files)} LAX files in directory for day {day}")
            
            # Select files for this day
            if len(all_files) <= files_per_day:
                selected_files = all_files
            else:
                indices = np.linspace(0, len(all_files)-1, files_per_day, dtype=int)
                selected_files = [all_files[i] for i in indices]
            
            logging.info(f"Selected {len(selected_files)} files for day {day}: {selected_files}")
            
            # Process the selected files
            traffic_data = []
            for file in selected_files:
                file_path = os.path.join(day_dir, file)
                file_data = extract_traffic(file_path)
                traffic_data.extend(file_data)
                logging.info(f"Processed {file}: {len(file_data)} lines")
            
            # Check if data extraction was successful
            if not traffic_data:
                logging.error(f"Failed to extract traffic data from the selected files for day {day}")
                continue
            
            # Preprocess the traffic data
            timestamps, packet_lengths, interval_summary = preprocess_traffic_data(traffic_data)
            
            # Store the processed data
            all_data[day] = {
                "timestamps": timestamps,
                "packet_lengths": packet_lengths,
                "interval_summary": interval_summary
            }
            
            logging.info(f"Successfully processed data for day {day}")
            
        except Exception as e:
            logging.error(f"Error processing day {day}: {e}")
    
    return all_data


def main():
    logging.info("Starting DNS traffic analysis with sequential prediction...")
    
    # Base directory for data
    base_dir = "/nfs/lander/traces/dns/B_Root_week-20210108/lander_br"
    
    # Days to process (can be modified to include more days)
    days_to_process = ["08", "09"]
    
    # Number of files to process per day - increased to 25
    files_per_day = 25
    
    # Process data from multiple days
    all_data = process_multiple_days(base_dir, days_to_process, files_per_day)
    
    if len(all_data) < 2:
        logging.error("Not enough days processed for training and validation.")
        return
    
    # Create time series data for training (08th day)
    if "08" in all_data:
        intervals_08, X_08, scalers_08 = create_interval_time_series(all_data["08"]["interval_summary"])
    else:
        logging.error("No data found for day 08 (training data)")
        return
    
    # Create time series data for validation using the same scalers (09th day)
    if "09" in all_data:
        # Extract the raw data from the validation intervals
        interval_summary_09 = all_data["09"]["interval_summary"]
        intervals_09 = [item[0] for item in interval_summary_09]
        avg_lengths_09 = [item[1] for item in interval_summary_09]
        counts_09 = [item[2] for item in interval_summary_09]
        std_devs_09 = [item[3] for item in interval_summary_09]
        packet_rates_09 = [item[4] for item in interval_summary_09]
        
        # Use the same scalers from training to normalize validation data
        count_scaler, length_scaler, std_scaler, rate_scaler = scalers_08
        normalized_counts_09 = count_scaler.transform(np.array(counts_09).reshape(-1, 1))
        normalized_lengths_09 = length_scaler.transform(np.array(avg_lengths_09).reshape(-1, 1))
        normalized_stds_09 = std_scaler.transform(np.array(std_devs_09).reshape(-1, 1))
        normalized_rates_09 = rate_scaler.transform(np.array(packet_rates_09).reshape(-1, 1))
        
        # Combine validation features
        X_09 = np.hstack((normalized_counts_09, normalized_lengths_09, normalized_stds_09, normalized_rates_09))
        
        logging.info(f"Created validation time series with {len(X_09)} data points and {X_09.shape[1]} features")
    else:
        logging.warning("No data found for day 09 (validation data), proceeding without validation")
        X_09 = None
        intervals_09 = None
    
    # Check if we have enough intervals for training
    if len(intervals_08) < 3:
        logging.error(f"Not enough interval data points for training. Found {len(intervals_08)} intervals.")
        return
    
    # Perform sequential prediction training with validation
    metrics, trained_model, results = sequential_prediction_training(
        X_08, intervals_08, val_X=X_09, val_intervals=intervals_09, 
        num_epochs=100,  # Increased from 50 to 100 for better convergence
        batch_size=1, 
        hidden_dim=64
    )
    
    if metrics:
        overall_mse, overall_mae = metrics
        logging.info(f"Sequential Prediction - MSE: {overall_mse:.4f}, MAE: {overall_mae:.4f}")
        
        # Save the trained model
        if trained_model:
            torch.save(trained_model.state_dict(), "dns_traffic_model.pth")
            logging.info("Model saved successfully")
    
    # Verify the model on the validation data
    if trained_model and X_09 is not None:
        verify_mse, verify_mae = verify_sequential_model(trained_model, X_09, intervals_09)
        logging.info(f"Verification on 09th data - MSE: {verify_mse:.4f}, MAE: {verify_mae:.4f}")
    
    logging.info("DNS traffic analysis with sequential prediction completed")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import config

def series_to_supervised(data, window=1, lag=1, dropnan=True):
    """
    Convert a time series to a supervised learning dataset
    
    Args:
        data: DataFrame of time series
        window: Number of lag observations as input (X)
        lag: Number of observations as output (y)
        dropnan: Boolean whether or not to drop rows with NaN values
    
    Returns:
        Pandas DataFrame of series framed for supervised learning
    """
    cols, names = list(), list()
    
    # Input sequence (t-n, ... t-1)
    for i in range(window-1, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    
    # Current timestep (t=0)
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    
    # Target timestep (t+1, t+2, ... t+lag)
    for i in range(1, lag + 1):
        cols.append(data.shift(-i))
        names += [('%s(t+%d)' % (col, i)) for col in data.columns]
    
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg


def get_device():
    """
    Get the appropriate device for PyTorch operations.
    
    Returns:
        torch.device: MPS if available, otherwise CPU
    """
    if config.USE_MPS and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for GPU acceleration.")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU.")
    
    return device

def prepare_data_for_pytorch(df, window_size, segment, is_training=True):
    """
    Prepare data for PyTorch models.
    
    Args:
        df: DataFrame containing the time series data
        window_size: Size of the sliding window for input features
        segment: Which segment to use (1 or 2)
        is_training: Whether this is training data (to fit the scaler)
        
    Returns:
        X: Input features (PyTorch tensor)
        y: Target values (PyTorch tensor)
        scaler: Fitted scaler (for inverse transform)
    """
    # Filter by segment
    df = df[df['segment'] == segment].copy()
    
    # Drop unnecessary columns
    data = df.drop(['segment', 'application_date'], axis=1).values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data) if is_training else scaler.transform(data)
    
    # Prepare sliding window data
    X, y = [], []
    for i in range(len(data_scaled) - window_size):
        X.append(data_scaled[i:i+window_size])
        y.append(data_scaled[i+window_size])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM/GRU [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Convert to PyTorch tensors
    device = get_device()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    return X_tensor, y_tensor, scaler

def train_test_val_split(X, y, test_size=0.2, val_size=0.1):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X: Input features
        y: Target values
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        
    Returns:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
    """
    # Calculate split indices
    n_samples = len(X)
    test_idx = int(n_samples * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    # Split the data
    X_train, y_train = X[:val_idx], y[:val_idx]
    X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
    X_test, y_test = X[test_idx:], y[test_idx:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

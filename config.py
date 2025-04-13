# Configuration parameters

# Data paths
TRAIN_PATH = "Dataset/train_fwYjLYX.csv"
TEST_PATH = "Dataset/test_1eLl9Yf.csv"
SAMPLE_SUBMISSION_PATH = "Dataset/sample_submission_IIzFVsf.csv"
MODEL_SAVE_PATH = "./models/saved"

# Model parameters
WINDOW_SIZE = 30  # Number of time steps to consider
FORECAST_HORIZON = 1  # Number of steps to predict ahead

# LSTM parameters
LSTM_HIDDEN_DIM = 50
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT = 0.2
LSTM_LEARNING_RATE = 0.001
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 100

# GRU parameters
GRU_HIDDEN_DIM = 50
GRU_NUM_LAYERS = 1
GRU_DROPOUT = 0.2
GRU_LEARNING_RATE = 0.001
GRU_EPOCHS = 100
GRU_BATCH_SIZE = 100

# Random Forest parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42

# SVR parameters
SVR_KERNEL = 'rbf'
SVR_C = 1.0
SVR_EPSILON = 0.1
SVR_GAMMA = 'scale'

# ARIMA parameters
ARIMA_P = 1  # Autoregressive order
ARIMA_D = 1  # Differencing order
ARIMA_Q = 1  # Moving average order

# Device configuration
USE_MPS = True  # Use MPS GPU acceleration

# Evaluation
TEST_SIZE = 0.2  # Percentage of data for testing
VAL_SIZE = 0.1   # Percentage of training data for validation
METRICS = ['mape', 'rmse', 'mae']  # Metrics to evaluate

# Random seed for reproducibility
RANDOM_SEED = 42
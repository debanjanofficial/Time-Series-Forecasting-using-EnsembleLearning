# Configuration parameters

# Data paths
TRAIN_PATH = "Dataset/train_fwYjLYX.csv"
TEST_PATH = "Dataset/test_1eLl9Yf.csv"
SAMPLE_SUBMISSION_PATH = "Dataset/sample_submission_IIzFVsf.csv"

# Model parameters
WINDOW_SIZE = 30  # Number of time steps to consider
LAG_SIZE = 1  # Number of steps to predict ahead

# Neural Network parameters (for LSTM and GRU)
NN_NODES = 50
EPOCHS = 100
BATCH_SIZE = 100

# Random Forest parameters
N_ESTIMATORS = 100
MAX_DEPTH = 10

# SVR parameters
KERNEL = 'rbf'
C = 1.0
EPSILON = 0.1

# ARIMA parameters
ARIMA_ORDER = (5, 1, 0)  # (p, d, q) order

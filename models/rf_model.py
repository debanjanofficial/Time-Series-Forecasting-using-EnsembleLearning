import numpy as np
from sklearn.ensemble import RandomForestRegressor
import config

def prepare_data_for_rf(X, y):
    """
    Prepare data for Random Forest model.
    """
    # Convert tensors to numpy arrays if they are torch tensors
    if hasattr(X, 'cpu'):
        X = X.cpu().numpy()
    if hasattr(y, 'cpu'):
        y = y.cpu().numpy()
    
    # Reshape X from 3D to 2D
    n_samples, n_steps, n_features = X.shape
    X_2d = X.reshape(n_samples, n_steps * n_features)
    
    # Reshape y from 2D to 1D if needed
    if len(y.shape) > 1 and y.shape[1] == 1:
        y_1d = y.ravel()
    else:
        y_1d = y
    
    return X_2d, y_1d

def train_rf_model(X_train, y_train, params=None):
    """
    Train Random Forest model.
    """
    # Prepare data
    X_train_2d, y_train_1d = prepare_data_for_rf(X_train, y_train)
    
    # Set model parameters
    if params is None:
        params = {
            'n_estimators': config.RF_N_ESTIMATORS,
            'max_depth': config.RF_MAX_DEPTH,
            'random_state': config.RANDOM_SEED
        }
    
    # Create and train the model
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=params['random_state']
    )
    
    model.fit(X_train_2d, y_train_1d)
    
    return model

def predict_rf(model, X_test):
    """
    Generate predictions using Random Forest model.
    """
    # Prepare test data
    X_test_2d, _ = prepare_data_for_rf(X_test, np.zeros((X_test.shape[0], 1)))
    
    # Generate predictions
    predictions = model.predict(X_test_2d)
    
    return predictions.reshape(-1, 1)

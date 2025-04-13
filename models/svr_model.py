import numpy as np
from sklearn.svm import SVR
import config
from models.rf_model import prepare_data_for_rf

def train_svr_model(X_train, y_train, params=None):
    """
    Train Support Vector Regression model.
    """
    # Prepare data
    X_train_2d, y_train_1d = prepare_data_for_rf(X_train, y_train)
    
    # Set model parameters
    if params is None:
        params = {
            'kernel': config.SVR_KERNEL,
            'C': config.SVR_C,
            'epsilon': config.SVR_EPSILON,
            'gamma': config.SVR_GAMMA
        }
    
    # Create and train the model
    model = SVR(
        kernel=params['kernel'],
        C=params['C'],
        epsilon=params['epsilon'],
        gamma=params['gamma']
    )
    
    model.fit(X_train_2d, y_train_1d)
    
    return model

def predict_svr(model, X_test):
    """
    Generate predictions using SVR model.
    """
    # Prepare test data
    X_test_2d, _ = prepare_data_for_rf(X_test, np.zeros((X_test.shape[0], 1)))
    
    # Generate predictions
    predictions = model.predict(X_test_2d)
    
    return predictions.reshape(-1, 1)

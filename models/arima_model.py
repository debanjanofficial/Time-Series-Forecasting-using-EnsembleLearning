import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import config

def train_arima_model(series, params=None):
    """
    Train ARIMA model.
    """
    # Set model parameters
    if params is None:
        params = {
            'p': config.ARIMA_P,
            'd': config.ARIMA_D,
            'q': config.ARIMA_Q
        }
    
    # Create and fit the model
    model = ARIMA(series, order=(params['p'], params['d'], params['q']))
    model_fit = model.fit()
    
    print(model_fit.summary())
    
    return model_fit

def auto_arima_model(series):
    """
    Automatically select best ARIMA parameters and train model.
    """
    # Find the best parameters
    model = auto_arima(
        series,
        start_p=1, start_q=1,
        max_p=3, max_q=3,
        m=1,  # No seasonality
        d=None,  # Let auto_arima determine 'd'
        seasonal=False,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    print(model.summary())
    
    return model

def predict_arima(model, n_periods):
    """
    Generate predictions using ARIMA model.
    """
    # Generate forecasts
    forecast = model.forecast(steps=n_periods)
    
    return forecast

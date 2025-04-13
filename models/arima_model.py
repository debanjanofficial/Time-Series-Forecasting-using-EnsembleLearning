import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import config

def check_stationarity(series):
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.
    """
    result = adfuller(series)
    p_value = result[1]
    is_stationary = p_value <= 0.05
    
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Is stationary: {is_stationary}")
    
    return is_stationary, p_value

def determine_arima_order(series):
    """
    Determine the best order for ARIMA model using a simple grid search.
    """
    # Check stationarity to determine d
    is_stationary, _ = check_stationarity(series)
    d = 0 if is_stationary else 1
    
    # Grid search for p and q
    best_aic = float('inf')
    best_order = None
    
    for p in range(0, 3):
        for q in range(0, 3):
            try:
                model = ARIMA(series, order=(p, d, q))
                model_fit = model.fit()
                aic = model_fit.aic
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    
                print(f"ARIMA({p},{d},{q}) - AIC: {aic}")
            except:
                continue
    
    print(f"Best order: ARIMA{best_order} with AIC: {best_aic}")
    return best_order

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
    best_order = determine_arima_order(series)
    
    # Train the model with the best parameters
    model = ARIMA(series, order=best_order)
    model_fit = model.fit()
    
    print(model_fit.summary())
    
    return model_fit

def predict_arima(model, n_periods):
    """
    Generate predictions using ARIMA model.
    """
    # Generate forecasts
    forecast = model.forecast(steps=n_periods)
    
    return forecast

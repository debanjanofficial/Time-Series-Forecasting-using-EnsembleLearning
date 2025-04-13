import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_metrics(y_true, y_pred):
    """
    Calculate various evaluation metrics.
    """
    metrics = {}
    
    # Mean Absolute Error
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Absolute Percentage Error
    try:
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    except:
        metrics['mape'] = np.nan
    
    return metrics

def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models on test data.
    """
    results = {}
    
    for name, model in models.items():
        # Generate predictions
        if name in ['lstm', 'gru']:
            y_pred = model(X_test).detach().cpu().numpy()
        elif name == 'arima':
            y_pred = model.forecast(len(y_test))
        else:  # Random Forest and SVR
            from models.rf_model import predict_rf
            from models.svr_model import predict_svr
            if name == 'rf':
                y_pred = predict_rf(model, X_test)
            else:
                y_pred = predict_svr(model, X_test)
        
        # Calculate metrics
        results[name] = calculate_metrics(y_test.cpu().numpy(), y_pred)
        
        print(f"{name.upper()} Model:")
        for metric, value in results[name].items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    return results

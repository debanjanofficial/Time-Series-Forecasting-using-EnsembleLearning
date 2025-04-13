import pandas as pd
import numpy as np
import torch
from models.lstm_model import predict_lstm
from models.gru_model import predict_gru
from models.arima_model import predict_arima
from models.rf_model import predict_rf
from models.svr_model import predict_svr
import config

def generate_predictions(models, X_test, scaler, segment):
    """
    Generate predictions for test data using trained models.
    """
    preds = {}
    
    # Generate predictions for each model
    if 'lstm' in models:
        lstm_preds = predict_lstm(models['lstm'], X_test).detach().cpu().numpy()
        preds['lstm'] = scaler.inverse_transform(lstm_preds)
    
    if 'gru' in models:
        gru_preds = predict_gru(models['gru'], X_test).detach().cpu().numpy()
        preds['gru'] = scaler.inverse_transform(gru_preds)
    
    if 'arima' in models:
        arima_preds = predict_arima(models['arima'], len(X_test))
        preds['arima'] = arima_preds.reshape(-1, 1)
    
    if 'rf' in models:
        rf_preds = predict_rf(models['rf'], X_test)
        preds['rf'] = scaler.inverse_transform(rf_preds)
    
    if 'svr' in models:
        svr_preds = predict_svr(models['svr'], X_test)
        preds['svr'] = scaler.inverse_transform(svr_preds)
    
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame()
    for model_name, pred in preds.items():
        predictions_df[f"{model_name}_pred"] = pred.flatten()
    
    # Add segment information
    predictions_df['segment'] = segment
    
    return predictions_df

def create_ensemble_prediction(predictions):
    """
    Create ensemble prediction by averaging model predictions.
    """
    # Get column names for model predictions
    pred_cols = [col for col in predictions.columns if '_pred' in col]
    
    if len(pred_cols) == 0:
        raise ValueError("No prediction columns found in DataFrame")
    
    # Create ensemble by averaging
    predictions['ensemble'] = predictions[pred_cols].mean(axis=1)
    
    return predictions

def generate_test_file_predictions(models, df_train, df_test, window_size=30, segment=1):
    """
    Generate predictions specifically for the test file.
    
    Args:
        models: Dictionary of trained models
        df_train: Training data DataFrame
        df_test: Test data DataFrame
        window_size: Size of sliding window
        segment: Segment to process
        
    Returns:
        DataFrame with predictions
    """
    print(f"Generating predictions for test file (segment {segment})...")
    
    # Filter data for the segment
    train_segment = df_train[df_train['segment'] == segment].sort_values('application_date')
    test_segment = df_test[df_test['segment'] == segment].sort_values('application_date')
    
    if len(test_segment) == 0:
        print(f"No test data found for segment {segment}")
        return pd.DataFrame()
    
    # Get the latest window_size values from training data
    latest_values = train_segment['case_count'].tail(window_size).values
    
    # Scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_segment['case_count'].values.reshape(-1, 1))
    
    # Scale the latest values
    latest_scaled = scaler.transform(latest_values.reshape(-1, 1)).flatten()
    
    # Create predictions for each day in the test set
    test_dates = test_segment['application_date'].values
    predictions = []
    model_predictions = {name: [] for name in models.keys()}
    
    # Get device
    if config.USE_MPS and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Current window starts with the latest training data
    current_window = latest_scaled.copy()
    
    for date in test_dates:
        # Prepare input for the models (reshape for LSTM/GRU input: [1, window_size, 1])
        X_input = current_window.reshape(1, len(current_window), 1)
        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
        
        # Get predictions from each model
        for name, model in models.items():
            if name == 'lstm':
                pred = model(X_tensor).detach().cpu().numpy()[0, 0]
            elif name == 'gru':
                pred = model(X_tensor).detach().cpu().numpy()[0, 0]
            elif name == 'arima':
                # For ARIMA, we work with the original scale
                original_window = scaler.inverse_transform(current_window.reshape(-1, 1)).flatten()
                pred_orig = model.forecast(1)[0]
                # Scale back to match other models
                pred = scaler.transform([[pred_orig]])[0, 0]
            elif name == 'rf' or name == 'svr':
                # Reshape for sklearn models: [1, window_size]
                X_flat = current_window.reshape(1, -1)
                pred = model.predict(X_flat)[0]
            
            model_predictions[name].append(pred)
        
        # Calculate ensemble prediction (average of all models)
        day_preds = [model_predictions[name][-1] for name in models.keys()]
        ensemble_pred = sum(day_preds) / len(day_preds)
        predictions.append(ensemble_pred)
        
        # Update window for next prediction
        current_window = np.append(current_window[1:], ensemble_pred)
    
    # Invert scaling to get original values
    predictions_original = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Create result DataFrame
    result = pd.DataFrame({
        'application_date': test_dates,
        'segment': segment,
        'ensemble': predictions_original.flatten()
    })
    
    # Add individual model predictions
    for name in models.keys():
        model_preds_original = scaler.inverse_transform(np.array(model_predictions[name]).reshape(-1, 1))
        result[f'{name}_pred'] = model_preds_original.flatten()
    
    return result

def prepare_submission(predictions, test_dates, df_sample_sub):
    """
    Prepare submission file with improved error handling.
    
    Args:
        predictions: Dictionary of predictions for each segment
        test_dates: Dictionary of test dates for each segment
        df_sample_sub: Sample submission DataFrame
        
    Returns:
        DataFrame ready for submission
    """
    # Create a copy of the sample submission
    submission = df_sample_sub.copy()
    
    # Print diagnostic information
    print(f"Sample submission shape: {submission.shape}")
    for segment in predictions.keys():
        segment_rows = (submission['segment'] == segment).sum()
        pred_rows = len(predictions[segment])
        print(f"Segment {segment}: {pred_rows} predictions vs {segment_rows} submission rows")
    
    # Add predictions for each segment
    for segment, preds in predictions.items():
        segment_idx = submission['segment'] == segment
        n_segment = segment_idx.sum()
        
        # Check if ensemble column exists
        if 'ensemble' not in preds.columns:
            print(f"Warning: 'ensemble' column not found in predictions for segment {segment}")
            print(f"Available columns: {preds.columns.tolist()}")
            # Try to use the first prediction column available
            pred_cols = [col for col in preds.columns if '_pred' in col or col == 'ensemble']
            if pred_cols:
                pred_col = pred_cols[0]
                print(f"Using '{pred_col}' column instead")
                ensemble_values = preds[pred_col].values
            else:
                print(f"No prediction columns found for segment {segment}, skipping")
                continue
        else:
            ensemble_values = preds['ensemble'].values
            
        n_pred = len(ensemble_values)
        
        # Handle length mismatch
        if n_pred != n_segment:
            print(f"Length mismatch for segment {segment}: {n_pred} predictions vs {n_segment} rows")
            
            if n_pred > n_segment:
                # If we have more predictions than needed, use only what we need
                submission.loc[segment_idx, 'case_count'] = ensemble_values[:n_segment]
                print(f"  Using first {n_segment} of {n_pred} predictions")
            else:
                # If we have fewer predictions than needed, pad with the mean/last value
                segment_indices = submission.index[segment_idx].tolist()
                
                # Use available predictions
                submission.loc[segment_indices[:n_pred], 'case_count'] = ensemble_values
                
                # For the rest, use the mean of available predictions
                mean_val = ensemble_values.mean()
                submission.loc[segment_indices[n_pred:], 'case_count'] = mean_val
                print(f"  Used {n_pred} predictions and filled remaining {n_segment-n_pred} with mean: {mean_val:.4f}")
        else:
            # Lengths match, proceed normally
            submission.loc[segment_idx, 'case_count'] = ensemble_values
            print(f"  Successfully assigned {n_pred} predictions for segment {segment}")
    
    return submission


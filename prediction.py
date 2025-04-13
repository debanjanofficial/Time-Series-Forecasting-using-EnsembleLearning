import pandas as pd
import numpy as np
import torch
from models.lstm_model import predict_lstm
from models.gru_model import predict_gru
from models.arima_model import predict_arima
from models.rf_model import predict_rf
from models.svr_model import predict_svr

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
    
    # Create ensemble by averaging
    predictions['ensemble'] = predictions[pred_cols].mean(axis=1)
    
    return predictions

def prepare_submission(predictions, test_dates, df_sample_sub):
    """
    Prepare submission file.
    """
    # Create a copy of the sample submission
    submission = df_sample_sub.copy()
    
    # Add predictions for each segment
    for segment, preds in predictions.items():
        segment_idx = submission['segment'] == segment
        submission.loc[segment_idx, 'case_count'] = preds['ensemble'].values
    
    return submission

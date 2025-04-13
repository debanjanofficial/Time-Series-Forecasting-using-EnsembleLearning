import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import config

# Import custom modules
from data_loader import load_data, preprocess_data
from data_preparation import (
    prepare_data_for_pytorch,
    train_test_val_split,
)
from visualization import (
    plot_time_series,
    plot_model_comparison,
    plot_training_history
)
from models.lstm_model import train_lstm_model
from models.gru_model import train_gru_model
from models.arima_model import train_arima_model, auto_arima_model, predict_arima
from models.rf_model import train_rf_model
from models.svr_model import train_svr_model
from evaluation import evaluate_models
from prediction import generate_predictions, create_ensemble_prediction, prepare_submission

def main():
    """
    Main function to run the time series forecasting pipeline.
    """
    print("Starting Time Series Forecasting Pipeline...")
    
    # Step 1: Load data
    print("\n--- Step 1: Loading Data ---")
    df_train, df_test, df_sample_sub = load_data()
    
    if df_train is None or df_test is None or df_sample_sub is None:
        print("Error loading data. Exiting...")
        return
    
    # Step 2: Preprocess data
    print("\n--- Step 2: Preprocessing Data ---")
    df_train, df_test, df_train_1, df_train_2 = preprocess_data(df_train, df_test)
    
    # Step 3: Exploratory Data Analysis
    print("\n--- Step 3: Exploratory Data Analysis ---")
    plot_time_series(df_train, 'case_count', segment=1, title="Case Count - Segment 1")
    plot_time_series(df_train, 'case_count', segment=2, title="Case Count - Segment 2")
    
    # Step 4: Time Series Data Preparation
    print("\n--- Step 4: Time Series Data Preparation ---")
    
    # Process each segment separately
    segments = [1, 2]
    all_models = {}
    all_predictions = {}
    
    for segment in segments:
        print(f"\nProcessing Segment {segment}...")
        
        # Prepare data for PyTorch models
        X, y, scaler = prepare_data_for_pytorch(df_train, window_size=config.WINDOW_SIZE, segment=segment)
        
        # Split into train, validation, and test sets
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_test_val_split(
            X, y, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Step 5: Train models
        print(f"\n--- Step 5: Training Models for Segment {segment} ---")
        models = {}
        
        # Train LSTM model
        print("\nTraining LSTM model...")
        lstm_model, lstm_history = train_lstm_model(X_train, y_train, X_val, y_val)
        models['lstm'] = lstm_model
        plot_training_history(lstm_history, title=f"LSTM Training History - Segment {segment}")
        
        # Train GRU model
        print("\nTraining GRU model...")
        gru_model, gru_history = train_gru_model(X_train, y_train, X_val, y_val)
        models['gru'] = gru_model
        plot_training_history(gru_history, title=f"GRU Training History - Segment {segment}")
        
        # Train ARIMA model
        print("\nTraining ARIMA model...")
        # Convert tensor to numpy array
        y_train_np = y_train.cpu().numpy()
        # Use auto_arima to find best parameters
        arima_model = auto_arima_model(y_train_np.flatten())
        models['arima'] = arima_model
        
        # Train Random Forest model
        print("\nTraining Random Forest model...")
        rf_model = train_rf_model(X_train, y_train)
        models['rf'] = rf_model
        
        # Train SVR model
        print("\nTraining SVR model...")
        svr_model = train_svr_model(X_train, y_train)
        models['svr'] = svr_model
        
        # Step 6: Model Comparison and Evaluation
        print(f"\n--- Step 6: Model Comparison and Evaluation for Segment {segment} ---")
        results = evaluate_models(models, X_test, y_test)
        plot_model_comparison(results, metric=['mae', 'rmse'])
        
        # Save models for this segment
        all_models[segment] = models
        
        # Step 7: Generate Predictions for Test Data
        print(f"\n--- Step 7: Generating Predictions for Test Data - Segment {segment} ---")
        
        # Generate predictions
        segment_predictions = generate_predictions(models, X_test, scaler, segment)
        
        # Create ensemble prediction
        segment_predictions = create_ensemble_prediction(segment_predictions)
        
        # Save predictions for this segment
        all_predictions[segment] = segment_predictions
    
    # Step 8: Create Submission
    print("\n--- Step 8: Creating Submission ---")
    
    # Prepare submission file
    submission = prepare_submission(all_predictions, 
                                   {1: df_test[df_test['segment'] == 1]['application_date'].values,
                                    2: df_test[df_test['segment'] == 2]['application_date'].values}, 
                                   df_sample_sub)
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")
    
    print("\nTime Series Forecasting Pipeline completed successfully.")

if __name__ == "__main__":
    main()

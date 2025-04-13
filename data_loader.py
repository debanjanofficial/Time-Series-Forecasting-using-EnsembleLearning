import pandas as pd
import numpy as np
from datetime import datetime
from config import TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH
import os

def load_data():
    """Load the train, test and sample submission data"""
    try:
        df_sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        df_test = pd.read_csv(TEST_PATH)
        df_train = pd.read_csv(TRAIN_PATH)
        
        print(f"Train data shape: {df_train.shape}")
        print(f"Test data shape: {df_test.shape}")
        print(f"Sample submission shape: {df_sample_sub.shape}")
        
        return df_train, df_test, df_sample_sub
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please check the file paths in the config.py file.")
        return None, None, None

def preprocess_data(df_train, df_test):
    """Preprocess the train and test data"""
    if df_train is None or df_test is None:
        return None, None, None
    
    # Convert dates to datetime
    df_train['application_date'] = pd.to_datetime(df_train['application_date'])
    df_test['application_date'] = pd.to_datetime(df_test['application_date'])
    
    # Group by date and segment
    df_train = df_train.sort_values('application_date').groupby(
        ['application_date', 'segment'], as_index=False
    )
    df_train = df_train.agg({'case_count': ['sum']})
    df_train.columns = ['application_date', 'segment', 'case_count']
    
    # Print date ranges
    print(f"Train date range: {df_train.application_date.min()} to {df_train.application_date.max()}")
    print(f"Test date range: {df_test.application_date.min()} to {df_test.application_date.max()}")
    
    # Split by segment
    df_train_1 = df_train[df_train['segment'] == 1]  # Segment 1 records
    df_train_2 = df_train[df_train['segment'] == 2]  # Segment 2 records
    
    print(f"Segment 1 records: {len(df_train_1)}")
    print(f"Segment 2 records: {len(df_train_2)}")
    
    return df_train, df_test, df_train_1, df_train_2

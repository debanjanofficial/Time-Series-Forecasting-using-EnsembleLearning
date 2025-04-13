import pandas as pd
import numpy as np
from datetime import datetime
from config import TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH

def load_data():
    """Load the train, test and sample submission data"""
    try:
        df_sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        df_test = pd.read_csv(TEST_PATH)
        df_train = pd.read_csv(TRAIN_PATH)
        
        return df_train, df_test, df_sample_sub
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None

def preprocess_data(df_train, df_test):
    """Preprocess the train and test data"""
    if df_train is None or df_test is None:
        return None, None, None
    
    # Convert dates to datetime
    df_train['application_date'] = pd.to_datetime(df_train['application_date'])
    df_test['application_date'] = pd.to_datetime(df_test['application_date'])
    
    # Aggregate daily cases by segment for segment 1
    daily_cases_1 = df_train[df_train['segment'] == 1].groupby(
        ['branch_id', 'state', 'zone', 'application_date'], 
        as_index=False
    )['case_count'].sum()
    
    # Aggregate daily cases by segment for segment 2
    daily_cases_2 = df_train[df_train['segment'] == 2].groupby(
        ['state', 'application_date'], 
        as_index=False
    )['case_count'].sum()
    
    # Aggregate train data by date and segment
    df_train_agg = df_train.sort_values('application_date').groupby(
        ['application_date', 'segment'], 
        as_index=False
    ).agg({'case_count': ['sum']})
    
    df_train_agg.columns = ['application_date', 'segment', 'case_count']
    
    # Split by segment
    df_train_seg1 = df_train_agg[df_train_agg['segment'] == 1]
    df_train_seg2 = df_train_agg[df_train_agg['segment'] == 2]
    
    return df_train_seg1, df_train_seg2, df_test, daily_cases_1, daily_cases_2

def get_date_info(df_train_seg1, df_train_seg2, df_test):
    """Get information about dates in the datasets"""
    if df_train_seg1 is None or df_train_seg2 is None or df_test is None:
        return None
    
    max_date_train = pd.to_datetime(df_train_seg1['application_date'].max()).date()
    max_date_test = pd.to_datetime(df_test['application_date'].max()).date()
    lag_size = (max_date_test - max_date_train).days
    
    date_info = {
        'min_date_train': pd.to_datetime(df_train_seg1['application_date'].min()).date(),
        'max_date_train': max_date_train,
        'max_date_test': max_date_test,
        'forecast_lag': lag_size,
        'seg1_date_range': (df_train_seg1['application_date'].max() - df_train_seg1['application_date'].min()).days,
        'seg2_date_range': (df_train_seg2['application_date'].max() - df_train_seg2['application_date'].min()).days
    }
    
    return date_info

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_time_series(data, column, segment=None, title=None):
    """
    Plot a time series.
    
    Args:
        data: DataFrame containing the time series
        column: Column to plot
        segment: If not None, filter by this segment
        title: Title for the plot
    """
    plt.figure(figsize=(12, 6))
    
    if segment is not None:
        data = data[data['segment'] == segment]
        if title is None:
            title = f"Time Series - {column} (Segment {segment})"
    
    if title is None:
        title = f"Time Series - {column}"
    
    sns.lineplot(x='application_date', y=column, data=data)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_predictions(actual, predictions, dates, title="Actual vs Predicted"):
    """
    Plot actual vs predicted values.
    
    Args:
        actual: Actual values
        predictions: Predicted values
        dates: Dates corresponding to values
        title: Title for the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual', marker='o')
    plt.plot(dates, predictions, label='Predicted', marker='x')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results, metric=None):
    """
    Plot model comparison.
    
    Args:
        results: Dictionary of model results
        metric: Metric to use for comparison
    """
    if metrics is None:
        metrics = ['mape']
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 6*len(metrics)))
    
    # Handle the case of a single metric (axes won't be array)
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        models = list(results.keys())
        # Filter out models that don't have this metric
        models = [model for model in models if metric in results[model]]
        values = [results[model][metric] for model in models]
        
        # Create bar chart
        bars = axes[i].bar(models, values)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom')
        
        # Add labels and title
        axes[i].set_title(f'Model Comparison - {metric.upper()}')
        axes[i].set_xlabel('Models')
        axes[i].set_ylabel(metric.upper())
    
    plt.tight_layout()
    plt.show()

def plot_training_history(history, title="Training History"):
    """
    Plot training history for neural network models.
    
    Args:
        history: Training history (loss values)
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

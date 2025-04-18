# Time-Series-Forecasting-using-LSTMs-ARIMA

## Objective

This project is a modular, multi-model approach to time series forecasting, focusing on predicting case counts for two distinct segments. Using a combination of deep learning (LSTM, GRU), traditional statistical methods (ARIMA), and machine learning techniques (Random Forest, SVR), I created an ensemble forecasting system capable of generating predictions over multiple time horizons.


## Project Structure

project/
│
├── config.py               # Configuration parameters
├── data_loader.py          # Data loading and preprocessing
├── data_preparation.py     # Time series data preparation
├── visualization.py        # Plotting functions
├── models/
│   ├── __init__.py
│   ├── lstm_model.py       # LSTM implementation
│   ├── gru_model.py        # GRU implementation
│   ├── arima_model.py      # ARIMA implementation
│   ├── rf_model.py         # Random Forest implementation
│   └── svr_model.py        # SVR implementation
├── evaluation.py           # Model evaluation metrics
├── prediction.py           # Generate predictions
└── main.py                 # Main script to run the process




## FlowChart
+----------------+     +--------------------+     +------------------------+
| Data Loading   |---->| Data Preprocessing |---->| Exploratory Data       |
+----------------+     +--------------------+     | Analysis (EDA)         |
                                                  +------------------------+
                                                            |
                                                            v
+----------------+     +--------------------+     +------------------------+
| Create         |<----| Generate           |<----| Time Series Data       |
| Submission     |     | Predictions        |     | Preparation            |
+----------------+     +--------------------+     +------------------------+
                             ^                             |
                             |                             v
                      +------+------+             +------------------------+
                      | Train Final  |<-----------| Model Comparison       |
                      | Models       |            | and Evaluation         |
                      +-------------+             +------------------------+
                                                            ^
                                                            |
                        +------------+------------+------------+------------+
                        |            |            |            |            |
                   +--------+   +--------+   +--------+   +--------+   +--------+
                   |  LSTM  |   |  GRU   |   | ARIMA  |   | Random |   |  SVR   |
                   | Model  |   | Model  |   | Model  |   | Forest |   | Model  |
                   +--------+   +--------+   +--------+   +--------+   +--------+

## Dataset

Dataset has two segment
Segment 1-
Segment 2-

Minimum date from training set: 2017-04-01
Maximum date from training set: 2019-07-23

Maximum date from training set: 2019-07-23
Maximum date from test set: 2019-10-24
Forecast Lag: 93


## Key Achievements
1. Modular Architecture: Successfully implemented a well-structured, modularized codebase following software engineering best practices, which separated concerns into data loading, preprocessing, model definition, training, evaluation, and prediction components.
2. Multi-Model Ensemble: Developed a comprehensive ensemble approach that leverages the strengths of five different forecasting methodologies, providing more robust predictions than any single model could offer.
3. Hardware Optimization: Leveraged Apple's Metal Performance Shaders (MPS) for GPU acceleration on M1 Mac, demonstrating the ability to utilize platform-specific optimizations for improved performance.
4. Time Series Visualization: Created informative visualizations that reveal clear patterns in the forecasting results, providing insights into the behavior of each segment over time.
5. Flexible Prediction Pipeline: Implemented a robust prediction system capable of generating forecasts for specific test periods using a rolling window approach, ensuring alignment with submission requirements.


## Statistical Analysis of Predictions

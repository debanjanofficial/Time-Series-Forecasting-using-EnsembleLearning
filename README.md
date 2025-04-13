# Time-Series-Forecasting-using-LSTMs-ARIMA


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

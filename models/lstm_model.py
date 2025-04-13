import torch
import torch.nn as nn
import torch.optim as optim
import config
from data_preparation import get_device

class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=1, output_dim=1, dropout=0.2):
        """
        Initialize LSTM model.
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass.
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: [batch_size, seq_len, hidden_dim]
        
        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out

def train_lstm_model(X_train, y_train, X_val, y_val, params=None):
    """
    Train LSTM model.
    """
    # Set device
    device = get_device()
    
    # Set model parameters
    if params is None:
        params = {
            'hidden_dim': config.LSTM_HIDDEN_DIM,
            'num_layers': config.LSTM_NUM_LAYERS,
            'dropout': config.LSTM_DROPOUT,
            'learning_rate': config.LSTM_LEARNING_RATE,
            'epochs': config.LSTM_EPOCHS,
            'batch_size': config.LSTM_BATCH_SIZE
        }
    
    # Create the model
    input_dim = X_train.shape[2]  # Number of features
    output_dim = 1
    
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        output_dim=output_dim,
        dropout=params['dropout']
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(params['epochs']):
        # Set model to training mode
        model.train()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record training loss
        history['train_loss'].append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            history['val_loss'].append(val_loss.item())
        
        # Print progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{params["epochs"]}], '
                  f'Train Loss: {loss.item():.4f}, '
                  f'Val Loss: {val_loss.item():.4f}')
    
    return model, history

def predict_lstm(model, X_test):
    """
    Generate predictions using LSTM model.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    return predictions

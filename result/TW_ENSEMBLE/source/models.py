import torch
from torch import nn
#================================== models ==================================#
class CNN_LSTM(nn.Module):
    def __init__(
        self,
        num_features: int,
        cnn_out_channels: int=64,
        kernel_size: int=1,
        hidden_size=100
    ):
        self.kernel_size = kernel_size
        
        super().__init__()
        self.cnn = nn.Conv1d(
            in_channels=num_features,
            out_channels=cnn_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )
                        
        self.dropout1d = nn.Dropout1d(p=0.2) # Make the whole dimension value = 0
        self.dropout = nn.Dropout(p=0.2) # Make single value = 0
        
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels, 
            hidden_size=hidden_size, 
            num_layers=3, 
            batch_first=True, 
            dropout=0.2
        )
        
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input:    (batch, time, features)
        # required: (batch, features, time)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        
        # CNN let some vector channels go to 0 -> decrease dependencies on certain channel vector
        x = self.dropout1d(x) 
        
        # LSTM batch first=True --> (batch, time, features)
        # But now is (batch, channels, time)
        x = x.transpose(1, 2)
        output, (h_n, c_n) = self.lstm(x)
        # Take the last layer, last time period output
        x = output[:, -1, :] # (batch, 100) (time is removed)
        # Can only drop one value, if use dropout1d, only remains batch with all 0 values
        x = self.dropout(x)
        prediction = self.linear(x)
        
        return prediction
        

class ENSEMBLE(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int=128,
        num_layers: int=1,
        device: str = "cpu"
    ):
        super().__init__()
        """
        batch_first=True --> Cuz data loader makes batch at the first dimension
        """
        
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.rnn = nn.RNN(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.dense = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=32),
            nn.Linear(in_features=32, out_features=1),
        )
        
        self.dropout = nn.Dropout(p=0.2)
        
        # The paper specifically asked for the initialzation
        for name, param in self.named_parameters():  # look all the parameters
            if "weight" in name:                    # If it is weight
                nn.init.xavier_uniform_(param)       # Use Xavier to randomly initialize value
            elif "bias" in name:                    # If it is bias (Something like the b in y=ax + b)
                nn.init.zeros_(param)               # Set all the initial values to 0
                
        # Make it runs on device
        self.to(device=device)
        
    
    def forward(self, x):
        # Input shape: [64, 21, 4] = [num of batches, time_span, num of features]
        # Output shape: 
        #   - out: [64, 21, 128] = [same, same, hidden_size]
        #   - h  : [1, 64, 128]  = [last time step, batches, hidden_size]
        # The return shape: [64, 1]
        out_gru, h_gru = self.gru(x)
        out_rnn, h_rnn = self.rnn(x)
        out_lstm, (h_lstm, _) = self.lstm(x)
        
        x = (h_gru[-1] + h_rnn[-1] + h_lstm[-1]) / 3
        x = self.dropout(x)
        prediction = self.dense(x)
        
        return prediction
    

class CNN_ENSEMBLE(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int=128,
        cnn_out_channels: int=128,
        kernel_size: int=5,
        num_layers: int=1,
        device: str = "cpu"
    ):
        super().__init__()
        """
        batch_first=True --> Cuz data loader makes batch at the first dimension
        """
        self.cnn = nn.Conv1d(
            in_channels=num_features,
            out_channels=cnn_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.rnn = nn.RNN(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.dense = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=32),
            nn.Linear(in_features=32, out_features=1),
        )
        
        self.dropout = nn.Dropout(p=0.2)
        
        # The paper specifically asked for the initialzation
        for name, param in self.named_parameters():  # look all the parameters
            if "weight" in name:                    # If it is weight
                nn.init.xavier_uniform_(param)       # Use Xavier to randomly initialize value
            elif "bias" in name:                    # If it is bias (Something like the b in y=ax + b)
                nn.init.zeros_(param)               # Set all the initial values to 0
                
        # Make it runs on device
        self.to(device=device)
        
    
    def forward(self, x):
        # Fit CNN input [batch, features, time_sequence]
        x = torch.transpose(x, 1, 2)
        x = self.cnn(x)
        
        # Return back to normal shape
        x = torch.transpose(x, 1, 2)
        out_gru, h_gru = self.gru(x)
        out_rnn, h_rnn = self.rnn(x)
        out_lstm, (h_lstm, _) = self.lstm(x)
        
        x = (h_gru[-1] + h_rnn[-1] + h_lstm[-1]) / 3
        x = self.dropout(x)
        prediction = self.dense(x)
        
        return prediction
    
    


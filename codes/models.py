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
        
        self.relu = nn.ReLU()
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
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
        x = self.relu(x)
        
        # If time_span = 1 --> (batch, features, time) time only equals 1
        # It cannot do pool with kernel 2 cuz only 1 value in that time
        if x.shape[-1] >= self.kernel_size: # time
            x = self.pool(x)
        
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
        


        
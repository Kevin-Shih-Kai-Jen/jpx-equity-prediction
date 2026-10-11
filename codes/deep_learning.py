import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from copy import deepcopy


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
        
        self.pool = nn.MaxPool1d(kernel_size=kernel_size)
        
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
        

def make_it_tensor(
    array: list | np.ndarray | pd.Series | pd.DataFrame
) -> torch.tensor:
    if isinstance(array, list):
        array = np.array(array)
    
    if isinstance(array, pd.Series) or isinstance(array, pd.DataFrame):
        array = array.to_numpy()
    
    
    return torch.from_numpy(array.astype(np.float32))
    

def make_loader(X, y, batch_size=64, shuffle=False):
    # DataFrame -> Numpy -> Tensor (DataFrame cannot change to Tensor directly)
    X = make_it_tensor(X)
    y = make_it_tensor(y)
    
    # (samples,) -> (samples, 1)
    # For matching the dimension of output from CNN-LSTM of (64, 1) 
    # While original y's shape is (64,) only
    if y.ndim == 1:
        y = y.unsqueeze(1)
        
    # Match answer with preidction
    dataset = TensorDataset(X, y)
    
    # Make it a dataloader
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def train(model, data_loader, loss_fn, optimizer, epoch=1, device="cpu"):
    model.train()
    model = model.to(device)
    
    total_loss = 0.0
    total_sample = 0
    
    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        prediction = model(X_batch)
        
        loss = loss_fn(prediction, y_batch)
        
        loss.backward()
        
        optimizer.step()
        
        batch_size = X_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_sample += batch_size
    
    if total_sample == 0:
        raise ValueError("Missing data (sample) for training")
    
    return total_loss / total_sample 


def evaluation(
    model: nn.Module, 
    data_loader: DataLoader, 
    loss_fn: nn.Module,
    device: str = "cpu"
) -> float:
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_sample = 0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            
            batch_size = X_batch.shape[0]
            total_loss += loss.item() * batch_size
            total_sample += batch_size
    
    if total_sample == 0:
        raise ValueError("Missing data (sample) for training")
    
    return total_loss / total_sample

import torch
from torch import nn
import torch.nn.functional as F
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
    
    
class MASTER(nn.Module):
    def __init__(
        self,
        gating_input_dim=12,
        gating_output_dim=6,
        beta=5.0,
        hidden_dim=128,
        intra_nhead=4,
        inter_nhead=2,
        dim_feedforward=256,
        activation_function="relu"        
    ):
        super().__init__()
        self.linear_gate = nn.Linear(gating_input_dim, gating_output_dim)
        self.beta = beta
        self.feature_projection = nn.Linear(
            gating_output_dim,
            hidden_dim
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.intra_stock_transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=intra_nhead,
            dim_feedforward=dim_feedforward,
            activation=activation_function,
            batch_first=True
        )
        
        self.inter_stock_transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=inter_nhead,
            dim_feedforward=dim_feedforward,
            activation=activation_function,
            batch_first=True
        )
        
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.linear_final_prediction = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Input:
            - features: (N, F_G)
            - x: (N, T, F)
        
        Output:
            - out: (N, )
        """
        x = self.gating(x)    
        x = self.feature_embedding(x)
        x = self.intra_stock_aggregation(x)
        x = self.inter_stock_aggregation(x)
        x = self.temporal_transformer(x)
        out = self.final_prediction(x)
        return out

    def gating(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Put the overall market features into linear function to change values of 
        features from x, Use average and volatility of normalized data to how extreme 
        different data is. (Mean and its fluctuations)
        
        Input:
            - features: (N, F_G)
            - x: (N, T, F)
        
        Output:
            - x: (N, T, F)
        """
        if self.beta <= 0:
            raise ValueError(f"beta should be greater than 0")
        
        lastest = x[:, -1, :] # (N, F)
        mean = lastest.mean(dim=0) # (F,)
        rms = lastest.square().mean(dim=0).sqrt() # (F,)
        market_features = torch.cat(
            [mean, rms],
            dim=-1
        )
        
        scores = x.size(-1) * F.softmax(
            self.linear_gate(market_features) / self.beta, 
            dim=-1
        )
        # Change scores to (1, 1, F)
        # -1: Remain the length of the current features
        x = x * scores.view(1, 1, -1)
        
        return x
    
    def feature_embedding(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Input:
            x: (N, T, F)

        Steps:
            1. Linear projection: F → D
            2. Generate sinusoidal positional encoding
            3. Add positional encoding
            4. Layer normalization

        Output:
            x: (N, T, D)
        """
        x = self.feature_projection(x)
        
        N, T, D = x.shape
        pos_T = torch.arange(start=0, end=T, step=1, device=x.device, dtype=x.dtype).unsqueeze(1)
        dim = torch.arange(0, D, 2, device=x.device, dtype=x.dtype) # (D/2,)
        freq = 1 / (10000 ** (dim / D))
        freq = freq.repeat_interleave(repeats=2) # (D, )
        
        angles = pos_T * freq # (T, D)
        
        angles[:, 0::2] = torch.sin(angles[:, 0::2])
        angles[:, 1::2] = torch.cos(angles[:, 1::2])
        
        angles = angles.unsqueeze(0) # (1, T, D)
        
        return self.input_norm(x + angles)

    def intra_stock_aggregation(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Use transformer to find relationships of a stock with lookback period data
         
        Input:
            - x: (N, T, D)
            
        Output:
            - x: (N, T, D)
        """
        return self.intra_stock_transformer(x)
    
    def inter_stock_aggregation(
            self,
            x: torch.Tensor
        ) -> torch.Tensor:
            """
            Use transformer to find relationships of different stocks
             
            Input:
                - x: (N, T, D)
                
            Change to:
                - x: (T, N, D)
            
            Output:
                - x: (N, T, D)
            """
            x = x.transpose(0, 1)
            x = self.inter_stock_transformer(x)
            x = x.transpose(0, 1)
            
            return x
    
    def temporal_transformer(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Take last time period embedding and @ with W, then dot product with all
        time period's dimension, softmax the score, and do attention_weighted sum
        
        Input: 
            - x: (N, T, D)
        
        Output:
            - x: (N, D)
        """
        last = x[:, -1, :]             # (N, D)
        last = self.W(last)            # (N, D)

        scores = torch.matmul(
            x, last.unsqueeze(-1)
        ).squeeze(-1)                   # (N, T)

        weights = F.softmax(scores, dim=1)  # (N, T)

        x = x * weights.unsqueeze(-1)  # (N, T, D)
        x = x.sum(dim=1)               # (N, D)

        return x



    def final_prediction(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Input: 
            - x: (N, D)
        
        Output:
            - x: (N, 1)
        """
        return self.linear_final_prediction(x)
        

class TimeGrad(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 40,
        lstm_num_layers: int = 1,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        total_denoise_steps: int = 100,
        n_linear_dim: int = 32,
    ):
        """
        Use diffusion model to predict Target, is designed for only 1 day prediction
        B: batch, 
        T: time series data
        F: features
        """
        super().__init__()
        self.lstm = nn.LSTM(
            num_features,
            hidden_size,
            lstm_num_layers,
            batch_first=True
        )
        
        self.predict_noise_model = nn.Sequential(
            nn.Linear(hidden_size + n_linear_dim + 1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        
        self.loss_fn = nn.MSELoss()
        self.n_linear = nn.Linear(1, n_linear_dim)
        beta = torch.linspace(beta_start, beta_end, total_denoise_steps)
        alpha = 1 - beta
        alpha_cum = torch.cumprod(alpha, dim=-1)
        prev_alpha_cum = torch.concat(
            [
                torch.ones_like(alpha_cum[:1]), # [:1] will remain shape while [0] won't
                alpha_cum[:-1]
            ],
            dim=0
        )
        
        sigma = torch.sqrt(
            beta * (1 - prev_alpha_cum) / (1 - alpha_cum)
        )
        
        # register_buffer:
        # Will be saved to state_dcit, will not be in gradient descent,
        # Can be shift to GPU (Better than torch.tensor)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cum", alpha_cum)
        self.register_buffer("sigma", sigma)
        self.total_denoise_steps = total_denoise_steps
    
    def forward(
        self,
        data_history: torch.Tensor,
        loss_fn: nn.Module,
        target: torch.Tensor | None=None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Input
            - data_history (B, T, F)
            - target (B, 1)
        
        Output
            - Training
                - predicted_noise (B, 1)
                - loss (B, 1)
            
            - Inferencing
                - predicted_target (B, 1)
        """
        training_mode = True if target is not None else False
        
        if training_mode:
            return self.training_flow(data_history, loss_fn, target)
        return self.inferencing_flow(data_history)
    
    def add_noise(
        self,
        target: torch.Tensor,
        n: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        During training, Target should be added with noise for training denoise
        models to denoise.
        
        Input
            - target (B, 1)
            - n (B, 1)
            - noise (B, 1)
        
        Output
            - noisy_target (B, 1)
        """
        alpha_cum = self.alpha_cum[n - 1]
        noisy_target = (
            torch.sqrt(alpha_cum) * target +
            torch.sqrt(1 - alpha_cum) * noise
        )
        
        return noisy_target
    
    def encode_history(
        self,
        data_history: torch.Tensor
    ) -> torch.Tensor:
        """
        Input data history into LSTM model to generate hidden state
        
        Input
            - data_history (B, T, F)
        
        Output
            - hidden state (B, hidden_size)
        """
        _, (h_n, _) =  self.lstm(data_history)
        return h_n[-1]
    
    def batch_data_preprocessor(
        self,
        data_history: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Do
            - data history (B, T, F), 
            - do add_noise() on y, 
            - randomly pick noise step n, 
            - randomly generate noise values.
        
        Do
            - Randomly pick denoise steps n
            - Generate standard Gaussian noise with the same shape as y
            - Implement add_noise(real Target, N)
        
        Input
            - data_history (B, T, F)
            - target (B, 1)
        
        Output
            - noisy_target (B, 1)
            - noise (B, 1)
            - hidden (B, hidden_size)
            - n (B, 1)
        """
        hidden = self.encode_history(data_history)
        n = torch.randint(
            low=1,
            high=self.total_denoise_steps + 1,
            size=[target.shape[0], 1],
            device=target.device,
            dtype=torch.long
        )
        noise = torch.randn_like(target)
        noisy_target = self.add_noise(
            target,
            n,
            noise
        )
        
        return noisy_target, noise, hidden, n
    
    def remove_noise(
        self,
        noisy_target: torch.Tensor,
        n: torch.Tensor,
        predicted_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform 1 reverse diffusion step.
        
        Input
            - noisy_target (B, 1)
            - n (B, 1)
            - predicted_noise (B, 1)
        
        Output
            - removed_noise_target (B, 1)
        
        y_{n-1} = (y_n - beta_n / sqrt(1 - alpha_bar_n) * predicted_noise) \
#           / sqrt(alpha_n) + sigma_n * z
        """
        mean = (
            (
                noisy_target - (
                    self.beta[n - 1] / 
                    torch.sqrt(1 - self.alpha_cum[n - 1]) * 
                    predicted_noise
                )
            ) / torch.sqrt(self.alpha[n - 1])
        )
        
        denoised_target = mean + self.sigma[n - 1] * torch.randn_like(noisy_target)
        
        return denoised_target
    
    def predict_noise(
        self,
        noisy_target: torch.Tensor,
        n: torch.Tensor,
        hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict the noise of current Target.
        
        Input:
            - noisy_target (B, 1)
            - n (B, 1)
            - hidden (B, T, F)
        
        Output
            - predicted_noise (B, 1)
        """
        # Float type matches linear projection's required dtype
        n_embedding = self.n_linear(n.to(dtype=hidden.dtype))
        input = torch.concat(
            [
                noisy_target,
                n_embedding,
                hidden
            ],
            dim=1
        )
        
        return self.predict_noise_model(input)
        
    def training_flow(
        self,
        data_history: torch.Tensor,
        loss_fn: nn.Module,
        target: torch.Tensor,      
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Do
            - Do batch_data_preprocessor
            - Train the MLP, LSTM model.
            - Predict noise
            - Calculate loss
        
        Input
            - data_history (B, T, F)
            - target (B, 1)
        
        Output
            - predicted_noise (B, 1)
            - loss (Scalar tensor)
            
        """
        noisy_target, noise, hidden, n = self.batch_data_preprocessor(
            data_history, 
            target
        )
        predicted_noise = self.predict_noise(noisy_target, n, hidden)
        loss = self.calculate_loss(predicted_noise, loss_fn, noise) 
        return predicted_noise, loss

    @torch.no_grad()
    def inferencing_flow(
        self,
        data_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Do
            - Generate noise from Gaussian Distribution(mean=0, var=1)
            - Start from steps = N, and denoise until step = 0
        
        Input
            - data_history (B, T, F)
        
        Output
            - predicted_target (B, 1)
        """
        noisy_target = torch.randn(
            data_history.shape[0], 1,
            device=data_history.device,
            dtype=data_history.dtype
        )
        hidden = self.encode_history(data_history)
        
        for step in range(self.total_denoise_steps, 0, -1):
            # Create a (B, 1) integer tensor filled with the current step
            n = torch.full(
                [data_history.shape[0], 1],
                step,
                device=data_history.device,
                dtype=torch.long 
            )
            
            # Match the floating-point dtype used by the model
            predicted_noise = self.predict_noise(noisy_target, n, hidden)
            noisy_target = self.remove_noise(noisy_target, n, predicted_noise)
            
        return noisy_target
    
    def calculate_loss(
        self,
        predictions: torch.Tensor,
        loss_fn: nn.Module,
        target: torch.Tensor,
    ):
        return loss_fn(predictions, target)
        
    
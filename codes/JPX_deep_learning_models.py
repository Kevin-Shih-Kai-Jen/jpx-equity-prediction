import torch
from torch import nn
import numpy as np
import time

import deep_learning
from data_loader import main_loader
import data_processor
import features

COL_NAMES = ["SecuritiesCode", "Date", "Open", "High", "Low", "Close", "Volume", "AdjustmentFactor", "Target"]
FEAUTRES = ["Open", "High", "Low", "Close", "Volume", "ExpectedDividend"]
TIME_SPAN = [42]


def print_running_time(func):
    def inner_func():
        print("\n============================== Program starts running ==============================\n")
        start_time = time.time()
        func()
        end_time = time.time()
        print("\n============================== Program finished running ==============================\n")
        diff = end_time - start_time
        hr = diff // 3600
        min = (diff - hr * 3600) // 60
        sec = diff - hr * 3600 - min * 60
        print(f"Total runtime: {hr}h, {min}m, {sec}s")
    return inner_func
        

@print_running_time
def model_1():  
    df = main_loader("/Users/coolguy/Desktop/JPX/data/stock_prices.csv", show_basic_data=False)
    
    df["ExpectedDividend"] = df["ExpectedDividend"].fillna(0)
    print("Stage 1: Loaded", flush=True)

    df = features.get_valid_rows(df, set(COL_NAMES))
    
    print("Stage 2: Validated", flush=True)

    df = data_processor.correct_prices_with_factors(df)

    print("Stage 3: Prices adjusted", flush=True)
    
    df, mean, std = features.normalize_data(
        df,
        ["Open", "High", "Low", "Close", "Volume", "ExpectedDividend"]
    )
    
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Stage 4: Normalized", flush=True)
    for time in TIME_SPAN:
        X, y = features.create_time_sequence(df, time_sequence=time, col_names=FEAUTRES)

        print("Stage 5: Created time sequence", flush=True)
        data_loader = deep_learning.make_loader(X, y)
        print("Stage 6: Data loaded", flush=True)
        model = deep_learning.CNN_LSTM(
            num_features=len(FEAUTRES),
            cnn_out_channels=64,
            kernel_size=time,
            hidden_size=100
        )
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters())
        
        train_avg_loss = deep_learning.train(model, data_loader, loss_fn, optimizer, epoch=10, device=device)
        print(f"CNN-LSTM {time} time span training average loss: {train_avg_loss}")
        eval_avg_loss = deep_learning.evaluation(model, data_loader, loss_fn, device)
        print(f"CNN-LSTM {time} time span evaluation average loss: {eval_avg_loss}")

    # Compare with all predictions = 0
    y_p = np.asarray(y, dtype="float32")
    baseline_mae = np.mean(np.abs(y_p))
    print(f"Zero baseline mae: {baseline_mae}")
    
    #========================================================#
    diagnose_model(model, data_loader)
    
    
#====================================== Diagnosing model ======================================#
def diagnose_model(model, validation_loader):
    model.eval()

    device = next(model.parameters()).device

    predictions = []
    targets = []

    with torch.inference_mode():
        for X_batch, y_batch in validation_loader:

            prediction = model(X_batch.to(device))

            predictions.append(prediction.cpu().reshape(-1))
            targets.append(y_batch.cpu().reshape(-1))

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    assert predictions.shape == targets.shape

    print("===== Full Validation Results =====")

    print("Prediction mean:", predictions.mean().item())
    print("Prediction std:", predictions.std().item())

    print("Target mean:", targets.mean().item())
    print("Target std:", targets.std().item())

    model_mae = torch.abs(predictions - targets).mean()
    zero_mae = torch.abs(targets).mean()

    print("Model MAE:", model_mae.item())
    print("Zero baseline MAE:", zero_mae.item())

    print(
        "MAE Improvement:",
        ((zero_mae - model_mae) / zero_mae * 100).item(),
        "%"
    )

    correlation = torch.corrcoef(
        torch.stack([predictions, targets])
    )[0, 1]

    print("Prediction-Target Correlation:", correlation.item())
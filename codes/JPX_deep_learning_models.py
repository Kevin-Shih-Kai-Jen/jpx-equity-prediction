import torch
from torch import nn
import pandas as pd

import deep_learning
import models
import features
import evaluation
from data_loader import main_loader
import data_processor
import utils

COL_NAMES = ["SecuritiesCode", "Date", "Open", "High", "Low", "Close", "Volume", "AdjustmentFactor", "Target"]
FEAUTRES = ["Open", "High", "Low", "Close", "Volume", "ExpectedDividend"]
TIME_SPAN = [42]

    
def common_process(
    train_and_val: list[str],
    col_names: list[str]
):
    df = main_loader("/Users/coolguy/Desktop/JPX/data/stock_prices.csv", show_basic_data=False)
    
    df["ExpectedDividend"] = df["ExpectedDividend"].fillna(0)
    print("Stage 1: Loaded", flush=True)

    valid_index = features.get_valid_rows(df, set(col_names)).index
    
    print("Stage 2: Validated", flush=True)

    df = data_processor.correct_prices_with_factors(df)
    
    df = df.loc[valid_index].copy()

    print("Stage 3: Prices adjusted", flush=True)
    
    # Get train date and evaluation date
    train_start, train_end, val_start, val_end = train_and_val
    train_df = data_processor.get_time_for_specific_period(df, train_start, train_end)
    val_df = data_processor.get_time_for_specific_period(df, val_start, val_end)
    
    # Calculate mean and std with training data
    train_df, mean, std = features.normalize_data(
        train_df,
        FEAUTRES
    )
    
    # Use the same mean and std for validation data
    val_df, _, _ = features.normalize_data(
        val_df,
        FEAUTRES,
        std=std,
        mean=mean
    )
    
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Stage 4: Normalized", flush=True)
    
    return train_df, val_df, device, mean, std


@utils.print_running_time
def model_1():  
    df, _, device, _, _ = common_process(["2017-01-01", "2021-07-01","2017-01-01", "2021-07-01"], COL_NAMES)
    
    for time in TIME_SPAN:
        X, y = features.create_time_sequence(df, time_sequence=time, col_names=FEAUTRES)

        print("Stage 5: Created time sequence", flush=True)
        data_loader = deep_learning.make_loader(X, y)
        print("Stage 6: Data loaded", flush=True)
        model = models.CNN_LSTM(
            num_features=len(FEAUTRES),
            cnn_out_channels=64,
            kernel_size=time,
            hidden_size=100
        )
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters())
        
        train_avg_loss = deep_learning.train(model, data_loader, loss_fn, optimizer, epochs=1, device=device)
        print(f"CNN-LSTM {time} time span training average loss: {train_avg_loss}")
        eval_avg_loss, all_predictions = deep_learning.evaluation(model, data_loader, loss_fn, device)
        print(f"CNN-LSTM {time} time span evaluation average loss: {eval_avg_loss}")

        eval_results = evaluation.overall_performance(df, all_predictions, time)
    
    evaluation.zero_baseline(y)
    
    
@utils.print_running_time
def model_2():
    #================================ Expanding window ================================#
    # Half of the year for validation
    splits = utils.create_start_end_date(
        start_time="2017-01-01",
        end_time="2021-07-01",
        gap=180,
        mode="expanding_window"
    )
    
    for time in TIME_SPAN:
        # Record all validations results
        total_earnings, track_all_predictions = 0, []
        
        model = models.CNN_LSTM(
            num_features=len(FEAUTRES),
            cnn_out_channels=64,
            kernel_size=3,
            hidden_size=100
        )
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters())
        
        for train_and_val in splits:
            train_df, val_df, device, mean, std = common_process(train_and_val, COL_NAMES)
            val_df = utils.compensate_val_lost_time(train_df, val_df, time)
            
            # Since the last train target uses eval's first day and second day close price to calculate the target
            # We have to abandon it to avoid using seen values
            cutoff = train_df["Date"].drop_duplicates().sort_values().iloc[-2]
            train_df = train_df.loc[train_df["Date"] < cutoff]
            
            X_train, y_train = features.create_time_sequence(train_df, time_sequence=time, col_names=FEAUTRES)
            X_val, y_val = features.create_time_sequence(val_df, time_sequence=time, col_names=FEAUTRES)

            print("Stage 5: Created time sequence", flush=True)
            
            train_data_loader = deep_learning.make_loader(X_train, y_train)
            val_data_loader = deep_learning.make_loader(X_val, y_val)

            print("Stage 6: Data loaded", flush=True)
            
            train_avg_loss = deep_learning.train(model, train_data_loader, loss_fn, optimizer, epochs=20, device=device)
            print(f"CNN-LSTM {time} time span training average loss: {train_avg_loss}")
            eval_avg_loss, all_predictions = deep_learning.evaluation(model, val_data_loader, loss_fn, device)
            print(f"CNN-LSTM {time} time span evaluation average loss: {eval_avg_loss}")

            eval_results = evaluation.overall_performance(val_df, all_predictions, time)

            # Will use matplotlib to draw the graph in the future
            total_earnings += eval_results["earnings"]
            track_all_predictions.append(eval_results)
        
        print(f"Total earnings (sum of daily returns): {total_earnings:.2%}")
        print(track_all_predictions)
        
        deep_learning.save_model(
            model=model,
            model_name='CNN_LSTM',
            mean=mean,
            std=std,
            train_start="2017-01-01",
            train_end="2021-01-01",
            feature_columns=FEAUTRES,
        )

            
            
        
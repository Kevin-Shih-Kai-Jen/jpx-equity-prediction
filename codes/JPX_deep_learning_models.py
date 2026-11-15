import torch
from torch import nn
import pandas as pd
from pathlib import Path

import deep_learning, models, features, evaluation, data_processor, utils, config
from data_loader import main_loader



    
def common_process(
    train_and_val: list[str],
    df: pd.DataFrame
):
    # Get train date and evaluation date
    train_start, train_end, val_start, val_end = train_and_val
    train_df = data_processor.get_time_for_specific_period(df, train_start, train_end)
    val_df = data_processor.get_time_for_specific_period(df, val_start, val_end)
    
    # Calculate mean and std with training data
    train_df, mean, std = features.normalize_data(
        train_df,
        config.FEAUTRES
    )
    
    # Use the same mean and std for validation data
    val_df, _, _ = features.normalize_data(
        val_df,
        config.FEAUTRES,
        std=std,
        mean=mean
    )
    
    device = utils.get_device()
    print(f"Using device: {device}", flush=True)

    print("Stage 4: Normalized", flush=True)
    
    return train_df, val_df, device, mean, std


def more_common_process(
    model,
    loss_fn,
    optimizer,
    epochs: int,
    col_names: list[str],
    model_name: str,
    split_gap: int | None,
    device: str="cpu",
    time_span: int | None=None,
    daily_batch=False
):
    # None keeps model_1's original in-sample evaluation.
    if split_gap is None:
        splits = [["2017-01-01", "2021-07-01", "2017-01-01", "2021-07-01"]]
    else:
        splits = utils.create_start_end_date(
            start_time="2017-01-01",
            end_time="2021-07-01",
            gap=split_gap,
            mode="expanding_window"
        )
    
    df = main_loader(config.DATA_PATH, show_basic_data=False)
    
    df["ExpectedDividend"] = df["ExpectedDividend"].fillna(0)
    print("Stage 1: Loaded", flush=True)

    valid_index = features.get_valid_rows(df, set(col_names)).index
    
    print("Stage 2: Validated", flush=True)

    df = data_processor.correct_prices_with_factors(df)
    
    df = df.loc[valid_index].copy()

    print("Stage 3: Prices adjusted", flush=True)
                
    time_spans = config.TIME_SPAN if time_span is None else [time_span]

    for time in time_spans:
        # Record all validations results
        total_earnings, track_all_predictions = 0, []
        
        for train_and_val in splits:
            train_df, val_df, device, mean, std = common_process(train_and_val, df)
            if split_gap is not None:
                val_df = utils.compensate_val_lost_time(train_df, val_df, time)

                # Remove the last two training dates to avoid target leakage.
                cutoff = train_df["Date"].drop_duplicates().sort_values().iloc[-2]
                train_df = train_df.loc[train_df["Date"] < cutoff]
            
            X_train, y_train, train_dates = features.create_time_sequence(train_df, time_sequence=time, col_names=col_names, return_dates=True)
            if split_gap is not None:
                X_val, y_val, val_dates = features.create_time_sequence(val_df, time_sequence=time, col_names=col_names, return_dates=True)

            print("Stage 5: Created time sequence", flush=True)
            
            train_data_loader = deep_learning.make_loader(
                X_train, 
                y_train,
                dates = train_dates if daily_batch else None
            )
            if split_gap is None:
                # model_1 evaluates on the same samples it trained on.
                val_data_loader = train_data_loader
            else:
                val_data_loader = deep_learning.make_loader(
                    X_val, 
                    y_val,
                    dates = val_dates if daily_batch else None
                )

            print("Stage 6: Data loaded", flush=True)
            
            train_avg_loss = deep_learning.train(model, train_data_loader, loss_fn, optimizer, epochs=epochs, device=device)
            print(f"{model_name} {time} time span training average loss: {train_avg_loss}")
            eval_avg_loss, all_predictions = deep_learning.evaluation(model, val_data_loader, loss_fn, device)
            print(f"{model_name} {time} time span evaluation average loss: {eval_avg_loss}")

            eval_results = evaluation.overall_performance(
                val_df,
                all_predictions,
                time,
                n_long=config.N_LONG,
                n_short=config.N_SHORT,
            )

            if split_gap is None:
                evaluation.zero_baseline(y_train)

            # Will use matplotlib to draw the graph in the future
            total_earnings += eval_results["earnings"]
            track_all_predictions.append(eval_results)
        
        print(f"Total earnings (sum of daily returns): {total_earnings:.2%}")
        print(track_all_predictions)
        
        if split_gap is not None:
            deep_learning.save_model(
                model=model,
                model_name=model_name,
                mean=mean,
                std=std,
                train_start=train_and_val[0],
                train_end=train_and_val[1],
                feature_columns=col_names,
            )


@utils.print_running_time
def model_1():
    device = utils.get_device()

    for time in config.TIME_SPAN:
        model = models.CNN_LSTM(
            num_features=len(config.FEAUTRES),
            cnn_out_channels=64,
            kernel_size=time,
            hidden_size=100
        ).to(device)

        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters())

        more_common_process(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=50,
            col_names=config.FEAUTRES,
            model_name="CNN_LSTM_model_1",
            split_gap=None,
            device=device,
            time_span=time
        )


@utils.print_running_time
def model_2():
    device = utils.get_device()

    for time in config.TIME_SPAN:
        model = models.CNN_LSTM(
            num_features=len(config.FEAUTRES),
            cnn_out_channels=64,
            kernel_size=3,
            hidden_size=100
        ).to(device)

        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters())

        more_common_process(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=50,
            col_names=config.FEAUTRES,
            model_name="CNN_LSTM",
            split_gap=180,
            device=device,
            time_span=time
        )


@utils.print_running_time
def model_3():
    model = models.ENSEMBLE(
        num_features=len(config.FEAUTRES),
        device=utils.get_device()
    )
    
    loss_fn = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    
    more_common_process(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=50,
        col_names=config.FEAUTRES,
        model_name="ENSEMBLE",
        split_gap=90
    )

        

def model_4():
    model = models.MASTER(
        gating_input_dim=len(config.FEAUTRES) * 2,
        gating_output_dim=len(config.FEAUTRES),
        beta=5.0,
        hidden_dim=128,
        intra_nhead=4,
        inter_nhead=2,
        dim_feedforward=256,
        activation_function="relu"
    ).to(utils.get_device())

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-5
    )
    
    more_common_process(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=50,
        col_names=config.FEAUTRES,
        model_name="MASTER",
        split_gap=90,
        daily_batch=True
    )


def model_5():
    model = models.TimeGrad(
        num_features=len(config.FEAUTRES)
    ).to(utils.get_device())
    
    loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-5
    )
    
    more_common_process(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=50,
        col_names=config.FEAUTRES,
        model_name="TimeGrad",
        split_gap=90,
        daily_batch=True
    )
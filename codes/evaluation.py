"""
In order to fit the requirements of JPX, we have to use sharpe, rank IC to check
the model's true performance. 

functions:
1. rank each stocks every day 
2. pick the first 200 and last 200
3. Use the given "Target" to calculate daily sharpe
4. calculate rank IC (Based on "Target")
5. Calculate the overall sharpe and rank IC (Average of given time span)


A better system design is to put the prediction values into the df
--> Can use a new df with columns of [SecuritiesCode, Date, Target, Prediction] 
    to prevent overheads
"""
import pandas as pd
import numpy as np
import torch
from collections.abc import Callable


def get_eval_df(
    df: pd.DataFrame, 
    predictions: np.ndarray,
    time_window: int
) -> pd.DataFrame:    
    df = df.copy().sort_values(["SecuritiesCode", "Date"])
    
    # record the date position to find out from what date we put in eval_df
    # Cuz we need previous windows
    positions = df.groupby("SecuritiesCode").cumcount()
    eval_df = df.loc[
        positions >= time_window - 1, # 0-indexing
        ["SecuritiesCode", "Date", "Target"]
    ]
    
    eval_df["Prediction"] = predictions
    return eval_df


def rank(
    df: pd.DataFrame,
    col_name: str
) -> pd.DataFrame:
    """
    Input:  [SecuritiesCode, Target | Predictions], column name    
    Output: [SecuritiesCode, Rank]
    """
    df[f"{col_name}_rank"] = df.groupby("Date")[col_name].rank(ascending=False)
    return df


def calculate_weights(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    The formula of weights:
    raw_weights = np.linspace(2, 1, 200)
    weights = 0.5 * raw_weights / raw_weights.sum()
    
    200 for long, 200 for short
    """
    if "Prediction_rank" not in df:
        df = rank(df, "Prediction")
    
    # 1. 1 ~ 200     
    df = df.sort_values("Prediction_rank")
    
    if len(df) < 400:
        raise ValueError("At least required 400 stocks")
    
    raw_weights = np.linspace(2, 1, 200)
    weights = 0.5 * raw_weights / raw_weights.sum()
    
    weight_column = np.zeros(len(df))
    weight_column[:200] = weights
    weight_column[-200:] = -weights[::-1]
    
    df["Weight"] = weight_column
    return df
    

def daily_earning(
    df: pd.DataFrame,
) -> np.array:
    """
    Use predicted ranking to calculate sharpe value with real Target
    1e-9 to avoid division error
    """
    return np.array([
        float((daily_df["Weight"] * daily_df["Target"]).sum()) 
        for _, daily_df in df.groupby("Date")
    ], dtype="float32")


def calculate_sharpe(
    df: pd.DataFrame,
    base_number: int | float | np.float32 | np.float64 = 0.0
) -> float:
    """
    Use df's everyday value to calculate each day's earnings sum and the overall std
    """
    if df.empty:
        raise ValueError("df is empty")
    
    daily_earnings = daily_earning(df)
    
    if len(daily_earnings) < 2:
        raise ValueError("Required more than 2 days' earnings")
    
    std = daily_earnings.std(ddof=1)
    if std == 0:
        raise ValueError("Standard Deviation can not be 0")
    
    return float((daily_earnings.mean() - base_number) / std)
        

def calculate_rank_IC(
    df: pd.DataFrame
) -> float:
    """
    Rank IC between Predictions and Targets
    """
    if "Prediction" not in df:
        raise ValueError("Missing required columns in df")
    
    return df.groupby("Date")[["Prediction", "Target"]].apply(
        (lambda day: day["Prediction"].corr(
            day["Target"], method="spearman"
        ))
    ).mean()


def overall_performance(
    df: pd.DataFrame,
    all_predictions: np.ndarray,
    time: int = 1
) -> dict[str: float]:
    """
    Calculate indicies like sharpe, rank IC...... and
    get the Average, std of the indicies in given periods.
    
    results = [0] * len(funcs)
    
    for
    """
    eval_df = get_eval_df(df, all_predictions, time)
    
    if eval_df.empty:
        raise ValueError("Missing data in DataFrame")
    
    weighted_days = []
    
    for _, daily_df in eval_df.groupby("Date"):
        weighted_df = calculate_weights(daily_df)
        weighted_days.append(weighted_df)
    
    eval_df = pd.concat(weighted_days, ignore_index=True)
    
    earnings = daily_earning(eval_df).sum()
    sharpe = calculate_sharpe(eval_df)
    rank_IC = calculate_rank_IC(eval_df)
    print("=============================== Evaluation =============================== ")
    print(f"Sharpe: {sharpe} \nRank IC: {rank_IC}\nEarnings: {earnings}")
    print("========================================================================== ")
    
    return {
        "sharpe": sharpe,
        "rank_IC": rank_IC,
        "earnings": earnings
    }
    
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
    
#========================================= Comparison Models =========================================#
def zero_baseline(target):
    # Compare with all predictions = 0
    y_p = np.asarray(target, dtype="float32")
    baseline_mae = np.mean(np.abs(y_p))
    print(f"Zero baseline mae: {baseline_mae}")
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_processor import check_JPX_stock_price_csv

def get_valid_rows(
        table: pd.DataFrame,
        features_col_name: set[str] | list[str]
    ) -> pd.DataFrame:
    # Make sure it doesn't change the original df
    df = table.copy()
    
    # So that not_to_remove can have correct indexing    
    results, duplicates = check_JPX_stock_price_csv(df.copy())
    if not duplicates.empty:
        raise ValueError("Duplicate SecuritiesCode and Date rows")
    not_to_remove = [True] * len(df)
    
    # dict[str, list[tuple[int, bool, TypeError | ValueError | None]]]
    for col_name, check_result in results.items():
        if col_name in features_col_name:
            # check_result position follows the order of rows -> can use df.iloc[index]
            for position, (_, ok, _) in enumerate(check_result):   
                if not ok:                 
                    not_to_remove[position] = False
    
    return df.iloc[not_to_remove]
        

def get_features(df: pd.DataFrame,
                 features_col_name: list[str]
    ) -> pd.DataFrame:
    
    missing = [
        n for n in features_col_name if n not in df
    ]
    
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    return df[features_col_name].copy()


def normalize_data(
    df: pd.DataFrame,
    col_names: list[str],
    std: np.ndarray | None = None,
    mean: np.ndarray | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    missing = [name for name in col_names if name not in df.columns]

    if missing:
        raise ValueError(f"Columns not in DataFrame: {missing}")
    
    if (std is None) != (mean is None):
        raise ValueError("std and mean must be provided together")

    normalized_df = df.copy()
    
    if std is None:
        scaler = StandardScaler()
        
        normalized_df[col_names] = scaler.fit_transform(df[col_names])
        mean = scaler.mean_
        std = scaler.scale_
    else:
        # std, mean are offered
        normalized_df[col_names] = (df[col_names] - mean) / std
    
    return normalized_df, mean, std
    
        


def create_time_sequence(
    df: pd.DataFrame | pd.Series,
    time_sequence: int = 1,
    col_names: list[str] = []
) -> tuple[list[list[float | int]], list[list[float | int]]]:
    if isinstance(time_sequence, bool) or not isinstance(time_sequence, (int, np.integer)) or time_sequence <= 0:
        raise ValueError("time_sequence must be a positive integer")

    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if not col_names:
        raise ValueError("Missing required col_names")
    
    missing = [n for n in col_names if n not in df]
    if missing:
        raise ValueError(f"No {missing} in df")
        
    
    X, y = [], []
    
    
    df = df.sort_values(["SecuritiesCode", "Date"])
    
    # groupby --> loop each stock
    # store into sequence: X: [[[Day 1 features], [Day 2 features]......], group1, group2]
    # groups -> time span data -> features of a day
    # X.shape: (amount_of_group, days_of_features, features) = (batch, data_in_batch, amount_of_features)
    # y.shape: [groups[[Target]]] = (1, 1, len(df["Target"])) to match X's dimensions
    for _, stock in df.groupby("SecuritiesCode"):
        
        # Converting to numpy can save 20 times more memory space * 2 more times of float64 -> float32
        X_np = stock[col_names].to_numpy(dtype="float32")
        y_np = stock["Target"].to_numpy(dtype="float32")
        
        for i in range(len(stock) - time_sequence + 1):
            X.append(X_np[i:i + time_sequence])
            y.append(y_np[i + time_sequence - 1])
    
    return X, y
        
        

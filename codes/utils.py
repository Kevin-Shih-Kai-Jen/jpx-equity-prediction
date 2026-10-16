import pandas as pd
import time
from datetime import datetime, timedelta


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


def create_start_end_date(
    start_time: str | datetime="2017-01-01",
    end_time: str | datetime="2021-07-01",
    gap: int=360,
    mode: str="expanding_window"
):  
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, "%Y-%m-%d")
    if isinstance(end_time, str):
        end_time = datetime.strptime(end_time, "%Y-%m-%d")
    
    if gap <= 0:
        raise ValueError("gap must be greater than 0")

    if start_time >= end_time:
        raise ValueError("start_time must be earlier than end_time")

    if mode != "expanding_window":
        raise ValueError("Unsupported mode")
            
    ans = []
    
    if mode == "expanding_window":
        train_end = start_time + timedelta(days=gap)
        
        while True:
            # Since the last 2 train will be abandoned later
            # --> no need to worry leaked data problem
            val_start = train_end
            val_end = val_start + timedelta(days=gap)
            
            if val_end > end_time:
                break
                
            ans.append(
                [start_time, train_end, val_start, val_end]
            )
            train_end = val_end
    
    return ans


def compensate_val_lost_time(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    lookback: int
) -> pd.DataFrame:
    """
    If lookback date is 42, then I need to compensate current df by adding 41
    previous data 50 let training start at the first val day
    """
    history = (
        train_df
        .sort_values(["SecuritiesCode", "Date"])
        .groupby("SecuritiesCode")
        .tail(lookback - 1)
    )
    
    val_df = pd.concat(
        [history, val_df],
        ignore_index=True
    )
    
    return val_df
    

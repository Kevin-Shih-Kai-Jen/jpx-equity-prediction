import pandas as pd
import numpy as np
from typing import Any, Callable
from datetime import datetime


def is_valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def find_duplicates(df: pd.DataFrame,
                    key_columns: list[str]
    ) -> pd.DataFrame:
    duplicates = df.duplicated(
        subset=key_columns,
        keep=False
    )
    return df.loc[duplicates]


NUMBER_TYPES = [
    int, float,
    np.int32, np.int64,
    np.float32, np.float64,
]


JPX_STOCK_PRICES_REQUIREMENTS = {
    "SecuritiesCode": (
        [int, np.int32, np.int64],
        [lambda x: x > 0]
    ),
    
    "Date": (
        [str],
        [is_valid_date]
    ),
    
    "Close": (
        NUMBER_TYPES,
        [lambda x: x > 0]
    ),
    
    "Open": (
    NUMBER_TYPES,
    [lambda x: x > 0]
    ),

    "High": (
        NUMBER_TYPES,
        [lambda x: x > 0]
    ),

    "Low": (
        NUMBER_TYPES,
        [lambda x: x > 0]
    ),
    
    "Volume": (
        NUMBER_TYPES,
        [
            lambda x: x >= 0,
            lambda x: x % 1 == 0 # No floating point values allowed but can be float type
        ]
    ),
    
    "Target": (
        NUMBER_TYPES,
        []
    )
}
    

def check_single_value(acceptable_types: list[type],
                       value: Any,
                       rules: list[Callable[[Any], bool]]
    ) -> tuple[bool, TypeError | ValueError | None]:
    
    # Check single value or a whole series
    if not pd.api.types.is_scalar(value):
        return False, TypeError("Expected a single value")
    
    if pd.isna(value):
        return False, ValueError("Value is missing")
    
    if type(value) not in acceptable_types:
        return False, TypeError("Value doesn't fit the acceptable type")
    
    # Add "float" cuz np.number doesn't incorporate float("inf")
    if isinstance(value, (float, np.number)) and not np.isfinite(value):
        return False, ValueError("Value is infinite or incorrect")
    
    if rules:
        for func in rules:
            # doesn't meet the rule
            if not func(value):
                return False, ValueError(f"{value} violates a validation rule")
    
    return True, None

def check_series(acceptable_types: list[type],
                series: pd.Series,
                rules: list[Callable[[Any], bool]] = None
    ) -> list[tuple[int, bool, TypeError | ValueError | None]]:
    # For checking whether the series is valid or not
    # return [(index, ok, errortype)]
    results = []
    
    for i, value in series.items():
        ok, ErrorType = check_single_value(acceptable_types, value, rules)
        results.append((i, ok, ErrorType))
    
    return results

def check_table(table: pd.DataFrame,
                series_requirements: dict[
                    str,
                    tuple[list[type], list[Callable[[Any], bool]]]
                ]
    ) -> dict[str, list[tuple[int, bool, TypeError | ValueError | None]]]:
    # Check the whole table
    if not isinstance(table, pd.DataFrame):
        raise TypeError("Expected a DataFrame")
    
    if table.empty:
        raise ValueError("Table is empty")
    
    
    results = {}
    for col_name, (acceptable_types, rules) in series_requirements.items():
        if col_name not in table.columns:
            raise ValueError(f"Missing required column: {col_name}")
        
        series_results = check_series(
            acceptable_types, 
            table[col_name], 
            rules
        )
        results[col_name] = series_results
    return results


    
def check_JPX_stock_price_csv(
        df: pd.DataFrame
    ) -> tuple[
        dict[str, list[tuple[int, bool, TypeError | ValueError | None]]], 
        pd.DataFrame
    ]:
    results = check_table(df, JPX_STOCK_PRICES_REQUIREMENTS)
    duplicated_table = find_duplicates(df, ["Date", "SecuritiesCode"])

    return results, duplicated_table
        

def correct_prices_with_factors(df: pd.DataFrame):
    """
    Adjust Open, High, Low, Close, Volume with AdjustmentFactor, ExpectedDividend
    """
    table = df.copy()
    #========================================= Fix wrong values =========================================#
    AdjustmentFactor_checked_results = check_series([int, float, np.float32, np.float64], table["AdjustmentFactor"])
    
    for i, ok, _ in AdjustmentFactor_checked_results:
        if not ok:
            table.at[i, "AdjustmentFactor"] = 1
    
    if (table["AdjustmentFactor"] <= 0).any():
        raise ValueError("AdjustmentFactor must be greater than 0")

    #===========================================================================================================#
    # sort values by SecuritiesCode, Date (Change str into Datetime)
    # cumprod values --> times adjustment factors (At) everyday(At = New_val * At-1)
    # Price =/ At, Volume *= At (ex. At = 0.5 --> price from 100 to 50, volume 20 to 40)
    table["Date"] = pd.to_datetime(table["Date"])
    table = table.sort_values(["SecuritiesCode", "Date"])
    
    prior_factor = table.groupby("SecuritiesCode")["AdjustmentFactor"].transform(
        lambda s: s.cumprod().shift(1, fill_value=1.0)
    )
    
    if not np.isfinite(prior_factor).all() or (prior_factor <= 0).any():
        raise ValueError("Cumulative AdjustmentFactor must be finite and greater than 0")
    
    for col_name in ["Open", "High", "Low", "Close"]:
        table[col_name] = table[col_name]/prior_factor
    table["Volume"] = table["Volume"] * prior_factor
    
    return table
        

def get_time_for_specific_period(
    df: pd.DataFrame,
    start_time: str,
    end_time: str
) -> pd.DataFrame:
    return df.loc[
        (df["Date"] >= pd.to_datetime(start_time)) &
        (df["Date"] < pd.to_datetime(end_time))
    ]
    
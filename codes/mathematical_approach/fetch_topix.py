"""
Fetch TOPIX data

"""
import pandas as pd

from AI_written import topix_validator
import data_loader, data_processor, config

def run():
    return calculate_daily_changes(preprocessor())

def calculate_daily_changes(
    topix: pd.DataFrame    
) -> pd.DataFrame:
    """
    Calculate T + 2 / T + 1 to align with JPX Target
    """
    changes = topix["Close"] / topix["Close"].shift(1) - 1
    
    # -2: Just like JPX target that requires T + 2 data, and current time is T
    topix["TopixReturn"] = changes.shift(-2)
    
    return topix["TopixReturn"]
    
def preprocessor(
    data_path: str = config.TOPIX_PATH
):
    """
    Get TOPIX data, clean
    """
    topix = data_loader.main_loader(data_path)
    result = topix_validator.validate_topix(
        topix, 
        check_table_func=data_processor.check_table, 
        end="2022-12-31"
    )
    
    if not result:
        raise ValueError("TOPIX data is flawed")
    
    topix["Date"] = pd.to_datetime(topix["Date"])

    return topix[["Date", "Close"]].set_index('Date').sort_index()




    
    
    
    
from pathlib import Path

COL_NAMES = ["SecuritiesCode", "Date", "Open", "High", "Low", "Close", "Volume", "AdjustmentFactor", "Target"]
FEAUTRES = ["Open", "High", "Low", "Close", "Volume", "ExpectedDividend"]
TIME_SPAN = [21]
# Portfolio selection: long/short capital shares follow the stock-count ratio.
# Example: N_LONG = 7, N_SHORT = 3 -> 70% long, 30% short.
N_LONG = 3
N_SHORT = 3
KAGGLE_DATA_PATH = Path("/kaggle/input/datasets/aiaila101/stock-price/stock_prices.csv")
DATA_PATH = (
    KAGGLE_DATA_PATH
    if KAGGLE_DATA_PATH.is_file()
    else Path("/Users/coolguy/Desktop/JPX/data/stock_prices.csv")
)
TOPIX_PATH = Path("/Users/coolguy/Desktop/JPX/data/TOPIX_daily.csv")
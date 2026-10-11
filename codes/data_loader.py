from pathlib import Path
import pandas as pd

def load_data(file_path):
    # Change to full path
    path = Path(file_path).expanduser()
    
    # Make sure there is a file
    if not path.is_file():
        raise FileNotFoundError(f"Could not find file in {path}")
    
    # read CSV and transform it into DataFrame
    df = pd.read_csv(path)
    
    # Check if there's values in the table
    if df.empty:
        raise ValueError("No values in the DataFrame")
    
    # return df
    return df

def print_loaded_data(df):
    print(df.head())
    print(df.shape)
    print(df.columns.tolist())
    print(df.dtypes)


def main_loader(file_path, show_basic_data = True):
    df = load_data(file_path=file_path)
    if show_basic_data:
        print_loaded_data(df)
    
    return df

if __name__ == "__main__":
    df = main_loader("../data/stock_prices.csv")
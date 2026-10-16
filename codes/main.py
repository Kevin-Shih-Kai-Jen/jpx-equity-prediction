import argparse
from pathlib import Path

from models import CNN_LSTM
from JPX_deep_learning_models import model_2, FEAUTRES, DEFAULT_DATA_PATH
from deep_learning import load_model

def main():
    parser = argparse.ArgumentParser(description="Run the JPX CNN-LSTM backtest.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to stock_prices.csv (default: the repository's data directory).",
    )
    args = parser.parse_args()
    model_2(data_path=args.data)

def use_loaded_model(model_path: str | Path):
    model = CNN_LSTM(
        num_features=len(FEAUTRES),
        cnn_out_channels=64,
        kernel_size=3,
        hidden_size=100
    )
    return load_model(
        initial_model=model,
        model_path=model_path,
    )
    


if __name__ == "__main__":
    main()

from pathlib import Path

from models import CNN_LSTM
from JPX_deep_learning_models import model_5
from deep_learning import load_model
from config import COL_NAMES, FEAUTRES

def main():
    model_5()

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
    
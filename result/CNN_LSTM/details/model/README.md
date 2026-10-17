# Final CNN-LSTM checkpoint

`CNN_LSTM_final.pt` is the original final inference checkpoint from the completed 8-fold, 20-epoch-per-fold run. It was copied byte for byte; only its filename changed for publication.

| Property | Value |
|---|---|
| Frozen training source | `b3cd1bfd617251c7eebc42c588e4067a09af1fe3` |
| Original filename | `CNN_LSTM_20261017_053542_185236_cbe77cda.pt` |
| Size | 924,681 bytes |
| SHA-256 | `b41c8fbea9ae426c782a4e1105e7fab6f4ddcc52d93fa65ede16dabcadb3e15f` |
| Training | 8 folds x 20 epochs; batch 64; lookback 42; seed 20261016 |
| State across folds | Model and Adam retained throughout training |
| Runtime used | Python 3.12.14; PyTorch 2.12.0+cpu |
| Final training input dates | 2017-01-04 through 2020-12-08, inclusive after target purging |
| Final scaler-fitting dates | 2017-01-04 through 2020-12-10, inclusive |

The checkpoint includes `model_state_dict`, `mean`, `std`, `feature_columns`, `train_start`, `train_end`, and `metadata`. Its architecture and preprocessing configuration are also recorded in `../experiment_config.json`.

## Load from the repository root

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("codes").resolve()))
from models import CNN_LSTM
from deep_learning import load_model

initial_model = CNN_LSTM(
    num_features=6,
    cnn_out_channels=64,
    kernel_size=3,
    hidden_size=100,
)

model, mean, std, feature_columns, train_start, train_end = load_model(
    initial_model,
    Path("result/CNN_LSTM/details/model/CNN_LSTM_final.pt"),
    map_location="cpu",
    weights_only=True,
    device="cpu",
)
```

`load_model` restores the weights and sets evaluation mode. For inference, reproduce the frozen source's row validation, price adjustments, ExpectedDividend handling and per-security 42-row sequences. Normalize the six input columns in their recorded order with the saved `mean` and `std`. Do not fit a new scaler on the inference period.

This is the final model after the eighth fold, not a model for each historical validation date. Evaluating it across all earlier folds would use information learned later and would not reproduce the reported out-of-sample path. Reproducing those results requires the original per-fold saved predictions or rerunning the full sequential experiment.

The inference checkpoint does not include Adam or random-number-generator states. Resumable training snapshots are separate from this published model.

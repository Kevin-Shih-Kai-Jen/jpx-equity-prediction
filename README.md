# JPX Equity Prediction

A personal quantitative-finance learning project using a **PyTorch CNN-LSTM** to predict JPX stock targets. The current implementation includes corporate-action adjustment, training-period normalization, per-security rolling windows, expanding-window training and validation, and daily ranking diagnostics.

**Updated 2026-10-16.** The repository now includes a configurable data path, per-epoch training progress, and a final checkpoint with experiment metadata. These changes were checked with small tests; they do not establish predictive performance or reproduce a full-data backtest. Earlier snapshots and reports remain available in Git history and the repository.

## Completed CNN-LSTM results

**Published 2026-10-17.** The frozen source commit `b3cd1bfd617251c7eebc42c588e4067a09af1fe3` completed all 8 folds, 20 epochs per fold, batch size 64 and lookback 42. Model and Adam state continued across folds.

Read the [English results PDF](result/CNN_LSTM/CNN_LSTM_Results_EN.pdf) for the original metrics and their limitations. Supporting data and reproducibility notes are in [details](result/CNN_LSTM/details/); the [final model checkpoint and loading guide](result/CNN_LSTM/details/model/) are included. These are historical backtest results, not a submitted Kaggle leaderboard score.

## Explore the project

| Component | Purpose |
| --- | --- |
| [Entry point](codes/main.py) | Runs `model_2()` with an optional CSV path; includes a checkpoint-loading helper |
| [Experiment pipeline](codes/JPX_deep_learning_models.py) | Data preparation, expanding windows, training, validation, and saving |
| [CNN-LSTM model](codes/models.py) | Convolution, pooling, three-layer LSTM, and regression head |
| [Training utilities](codes/deep_learning.py) | Tensor conversion, loaders, training, inference, and checkpoint I/O |
| [Data loader](codes/data_loader.py) and [processor](codes/data_processor.py) | CSV loading, corporate-action adjustment, and date selection |
| [Features](codes/features.py) | Row validation, normalization, and per-security sequences |
| [Evaluation](codes/evaluation.py) | Daily ranking, spread returns, Sharpe, and Rank IC |
| [Utilities](codes/utils.py) | Split dates, validation history, and runtime reporting |
| [Latest reflection: 2026-10-16](Reflections/CNN-LSTM%20model/CNN_LSTM_Daily_Update_2026_10_16.pdf) | Code corrections, runtime investigation, and next steps |
| [Reflection: 2026-10-13](Reflections/CNN-LSTM%20model/CNN_LSTM_Daily_Update_2026-10-13.pdf) | Earlier daily update |
| [Reflection: 2026-10-11](Reflections/CNN-LSTM%20model/CNN_LSTM_Daily_Update_2026-10-11.pdf) | Earlier daily update |

## Current default pipeline

`main.py` calls `model_2()`, which uses:

- A 42-row lookback and six inputs: Open, High, Low, Close, Volume, and ExpectedDividend.
- Expanding training periods and validation blocks defined with `gap=180` calendar days.
- Normalization statistics calculated from the training period and reused for validation.
- Up to 41 preceding rows per security appended to validation as input history.
- Removal of the final two distinct training dates from fitting to address forward-looking target overlap.
- CNN kernel size 3, 64 output channels, and LSTM hidden size 100.
- L1 loss, Adam, batch size 64, and 20 epochs per split.
- Per-epoch progress, including the average training loss.
- Apple MPS when available, otherwise CPU. CUDA is not selected by the current source.

The model and optimizer are initialized before the split loop and retained across splits. This is continued training on expanding data, rather than an independent model fitted from scratch for each split. The older `model_1()` remains in the source as a separate in-sample experiment.

The default date range produces eight complete validation blocks. Validation ends on June 8, 2021; the final partial block before July 1 is not included. The lookback counts valid rows per security, so missing observations can make a window span more than 42 trading days.

Only after all eight splits finish successfully, the final model is saved once as a uniquely named `.pt` file in the repository's `checkpoints/` directory. The file includes normalization statistics, feature columns, model configuration, and dates taken from the actual training and validation periods. `train_start` and `train_end` are inclusive training-input dates after target purging; metadata separately records the scaler-fitting range and validation boundaries. Repeated completed runs produce separate files. There are no intermediate checkpoints or automatic resume support.

## Run locally

1. Obtain `stock_prices.csv` from the [JPX competition data page](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data). See [data setup](data/README.md).
2. Install the dependencies in [requirements.txt](requirements.txt). Those pins record the initial publication environment; a fresh installation has not been verified by this update.
3. Place the file at `data/stock_prices.csv` in this repository, or use the `--data` option to point to another local file.
4. From the repository root, run:

   ```bash
   python codes/main.py
   ```

   To use a CSV stored elsewhere:

   ```bash
   python codes/main.py --data /path/to/stock_prices.csv
   ```

The default CSV path and checkpoint directory are resolved from the repository location. Full-data execution materializes rolling windows in memory and can take several hours. It does not save individual prediction rows. The final checkpoint supports later inference through `use_loaded_model()`; it is not a resumable training snapshot.

## Verification and history

Run the current small regression checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

The checks in [tests/test_pipeline.py](tests/test_pipeline.py) use small inputs and temporary files. They do not require the competition CSV or rerun the full training experiment.

The [2026-10-16 sync record](docs/SYNC_2026-10-16.md) and [file manifest](docs/source_manifest_2026-10-16.json) document the earlier, unmodified source upload. Their byte comparisons and syntax checks apply to [that snapshot](https://github.com/Kevin-Shih-Kai-Jen/jpx-equity-prediction/tree/5c6657ef1d39d604f2a2de23a2ca99e6d87b2944), before the current code changes. No full backtest was rerun for that 2026-10-16 source-sync record; the separately completed experiment is linked above.

The [initial publication record](docs/PUBLICATION.md), [original manifest](docs/source_manifest.json), [synthetic test output](docs/test_output.txt), and [historical debugging notes](https://github.com/Kevin-Shih-Kai-Jen/jpx-equity-prediction/blob/5c6657ef1d39d604f2a2de23a2ca99e6d87b2944/codes/JPX_debug_notes.md) describe the **2026-10-11 snapshot**. The obsolete test and debugging-note files have been removed from `codes/`; the [original test file](https://github.com/Kevin-Shih-Kai-Jen/jpx-equity-prediction/blob/5c6657ef1d39d604f2a2de23a2ca99e6d87b2944/codes/test_jpx_debug.py) remains available in Git history. Its recorded 27 passing tests must not be interpreted as a result for the current code.

The three dated reflection PDFs linked above are retained. Duplicate copies were removed after confirming matching SHA-256 hashes. The [previous October 13 upload record and generation files](Reflections/CNN-LSTM%20model/2026-10-13/) are preserved. Reflection documents are research notes, not newly reproduced performance evidence.

Raw competition CSVs, the entire `checkpoints/` directory, Python caches, and macOS metadata are excluded from Git. The single published final checkpoint under `result/CNN_LSTM/details/model/` is an explicit exception to the model-file ignore rule. The original desktop source and data remain unchanged.

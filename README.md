# JPX Equity Prediction

A personal quantitative-finance learning project using a **PyTorch CNN-LSTM** to predict JPX stock targets. The current implementation includes corporate-action adjustment, training-period normalization, per-security rolling windows, expanding-window training and validation, and daily ranking diagnostics.

**Latest source sync: 2026-10-16.** This update copies the current desktop source and reflection PDFs without changing their contents. It does not publish a completed full backtest or establish predictive performance. Earlier snapshots and reports remain available in Git history and the repository.

## Explore the project

| Component | Purpose |
| --- | --- |
| [Entry point](codes/main.py) | Runs `model_2()`; includes a checkpoint-loading helper |
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
- Apple MPS when available, otherwise CPU. CUDA is not selected by the current source.

The model and optimizer are initialized before the split loop and retained across splits. This is continued training on expanding data, not an independent model fitted from scratch for each split. The older `model_1()` remains in the source as a separate in-sample experiment.

## Run locally

1. Obtain `stock_prices.csv` from the [JPX competition data page](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data). See [data setup](data/README.md).
2. Install the dependencies in [requirements.txt](requirements.txt). Those pins record the initial publication environment; this sync does not verify a fresh installation.
3. Change the machine-specific CSV path in `common_process()` in [the experiment pipeline](codes/JPX_deep_learning_models.py) to your local path.
4. From the repository root, run:

   ```bash
   python codes/main.py
   ```

Full-data execution materializes rolling windows in memory and can take several hours. Model saving writes to `checkpoints/CNN_LSTM` relative to the working directory. The source is preserved as supplied, including its checkpoint metadata and local path; these should be reviewed before reusing a checkpoint in another experiment.

## Verification and history

The [2026-10-16 sync record](docs/SYNC_2026-10-16.md) and [file manifest](docs/source_manifest_2026-10-16.json) document the current upload. All nine supplied Python files passed syntax parsing, and all twelve copied files were checked against their source bytes. No full backtest was rerun for this upload.

The [initial publication record](docs/PUBLICATION.md), [original manifest](docs/source_manifest.json), [synthetic test output](docs/test_output.txt), and [debugging notes](codes/JPX_debug_notes.md) describe the **2026-10-11 snapshot**. The historical test file is retained, but its old API assumptions have not been updated for the current module structure. Its recorded 27 passing tests must not be interpreted as a test result for this sync.

Earlier [experiment reflections](Reflections/CNN-LSTM%20model/cnn_lstm_jpx_10_epoch_experiment_report_en.pdf) and the [previous October 13 upload record](Reflections/CNN-LSTM%20model/2026-10-13/) are preserved. Reflection documents are research notes, not newly reproduced performance evidence.

Raw competition CSVs, Python caches, and macOS metadata are excluded under the repository's existing policy. The original desktop files are retained unchanged.

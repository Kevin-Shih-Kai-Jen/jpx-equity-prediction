# JPX Equity Prediction

A personal quantitative-finance learning project that builds a **PyTorch CNN-LSTM regression pipeline** for JPX stock-price data. It covers data validation, price and volume adjustment, feature scaling, per-security rolling windows, model training, and diagnostic evaluation against a zero-prediction baseline.

This repository presents an implementation in progress for an internship portfolio. Current evaluation is **in-sample**; it does not establish out-of-sample predictive ability or trading performance.

## Explore the project

| Component | What it demonstrates |
| --- | --- |
| [Data loader](codes/data_loader.py) | CSV loading and basic data inspection |
| [Data processor](codes/data_processor.py) | Required-field validation and corporate-action adjustment logic |
| [Features](codes/features.py) | Standardization and rolling windows grouped by security |
| [Deep learning](codes/deep_learning.py) | CNN-LSTM architecture, tensor conversion, training, and evaluation |
| [Experiment entry point](codes/JPX_deep_learning_models.py) | Pipeline configuration and prediction diagnostics |
| [Synthetic tests](codes/test_jpx_debug.py) | Checks for shapes, stock boundaries, data validation, and loss accounting |
| [Experiment reflection (PDF)](Reflections/CNN-LSTM%20model/cnn_lstm_jpx_10_epoch_experiment_report_en.pdf) | Historical results, interpretation, and next questions |
| [Debugging notes](codes/JPX_debug_notes.md) | Historical debugging exercises; some described issues have since been fixed |

## Current pipeline

```text
JPX stock_prices.csv
    -> validate rows and adjust prices / volume
    -> standardize OHLCV + ExpectedDividend
    -> 42-row windows within each security
    -> Conv1d -> ReLU -> conditional pooling -> dropout
    -> three-layer LSTM -> dropout -> linear prediction
    -> L1 loss with Adam; MAE and pooled Pearson diagnostics
```

Each window is paired with the `Target` value on its final row. Missing `ExpectedDividend` values are filled with zero. Execution selects Apple MPS when available and otherwise CPU.

## Installation and checks

Use Python **3.11 or newer** for the pinned dependencies. The published checks ran on Python 3.14.2 on macOS; other platforms have not been verified.

```bash
git clone https://github.com/Kevin-Shih-Kai-Jen/jpx-equity-prediction.git
cd jpx-equity-prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s codes -p test_jpx_debug.py -v
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

**Publication check, 2026-10-11: all 27 existing synthetic tests passed.** No JPX CSV is needed for these tests. See the [test output](docs/test_output.txt) and [publication notes](docs/PUBLICATION.md) for the verification scope.

## Run the experiment

1. Obtain `stock_prices.csv` from the [JPX competition data page](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data), subject to the provider's access requirements and terms. See [data setup](data/README.md).
2. In `model_1()` in [JPX_deep_learning_models.py](codes/JPX_deep_learning_models.py), change the argument to `main_loader(...)` to the path of your local CSV. The original source snapshot still contains the author's machine-specific path.
3. From the repository root, run:

   ```bash
   python codes/main.py
   ```

This launches training. The implementation materializes all rolling windows in memory; full-data execution can require substantial memory. Full training was not rerun for this publication.

## What the current snapshot does and does not verify

- **Evaluation:** training and evaluation use the same loader, and the scaler is fit on the full input. A chronological split and training-only scaler fit remain future work.
- **Epoch count:** the caller passes `epoch=10`, but `train()` currently makes one pass through the loader and does not use this argument. The historical PDF's 10-epoch description is not reproduced by this source snapshot.
- **Temporal architecture:** the current convolution kernel and input window both have length 42. Convolution therefore leaves one time step for the LSTM, and pooling is skipped. This differs from reference settings discussed in the PDF.
- **Diagnostics:** the code calculates MAE and pooled Pearson correlation. These are not a daily ranking evaluation, Sharpe ratio, or a trading backtest.
- **Data semantics:** synthetic tests do not validate real corporate-action timing, trading-calendar continuity, or every data-quality decision.
- **Reproducibility:** no trained weights, fixed-seed experiment record, or raw training logs were included in the source folder. The PDF is preserved as a historical reflection rather than a newly verified run.

## Snapshot and provenance

The initial publication preserves all seven Python files, the debugging notes, and the PDF **byte for byte** from the supplied working folder. Added files provide repository documentation, dependency versions, ignore rules, and verification records. No model implementation was rewritten as part of the upload.

The [source manifest](docs/source_manifest.json) records file sizes and SHA-256 hashes. Raw competition CSVs, Python caches, and macOS metadata are excluded. External data remains subject to its provider's terms; this initial snapshot does not add a software license.

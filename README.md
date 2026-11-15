# JPX Equity Prediction

A personal quantitative-finance research project using PyTorch to predict JPX stock targets and evaluate daily long/short rankings.

## Research results and reflections

Start with the **[results and research index](result/README.md)**. It brings the research questions, hypotheses, reflections, experiment reports, figures, and supporting evidence together in one place.

The latest report is **[MASTER: learning, results and performance reflection](result/MASTER/MASTER_Research_Reflection_20261115.pdf)**. Earlier TimeGrad, Ensemble, CNN-LSTM, CNN-Ensemble, and modified PRISM research is organized in the same index. Reports describe their recorded experiments; the current source is a later snapshot and does not automatically reproduce those historical results.

## Current source - 2026-11-15

The ten Python modules in `codes/` are synchronized with the author's latest supplied files from 2026-11-15. Their timestamped upload names are restored to the module names used by Python imports, including the previously missing `config.py`.

| File | Purpose |
| --- | --- |
| [main.py](codes/main.py) | Calls `model_5()`, the TimeGrad experiment; also contains a CNN-LSTM checkpoint-loading helper |
| [config.py](codes/config.py) | Data paths, input columns, lookback, and long/short stock counts |
| [JPX_deep_learning_models.py](codes/JPX_deep_learning_models.py) | Shared preprocessing, expanding-window evaluation, and experiment entry points |
| [models.py](codes/models.py) | CNN-LSTM, Ensemble, MASTER, and TimeGrad implementations |
| [deep_learning.py](codes/deep_learning.py) | Tensor conversion, regular and daily loaders, training, inference, and model saving/loading |
| [data_loader.py](codes/data_loader.py) and [data_processor.py](codes/data_processor.py) | CSV loading, corporate-action adjustment, and date selection |
| [features.py](codes/features.py) | Row validation, training-period normalization, and per-security sequences |
| [evaluation.py](codes/evaluation.py) | Ranking, portfolio weights, daily returns, Sharpe, and Rank IC |
| [utils.py](codes/utils.py) | Date splits, validation history, device selection, and runtime reporting |

### Default experiment

- TimeGrad via `model_5()`, with a 21-observation lookback and six inputs: Open, High, Low, Close, Volume, and ExpectedDividend.
- Expanding windows over 2017-01-01 to 2021-07-01, with 90-calendar-day validation blocks.
- Training-period normalization reused for validation; earlier observations provide validation input history, and the final two distinct training dates are excluded from fitting.
- Daily batches, MSE loss, Adam with learning rate `1e-5`, up to 50 epochs, and training-loss early stopping with patience 10.
- Three long and three short stocks (`N_LONG = N_SHORT = 3`); capital shares follow the stock-count ratio.
- CUDA when available, otherwise Apple MPS, otherwise CPU.

The model and optimizer continue across folds. After the split loop, the source saves the final TimeGrad model to `checkpoints/TimeGrad`, relative to the working directory. Repeating a run uses the same checkpoint filename.

## Run

1. Obtain `stock_prices.csv` from the [JPX competition data page](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data).
2. Install the dependencies in [requirements.txt](requirements.txt). Its version pins describe the earlier publication environment; a fresh installation was not tested for this source update.
3. Check `DATA_PATH` in [codes/config.py](codes/config.py). The supplied source first checks its Kaggle dataset path, then falls back to the author's local desktop path. Set this value to your CSV location when using a different environment. See [data setup](data/README.md).
4. From the repository root, run:

   ```bash
   python codes/main.py
   ```

The supplied entry point does not implement a `--data` argument. Full-data training can take substantial time and memory. Raw CSVs and generated checkpoints are excluded from Git; the published historical CNN-LSTM checkpoint is retained under `result/`.

## Verification and history

This source sync passed Python syntax checks and imports for all ten modules. No full training run or backtest was performed for this update.

Research notes formerly stored separately are now in `result/`, with duplicate index descriptions consolidated. Distinct historical PDFs and their supporting files are preserved. Earlier source versions, removed tests, and publication records remain available in Git history.

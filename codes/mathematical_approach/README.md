# OLS residual-momentum baseline

This folder preserves the author's traditional ordinary least squares (OLS) experiment, uploaded on 2026-11-23. The author's assessment is that this simple version has little predictive power. It is retained as a research baseline for comparison with the deep-learning experiments. No quantified performance claim or new full-data backtest is included in this upload.

## Files

- `fetch_topix.py`: loads the configured TOPIX CSV, validates it, and aligns index returns with the JPX target horizon.
- `pipeline.py`: estimates per-stock OLS parameters, ranks stocks by average residuals, and evaluates daily long/short portfolios.
- `__init__.py`: package marker.
- [JPX research 02: from stock-selection signals to optimal holdings](JPX研究02：從選股到最佳持倉.png): the supplied research diagram. Its signal calibration, HMM risk model, transaction costs, and portfolio optimization describe a broader research design; those components are not implemented by this OLS pipeline.

The Python files and diagram are copied unchanged from the supplied folder. The required [TOPIX validator](../AI_written/topix_validator.py) and `TOPIX_PATH` in [config.py](../config.py) are also included from the author's local source.

## Implemented method

For each scoring date `T`, the code selects the most recent 90 TOPIX observations with index dates before `T`:

1. Estimate a stock's intercept and market beta from the first 70 observations (`B = 70`): `stock_target = alpha + beta * topix_return + residual`.
2. Apply those coefficients to the last 20 observations (`G = 20`).
3. Use `mean(stock_target - (alpha + beta * topix_return))` over that 20-observation window as the stock's `Prediction` score.
4. Rank stocks by this residual score and evaluate against the target at `T`.

Thus, `Prediction` is a residual-momentum ranking score, not a calibrated forecast of the next return. The defaults select 200 long and 200 short stocks using the repository's shared weighting functions. Printed summaries include mean daily Rank IC, the additive sum of portfolio returns, and the unannualized daily mean/std Sharpe ratio.

## Run

Use Python 3.12 or newer; the preserved source uses nested quote syntax in f-strings. Install the repository's [dependencies](../../requirements.txt), and set both paths in [codes/config.py](../config.py):

- `DATA_PATH`: JPX CSV containing `Date`, `SecuritiesCode`, and `Target`.
- `TOPIX_PATH`: TOPIX CSV containing `Date`, `Open`, `High`, `Low`, and `Close`; `Volume` is optional and validated when present. The current call validates dates within 2017-01-01 through 2022-12-31.

The supplied paths include the author's local desktop location and must be adjusted for another environment. Raw market data is not included. From the repository root:

```bash
cd codes
python -m mathematical_approach.pipeline
```

The current timing decorator accepts no arguments, so `run()` uses its defaults of 200 long and 200 short stocks. For different portfolio sizes, call `everyday_calculation(target, topix, n_long=..., n_short=...)` after loading data with `pre_processor()` and `fetch_topix.run()`. The existing `codes/main.py` remains the deep-learning entry point.

## Interpretation and verification

The source is preserved as an experimental snapshot. In particular, it uses JPX targets and TOPIX returns shifted to the `t+1` to `t+2` horizon, but selects history using only `index < T`. A label dated `T-1` therefore uses prices at `T+1`, unavailable when scoring at `T`. Label availability needs to be corrected before treating this as a causal out-of-sample backtest. The code also assumes sufficient stock history and one target row per stock on each scoring date.

Upload checks: Python syntax compilation, module imports, and a small synthetic-data pipeline run. These checks establish source integration only; they do not establish predictive performance or reproduce the author's market-data experiment.

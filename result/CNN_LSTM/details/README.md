# CNN-LSTM result details

Read `../CNN_LSTM_Results_EN.pdf` first. All material in this folder supports that English report.

The experiment used commit `b3cd1bfd617251c7eebc42c588e4067a09af1fe3` of `Kevin-Shih-Kai-Jen/jpx-equity-prediction`. All 8 folds completed 20 epochs and validation. This is a historical research backtest, not a Kaggle leaderboard submission.

| File | Purpose |
|---|---|
| `summary.json` | Full-precision results, timing and completion audit |
| `experiment_config.json` | Architecture, optimizer, data hash, versions and exact fold contexts |
| `fold_metrics.csv` | Eight fold results, validation MAE, non-annualized Sharpe and Rank IC |
| `daily_metrics.csv` | 958 daily aggregate spread returns, IC and target-weight changes |
| `monthly_metrics.csv` | Monthly aggregates of the same daily series |
| `cost_sensitivity.csv` | Illustrative cost scenarios, not realized net performance |
| `evaluation_protocol.md` | Formulae, JPX comparison, limitations and metric definitions |
| `artifacts.json` | Original source, prediction and final-checkpoint hashes |
| `model/` | Final trained model checkpoint and loading instructions |
| `figures/` | English charts used in the PDF, in PNG and SVG formats |
| `reproduce_analysis.py` | Recompute results from the original data and saved predictions; no training |
| `build_report.py` | Regenerate the PDF and charts from these aggregate files |
| `verify_results.py` | Check aggregate values, counts, units and cost arithmetic |
| `requirements-report.txt` | Dependencies for report regeneration |

## Regenerate the report

From this folder, in an environment with the report dependencies:

```bash
python verify_results.py
python build_report.py
```

The second command writes the PDF one level above `details/` and refreshes the charts. It does not run the model or modify saved model state.

## Recompute the analysis from predictions

This requires the original `stock_prices.csv` and the separate completed-run archive. Use the training runtime versions recorded in `experiment_config.json` for the original modules (including SciPy for Spearman correlation). The report package includes the final inference checkpoint in `model/`; raw data, per-fold predictions and training-resume state are separate.

```bash
python reproduce_analysis.py \
  --repo /path/to/restored-run \
  --data /path/to/stock_prices.csv \
  --results /path/to/restored-run/cloud_run/results_20261016 \
  --output-dir /path/to/new-analysis
```

Source and dataset hashes are checked before analysis. Each fold is cross-checked against its saved MAE, Sharpe, Rank IC and earnings. The output directory must be new. The script reconstructs original evaluation ordering, dates and weights; it skips feature normalization because normalization does not alter Target, security IDs, dates or row order.

## Units

`spread_return` and `earnings` use the project's gross exposure of 1: +0.5 long and -0.5 short. `earnings` is a sum of daily gross spread returns, not a compounded return. One basis point (bp) is 0.0001 in decimal-return units; 100 bp = 1 percentage point (pp). All primary Sharpe results are daily and non-annualized. The annualized field is only a square-root-of-252 reference, not a JPX leaderboard score.

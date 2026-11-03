# Modified PRISM: training and backtest evidence (2026-11-03)

[Read the five-page Chinese report](PRISM_Research_Report_20261103_ZH.pdf).

This is a research report, not a claim of stable alpha or an exact replication of the original paper.

## Results

| Experiment | Folds | Earnings | Sharpe (logged) | Rank IC |
|---|---:|---:|---:|---:|
| Earlier 180-calendar-day configuration (summary only) | 8 | -5.7801% | -0.02707 | +0.000071 |
| Current 220-calendar-day configuration (full log) | 6 | +20.7503% | +0.09051 | -0.000360 |

All six current folds have positive Earnings, but their periods and train/validation splits differ from the earlier experiment. This is not a controlled improvement comparison. Constant-input correlation warnings remain unresolved. No daily predictions were supplied, so drawdown, daily hit rate, costs and significance were not calculated.

## Architecture and provenance

Reference implementation: RevIN -> GRU temporal summary -> cross-stock Spatial Transformer -> VQ (512 x 128) -> Stage 1 prediction head. Stage 2 freezes the code encoder/codebook and combines temporal RoPE attention with codeword-guided top-2 MoE (8 experts) and a final head. The reconstruction decoder was intentionally removed to prioritize predictive representations; benefit has not been established by an ablation.

The researcher selected the research direction and architectural modifications. ChatGPT assisted implementation, debugging, training integration and this report; substantial portions of the integrated script are AI-generated. No claim of fully independent handwritten implementation is made.

**Important:** `details/train_prism_reference.py` is the earlier AI-assisted reference script, NOT the exact Kaggle executable. Its split default is 180 and it lacks the later resume additions visible in the log. Do not run it expecting to reproduce the six-fold result. The executed notebook, configuration, weights and per-stock daily predictions are still required.

## Evidence files

- `details/kaggle_220day_log.txt`: byte-preserved current raw log (`download(1).txt`).
- `details/previous_180day_summary.txt`: byte-preserved user-supplied earlier summary (`result.txt`); not an original log.
- `details/fold_metrics.csv`, `details/summary.json`: extracted current results.
- `details/fold_results.png`: fold-level chart, not an equity curve.
- `details/manifest.json`: SHA-256 and provenance.

Reference evaluator: Earnings is a sum of daily weighted Targets, Sharpe is unannualized, and mean daily Spearman drops undefined values. These definitions have not been verified against the final Kaggle script. Raw competition data and model weights are not included.

Priority checks: constant predictions and tie ordering; expert usage (validation auxiliary loss often 0.08 vs training ~0.02); fixed-period baselines and ablations; multiple seeds; turnover and costs.

# Simplified TimeGrad: architecture, results and reflection

Report date: 2026-11-14. Published to this repository: 2026-11-15.

- [Read the three-page PDF](TimeGrad_Architecture_Results_Reflection.pdf)
- [Download the editable three-slide PowerPoint](TimeGrad_Architecture_Results_Reflection.pptx)

## Contents

| Slide | Focus |
| --- | --- |
| 1 | A 21-observation LSTM encoder and MLP denoiser; one return target per stock and 100 reverse diffusion steps |
| 2 | Results across 17 expanding folds: +17.52% additive return before costs, mean fold Rank IC of 0.00147, and 10 profitable folds |
| 3 | The author's reflection on simplifying the model, current limitations, and experiments on sampling, denoiser capacity, and joint OHLCV forecasts |

The aggregate return is additive, not compounded or annualized. Rank IC is the equal-weight average of fold means. Folds 13 and 15 contribute +27.10 percentage points together; the other 15 folds total -9.58 percentage points. These results do not establish stable predictive value.

## Evidence and interpretation

The documents summarize the saved TimeGrad notebook and `jpx-timegrad-gpu.log`. This publication adds the existing presentation and PDF; it does not rerun training. The source notebook and log are not included in this publication.

The reflection was edited with AI assistance from the author's supplied text. Possible explanations for weak performance remain hypotheses: more epochs, a larger denoiser, or more forecast targets are proposed experiments, not established fixes. Accuracy and runtime advantages over simpler models still require comparison.

[Results and reflections index](../../README.md) · [Repository home](../../../README.md)

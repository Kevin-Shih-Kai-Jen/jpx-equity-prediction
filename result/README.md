# Results and research reflections

This is the combined index for research questions, hypotheses, findings, validation plans, experiment reports, and supporting evidence. Start with the latest reflection, then follow the relevant model family's reports for detail.

## Latest: simplified TimeGrad (2026-11-14)

**[Architecture, results and reflection - PDF](TimeGrad/2026-11-14/TimeGrad_Architecture_Results_Reflection.pdf)** · [Editable PowerPoint](TimeGrad/2026-11-14/TimeGrad_Architecture_Results_Reflection.pptx) · [Report guide](TimeGrad/2026-11-14/README.md)

Slides 1-2 cover the simplified diffusion architecture and 17-fold results. The reported +17.52% additive return before costs accompanies a near-zero mean fold Rank IC (0.00147), with gains concentrated in two folds. Slide 3 contains the author's reflection on forecasting a single return target, the weak and unstable predictive results, implementation choices, and next experiments.

## Cross-model research log

**[JPX Model Research Log](CROSS_MODEL/reflections_1.pdf)** is a one-page record of three experiments: removing ReLU from Ensemble, removing ReLU and pooling from CNN-LSTM, and adding CNN before Ensemble. It preserves the author's hypotheses, findings, proposed validation criteria, and research priorities. The original English PDF is unchanged.

## Modified PRISM (2026-11-03)

[Report, source evidence, and AI-use disclosure](PRISM/2026-11-03/README.md) · [English PDF](PRISM/2026-11-03/PRISM_Research_Report_20261103_EN.pdf) · [Chinese PDF](PRISM/2026-11-03/PRISM_Research_Report_20261103_ZH.pdf)

The six-fold, 220-calendar-day run reports +20.7503% Earnings and near-zero Rank IC, compared with -5.7801% in the prior eight-fold summary. Different test periods prevent a controlled improvement claim. Constant-input warnings remain unresolved. The included AI-assisted source is a reference snapshot; the exact Kaggle executable and daily predictions were not supplied.

## Ensemble

Read these in order:

1. [Original Ensemble evaluation](ENSEMBLE/JPX_ENSEMBLE_Report_EN.pdf)
2. [Full no-ReLU study](ENSEMBLE/JPX_No_ReLU_Full_Study_EN.pdf)
3. [Long/short and tail-ranking follow-up](ENSEMBLE/JPX_ENSEMBLE_Long_Short_Tail_Study_EN.pdf)

## CNN-LSTM

1. [Original CNN-LSTM results](CNN_LSTM/CNN_LSTM_Results_EN.pdf)
2. [42-observation ReLU/pooling ablation](CNN_LSTM/JPX_CNN_LSTM_42Day_No_ReLU_No_Pooling_Report_EN.pdf) - English translation of the supplied Chinese report.

The original baseline used frozen source commit `b3cd1bfd617251c7eebc42c588e4067a09af1fe3`, eight folds, 20 epochs per fold, batch size 64, and lookback 42, with model and Adam state continuing across folds. Its [metrics, figures, and reproduction notes](CNN_LSTM/details/) and [final checkpoint](CNN_LSTM/details/model/) remain available. These historical backtest results are not a submitted Kaggle leaderboard score.

### Earlier CNN-LSTM updates

| Date | Research note |
| --- | --- |
| 2026-10-16 | [Code corrections, runtime investigation, and next steps](CNN_LSTM/CNN_LSTM_Daily_Update_2026_10_16.pdf) |
| 2026-10-13 | [In-sample evaluation and validation plans](CNN_LSTM/CNN_LSTM_Daily_Update_2026-10-13.pdf) |
| 2026-10-11 | [Reference architecture, preliminary results, and diagnostic next steps](CNN_LSTM/CNN_LSTM_Daily_Update_2026-10-11.pdf) |

The October 13 [reported metrics, PDF generator, and historical upload record](CNN_LSTM/2026-10-13/) are retained. Paths in that historical upload record refer to the original artifact commit.

## CNN-Ensemble

[Training and backtest analysis, 2026-10-19](CNN_ENSEMBLE/JPX_CNN_Ensemble_Training_Analysis_20261019_EN.pdf) - English translation of the supplied Chinese report.

## Archive notes

The 2026-10-21 publication added five results reports and one reflection: two Chinese reports were translated into English, and four supplied English PDFs were preserved unchanged. The reports retain their original results and interpretations.

The 2026-11-15 reorganization combines the former reflection index with this results index. PDF file hashes and extracted text were compared; the historical documents are distinct and are retained once at their canonical paths. Reorganizing these files and synchronizing the current source did not rerun any experiment. Earlier publication manifests remain in Git history.

[Back to the repository](../README.md)

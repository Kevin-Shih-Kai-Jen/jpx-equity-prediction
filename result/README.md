# Results reports - supporting reading

**Start with the [reflections](../Reflections/README.md).** The reflections explain the research questions and decisions; these reports provide the detailed results and supporting analysis.

## Ensemble

Read these in order:

1. [Original Ensemble evaluation](ENSEMBLE/JPX_ENSEMBLE_Report_EN.pdf)
2. [Full no-ReLU study](ENSEMBLE/JPX_No_ReLU_Full_Study_EN.pdf)
3. [Long/short and tail-ranking follow-up](ENSEMBLE/JPX_ENSEMBLE_Long_Short_Tail_Study_EN.pdf)

## CNN-LSTM

1. [Original CNN-LSTM results](CNN_LSTM/CNN_LSTM_Results_EN.pdf)
2. [42-observation ReLU/pooling ablation](CNN_LSTM/JPX_CNN_LSTM_42Day_No_ReLU_No_Pooling_Report_EN.pdf) - English translation of the supplied Chinese report.

The original baseline's [metrics, figures, and reproduction notes](CNN_LSTM/details/) and [final checkpoint](CNN_LSTM/details/model/) remain available.

## CNN-Ensemble

[Training and backtest analysis, 2026-10-19](CNN_ENSEMBLE/JPX_CNN_Ensemble_Training_Analysis_20261019_EN.pdf) - English translation of the supplied Chinese report.

## Publication notes

The 2026-10-21 upload adds five results reports and one reflection. The two Chinese reports were translated into English; the four supplied English PDFs were preserved unchanged. This upload organizes existing research and does not add a training run.

The reports and reflections preserve their original content, reported results, and interpretations. See the [publication manifest](../docs/publication_2026-10-21.json) for file provenance and checksums.

[Back to the reflections](../Reflections/README.md) · [Back to the repository](../README.md)

## Modified PRISM (2026-11-03)

[Five-page English report and evidence](PRISM/2026-11-03/README.md): six-fold 220-day result alongside the earlier eight-fold summary, architecture changes, AI assistance, and unresolved evaluation diagnostics. Exact Kaggle executable and daily predictions are not yet supplied.

## Simplified TimeGrad (2026-11-14)

[Three-slide report and reflection](TimeGrad/2026-11-14/README.md) · [PDF](TimeGrad/2026-11-14/TimeGrad_Architecture_Results_Reflection.pdf) · [Editable PowerPoint](TimeGrad/2026-11-14/TimeGrad_Architecture_Results_Reflection.pptx)

Architecture, 17-fold results, and the author's reflection. The reported +17.52% additive return before costs accompanies a near-zero mean fold Rank IC (0.00147); gains concentrate in two folds. Slide 3 explains the implementation choices, limitations, and next experiments.

# Data setup

The experiment entry point reads `stock_prices.csv` from the [JPX Tokyo Stock Exchange Prediction competition](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data).

Download the competition data through your own account as required by the provider and use it under the applicable terms. Place the CSV here or in another local directory, then point the `main_loader(...)` call in `codes/JPX_deep_learning_models.py` to that file.

The current pipeline uses these columns:

```text
SecuritiesCode, Date, Open, High, Low, Close, Volume,
AdjustmentFactor, ExpectedDividend, Target
```

The supplied working folder also contains `financials.csv`, `trades.csv`, `secondary_stock_prices.csv`, `options.csv`, and `stock_list.csv`; the current entry point does not load them. No raw CSV is redistributed in this repository. The original local data files were not changed or removed during publication.

Files in this directory, except this guide, are ignored by Git. Synthetic tests can run without downloading market data.

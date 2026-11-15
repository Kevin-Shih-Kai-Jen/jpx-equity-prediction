# Data setup

The experiment entry point reads `stock_prices.csv` from the [JPX Tokyo Stock Exchange Prediction competition](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data). Download it through your own account as required by the provider and use it under the applicable terms.

The current source selects its CSV through `DATA_PATH` in [codes/config.py](../codes/config.py):

1. `/kaggle/input/datasets/aiaila101/stock-price/stock_prices.csv`, if that file exists.
2. Otherwise, `/Users/coolguy/Desktop/JPX/data/stock_prices.csv`, the author's supplied local path.

For another location, change `DATA_PATH` to your actual CSV path. For example, to use this repository's `data/stock_prices.csv`, set:

```python
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "stock_prices.csv"
```

Then run from the repository root:

```bash
python codes/main.py
```

The supplied entry point does not parse a `--data` argument.

The pipeline uses these columns:

```text
SecuritiesCode, Date, Open, High, Low, Close, Volume,
AdjustmentFactor, ExpectedDividend, Target
```

No raw CSV is redistributed in this repository. Files in this directory, except this guide, are ignored by Git. The current entry point does not load `financials.csv`, `trades.csv`, `secondary_stock_prices.csv`, `options.csv`, or `stock_list.csv`.

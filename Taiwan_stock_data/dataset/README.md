# 台股資料集 stock_price_final.csv（JPX 格式）

`stock_price_final.csv` 有 392 MB，超過 GitHub 單檔 100 MB 的上限，因此壓縮後切成 3 個分割檔存放在這裡。

| 項目 | 內容 |
|---|---|
| 筆數 | 4,461,650 列 |
| 股票數 | 2,081 檔（上市＋上櫃普通股） |
| 期間 | 2017-01-03 ～ 2026-11-23（2,408 個交易日） |
| 欄位 | 與 Kaggle JPX `stock_prices.csv` 相同：`RowId, Date, SecuritiesCode, Open, High, Low, Close, Volume, AdjustmentFactor, ExpectedDividend, SupervisionFlag, Target` |
| 價格來源 | FinMind `TaiwanStockPrice` 未調整原始報價 |
| 產生程式 | 上一層資料夾的管線（`03` → `05`），欄位定義見 [../README.md](../README.md) |

## 還原成 CSV

### 方法一：Python（會自動驗證校驗碼）
```bash
python Taiwan_stock_data/dataset/join_dataset.py
```
完成後會產生 `Taiwan_stock_data/dataset/stock_price_final.csv`。這個檔名已列入 `.gitignore`，不會被誤 commit。

### 方法二：PowerShell
```powershell
cd Taiwan_stock_data\dataset
cmd /c copy /b stock_price_final.zip.001+stock_price_final.zip.002+stock_price_final.zip.003 stock_price_final.zip
Get-FileHash stock_price_final.zip   # 應與 SHA256SUMS.txt 中 stock_price_final.zip 那行相同
Expand-Archive stock_price_final.zip -DestinationPath .
```

### 方法三：7-Zip
直接用 7-Zip 開啟 `stock_price_final.zip.001`，它會自動讀取後續的分割檔。

## 在 JPX 模型中使用
`codes/config.py` 的 `DATA_PATH` 改指向這個檔案即可，例如：

```python
DATA_PATH = Path(__file__).resolve().parents[1] / "Taiwan_stock_data" / "dataset" / "stock_price_final.csv"
```

`SecuritiesCode` 是 4 碼文字（例如 `2330`），讀檔時建議加上 `dtype={"SecuritiesCode": str}`。

## 已知限制（使用前請先看）
* **興櫃時期資料**：約 379 檔股票保留了上市櫃前的興櫃交易紀錄。這段期間 FinMind 的 `Open` 是前一日均價，常落在當日高低價之外，因此已被清成 `NA`，約 9.6 萬列價格為 `NA` 但成交量大於 0；興櫃也沒有 ±10% 漲跌幅限制。訓練前建議剔除各股上市櫃日之前的資料。
* **只調整股票股利**：`AdjustmentFactor` 只反映配股。現金股利、現金減資、現金增資除權都沒有調整，這些日期附近的 `Target` 會反映價格跳空。
* **倖存者偏差**：股票清單取自擷取當下仍上市櫃的公司，期間內已下市的公司不在資料中。
* `ExpectedDividend`、`SupervisionFlag` 全部為 `NA`。
* `check_prices.py` 目前未全部通過，剩餘錯誤主要來自上述興櫃資料。

## 檔案
| 檔案 | 說明 |
|---|---|
| `stock_price_final.zip.001` ～ `.003` | 壓縮檔分割（45 + 45 + 15.6 MB） |
| `SHA256SUMS.txt` | 原始 CSV、合併後 zip、各分割檔的 SHA256 校驗碼 |
| `join_dataset.py` | 合併、驗證、解壓縮的工具 |

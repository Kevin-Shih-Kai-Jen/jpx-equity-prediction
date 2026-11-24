# 台灣股票市場資料集（相容 JPX 競賽規格）

## 1. 資料集概述（Overview）
本資料集提供台灣股票市場（台股）的歷史價量與公司行動資料。資料維度、欄位定義與防穿透邏輯皆對齊 Kaggle「JPX Tokyo Stock Exchange Prediction」競賽的官方格式，可直接作為機器學習模型的訓練基底，不需大幅修改原 JPX 模型架構即可匯入測試。

* **涵蓋市場**：臺灣證券交易所（TWSE）上市普通股、證券櫃檯買賣中心（TPEx）上櫃普通股。
* **排除標的**：ETF、ETN、受益證券、存託憑證（含 91xx TDR）、大盤指數、權證、特別股、興櫃股。只保留代碼為 4 碼數字（1000–9999）的普通股。
* **時間範圍**：2017-01-01 至最新完成擷取的交易日。
* **資料來源**：全部使用 FinMind API。

| 資料 | FinMind dataset | 說明 |
|---|---|---|
| 股票清單 | `TaiwanStockInfo` | 上市櫃代碼、名稱、市場別、產業別 |
| 歷史價量 | `TaiwanStockPrice` | 交易所公布的**未調整原始報價** |
| 除權息 | `TaiwanStockDividend` | 股票股利、現金股利與除權／除息交易日 |

### 為什麼不用 Yahoo Finance
早期版本使用 yfinance。Yahoo 會把台股「配股」當成股票分割，回溯調整歷史價格，而且 Close 與 Open/High/Low 的調整幅度有時不一致（例：1264 德麥 2017-07-03 出現 Close < Low）。用這份價格再乘上 `AdjustmentFactor`，等於重複調整。FinMind 的原始報價在除權日會正常跳空，與 `corporate_actions.csv` 的除權日完全對得上。

## 2. 品質聲明與倖存者偏差（Survivorship Bias）
股票清單取自擷取當下的 `TaiwanStockInfo`，因此本資料集存在倖存者偏差：只涵蓋「擷取當下仍維持上市／上櫃狀態」的企業。期間內已下市、下櫃或因併購而消滅的公司不在清單內，資料並未完整涵蓋歷史全市場。

## 3. 檔案結構（File Descriptions）

### 最終資料
| 檔案 | 說明 |
|---|---|
| `stock_price_final.csv` | **主表**。每日價量、調整因子與目標報酬率，欄位見第 4 節。 |
| `stock_list.csv` | 股票清單，記載納入的公司代碼、名稱、市場別（twse / tpex）與產業別。 |
| `corporate_actions.csv` | 公司行動表，記載歷史除權息紀錄，欄位見第 5 節。 |

### 中間產物與快取
| 檔案／資料夾 | 說明 |
|---|---|
| `stock_price_full.csv` | 全市場原始報價合併檔（尚未加入調整因子與 Target），供 `05_merge_factors.py` 使用。 |
| `data/raw_finmind_prices/stock_<代碼>.csv` | 單檔報價快取。區間內沒有報價的股票會存成只有表頭的檔案，避免重跑時浪費 API 配額。 |
| `data/raw_corporate_actions/actions_<代碼>.csv` | 單檔除權息快取。 |
| `data/price_anomalies.csv` | 下載時發現的 OHLC 邏輯異常列（有異常才會產生）。 |
| `check_issues.csv` | `check_prices.py` 輸出的問題明細。 |

`stock_price_full.csv`、`stock_price_final.csv` 與 `finmind_token.txt` 已列入 `.gitignore`，不會被 commit。

## 4. stock_price_final.csv 欄位定義與處理規則

| 欄位 | 定義與處理規則 |
|---|---|
| `RowId` | 唯一識別碼，格式為 `YYYYMMDD_SecuritiesCode`（例：`20240102_2330`）。 |
| `Date` | 台灣市場實際交易日，格式為 `YYYY-MM-DD`。含 2017–2018 年證交所補班開市的 5 個週六。 |
| `SecuritiesCode` | 股票代碼，保留為文字型態。 |
| `Open` / `High` / `Low` / `Close` | 原始開盤／最高／最低／收盤價（新台幣元），**未復權**。缺值填 `NA`。 |
| `Volume` | 成交股數（非張數），取自 FinMind `Trading_Volume`。 |
| `AdjustmentFactor` | 單次價格調整因子，無事件的日子為 `1.0`。只在「除權日當天」觸發，公式為 `1 / (1 + StockDividend / 10)`。 |
| `ExpectedDividend` | 預期每股現金股利，本資料集統一填 `NA`。理論上應在宣告日至除息日之間填寫，但缺乏可靠的歷史宣告日，為避免未來資料滲透（look-ahead bias），依「不猜測」原則全數留空。 |
| `SupervisionFlag` | 監理狀態（注意股／處置股），本資料集統一填 `NA`。台股處置機制與 JPX 無法直接等價，目前保留空值。 |
| `Target` | 預測目標：明日收盤至後日收盤的調整後報酬率。**僅作標籤，不可作為特徵輸入。** |

### 價格清洗規則（`03_full_market_pipeline.py`）
* **無成交日**：FinMind 的價格欄位為 0 時，OHLC 改為 `NA`，Volume 為 0。這是正常現象，不列入異常。
* **OHLC 邏輯異常**：出現 `High < Low`，或 Open／Close 超出 High–Low 區間時，該列 OHLC 清成 `NA`，避免污染 Target，並記錄到 `data/price_anomalies.csv`。

### AdjustmentFactor 計算（`05_merge_factors.py`）
* 只使用 `StockDividend > 0` 且有 `StockExDate` 的除權事件。同一股票、同一除權日若有多筆，股票股利會先加總。
* 除權日遇到休市（例如 2024-07-24、07-25 颱風假）或該股當天沒有報價時，因子會**順延到該股下一筆報價**（最多 14 天）。否則因子會整筆遺失，例如 3293 鈊象 2024-07-24 配股 10 元。
* 台股面額 10 元，配股 1 元 = 每股配 0.1 股，所以因子為 `1 / (1 + StockDividend / 10)`。
* **現金股利、現金減資、股票分割目前不調整**，因此除息日或減資復牌附近的 Target 會反映這些價格跳動。

### Target 計算（`05_merge_factors.py`）
調整後收盤價以「除權日之後所有因子的累積乘積」回溯調整：

```
AdjClose(d) = Close(d) × Π AdjustmentFactor(k)，k 為 d 之後的每個交易日
Target(t)   = AdjClose(t+2) / AdjClose(t+1) − 1
            = Close(t+2) / (Close(t+1) × AdjustmentFactor(t+2)) − 1
```

* 計算時依 `SecuritiesCode` 分組並依 `Date` 排序，t+1、t+2 指的是**該股票**的下一、下下筆交易資料。
* t+1 或 t+2 的收盤價為 `NA` 或 ≤ 0（例如停牌）時，Target 填 `NA`。
* 每檔股票最後兩個交易日沒有未來價格，Target 為 `NA`。

## 5. corporate_actions.csv 欄位定義

| 欄位 | 說明 |
|---|---|
| `SecuritiesCode` | 股票代碼。 |
| `AnnounceDate` | 宣告日。FinMind 未提供可靠值，統一填 `NA`。 |
| `StockExDate` | 除權交易日（`StockExDividendTradingDate`），無效日期填 `NA`。 |
| `CashExDate` | 除息交易日（`CashExDividendTradingDate`），無效日期填 `NA`。 |
| `CashDividend` | 每股現金股利（元）= 盈餘分配 + 法定盈餘公積。 |
| `StockDividend` | 每股股票股利（元）= 盈餘轉增資 + 法定盈餘公積轉增資。 |
| `Source` | 資料來源，固定為 `FinMind TaiwanStockDividend`。 |

## 6. 開發與重製（Reproducibility）

### 環境
```bash
pip install pandas numpy requests tqdm
```
（`yfinance` 只有舊版雛形 `01`、`02` 需要。）

### FinMind Token
在專案資料夾建立 `finmind_token.txt`，內容只放一行 Token；也可以改設環境變數 `FINMIND_TOKEN`。此檔已列入 `.gitignore`，請勿上傳。

`StocksSymbol.py`、`03`、`04` 都透過共用的 `finmind_auth.py` 讀取 Token：
* 優先讀檔案，找不到才讀環境變數。不論從哪個資料夾執行，都會找程式旁邊的 `finmind_token.txt`。
* 能容忍記事本存的 UTF-8 BOM、PowerShell 產生的 UTF-16 檔、前後引號，以及不小心貼上的 `Bearer ` 前綴。
* 啟動時驗證 Token，無效就直接停止。
* 單獨執行 `python finmind_auth.py`，可以確認 Token 是否有效與本小時剩餘額度。

### 執行順序
請在專案資料夾內執行：

| 步驟 | 指令 | 產出 |
|---|---|---|
| 1 | `python StocksSymbol.py` | `stock_list.csv` |
| 2 | `python 03_full_market_pipeline.py` | `data/raw_finmind_prices/`、`stock_price_full.csv` |
| 3 | `python 04_corporate_actions.py` | `data/raw_corporate_actions/`、`corporate_actions.csv` |
| 4 | `python 05_merge_factors.py` | `stock_price_final.csv`（結束前印出驗證報告） |
| 5 | `python check_prices.py stock_price_final.csv` | 檢查報告、`check_issues.csv` |

步驟 2 與步驟 3 互不相依，順序可以對調。步驟 4 需要等兩者都完成。

### 03_full_market_pipeline.py 參數
```bash
python 03_full_market_pipeline.py                                  # 只下載還沒抓過的股票，然後合併
python 03_full_market_pipeline.py --force                          # 忽略快取，全部重新下載
python 03_full_market_pipeline.py --start 2017-01-01 --end 2026-11-20
```

### 斷點續傳與 API 配額
* `03` 與 `04` 都會先檢查快取，已下載的股票會跳過。中斷後重新執行即可接續，抓取失敗的股票也會在重跑時補抓。
* 程式啟動時會驗證 Token，並印出本小時已使用的次數與上限。
* 碰到 FinMind 每小時上限（HTTP 402／429）時，程式會自動休眠 65 分鐘後繼續，期間請勿關閉視窗。

### 資料檢查（check_prices.py）
`check_prices.py` 用**另一條公式**獨立重算 Target，不呼叫 `05` 的同一個函式，以免程式寫錯時自己驗證自己通過。檢查項目：

1. 欄位、日期格式、週末日期、RowId、重複鍵
2. OHLC 合理性、部分缺值、Volume
3. 是否為原始報價：價格是否落在台股升降單位上，用來偵測調整後價格
4. 漲跌幅是否超過 ±10%（排除除權息日、新上市前 5 日、停牌後復牌）
5. 除權事件與 AdjustmentFactor 是否吻合（含休市順延）、有無重複調整
6. Target 獨立重算
7. 交易日曆與缺口
8. `ExpectedDividend`／`SupervisionFlag` 是否維持 `NA`
9. 股票清單覆蓋率

```bash
python check_prices.py stock_price_final.csv
python check_prices.py data/raw_finmind_prices          # 也可以直接檢查單檔快取資料夾
python check_prices.py stock_price_final.csv --actions corporate_actions.csv --stock-list stock_list.csv --out check_issues.csv
```

有任何 ❌ 時結束碼為 1，可以接在 pipeline 後面，當作「不通過就不能訓練」的關卡。

## 7. 其他程式

| 檔案 | 說明 |
|---|---|
| `finmind_auth.py` | FinMind Token 共用模組（讀取、驗證、組 Authorization header）。 |
| `stock_search.py` | 互動式查詢工具：輸入公司名稱或代碼，從 `stock_list.csv` 模糊搜尋。 |
| `01_jpx_format_builder.py` | **舊版雛形**。用 yfinance 產生 2330 單檔 JPX 格式（`stock_price_2330.csv`），不屬於正式管線。 |
| `02_jpx_integrated_builder.py` | **舊版雛形**。用 yfinance 加上手動模擬的 2330 除權息資料，示範 ExpectedDividend／AdjustmentFactor 的合併邏輯（`stock_price_integrated_2330.csv`），不屬於正式管線。 |

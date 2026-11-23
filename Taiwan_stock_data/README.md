# 台灣股票市場資料集 (相容 JPX 競賽規格)

## 1. 資料集概述 (Overview)
本資料集旨在提供台灣股票市場（台股）的歷史價量與公司行動資料，並嚴格將資料維度、欄位定義與防穿透邏輯對齊 Kaggle 上「JPX Tokyo Stock Exchange Prediction」競賽的官方格式。此資料集可作為機器學習模型的訓練基底，無須大幅修改原 JPX 模型架構即可直接匯入測試。

* 涵蓋市場：台灣證券交易所 (TWSE) 上市普通股、證券櫃檯買賣中心 (TPEx) 上櫃普通股。
* 排除標的：ETF、權證、興櫃股、特別股及 TDR。
* 時間範圍：2017-01-01 至最新完成擷取之交易日。
* 資料來源：歷史價量使用 Yahoo Finance API，除權息與公司行動使用 FinMind API。

## 2. 品質聲明與倖存者偏差 (Survivorship Bias)
受限於免費開源資料庫對台股歷史 metadata 的維護限制，本資料集存在倖存者偏差。資料集主要涵蓋「擷取當下仍維持上市／上櫃狀態」之企業。期間內已下市、下櫃或因併購而消滅之公司，因 API 無法提供歷史時區或完整報價，已於抓取過程中自動剔除，並未完整涵蓋歷史全市場。

## 3. 檔案結構 (File Descriptions)
* stock_price.csv：主表，包含每日價量、調整因子與目標報酬率。
* stock_list.csv：股票清單，記載納入本資料集的公司代碼與所屬市場。
* corporate_actions.csv：公司行動表，記載歷史除權息紀錄。
* trading_status.csv：(本版暫無) 預留予處置股/注意股紀錄。

## 4. stock_price.csv 欄位定義與處理規則

* RowId: 唯一識別碼，格式為 YYYYMMDD_SecuritiesCode (例: 20240102_2330)。
* Date: 台灣市場實際交易日，格式為 YYYY-MM-DD。
* SecuritiesCode: 股票代碼，保留為文字型態。
* Open: 原始開盤價 (新台幣元)。未復權，缺值填 NA。
* High: 原始最高價 (新台幣元)。未復權，缺值填 NA。
* Low: 原始最低價 (新台幣元)。未復權，缺值填 NA。
* Close: 原始收盤價 (新台幣元)。未復權，缺值填 NA。
* Volume: 成交股數 (非張數)。包含普通交易與零股。
* AdjustmentFactor: 單次價格調整因子。無事件日預設為 1.0。於「除權日當天」觸發，計算公式為 1 / (1 + (StockDividend / 10))。
* ExpectedDividend: 預期每股現金股利。本資料集統一填 NA。為避免未來數據滲透 (Look-ahead bias)，原需於宣告日至除息日區間填寫，因缺乏可靠歷史宣告日，基於不猜測原則全數留空。
* SupervisionFlag: 監理狀態 (注意股/處置股)。本資料集統一填 NA。台股處置機制與 JPX 無法直接等價，目前保留空值。
* Target: 預測目標 (明日收盤至後日收盤之報酬率)。僅作標籤，不可作為特徵輸入。公式為 (AdjClose_t2 / AdjClose_t1) - 1。若遇停牌或無報價，填補為 NA。

## 5. 開發與重製 (Reproducibility)
若需重新執行抓取，請確保環境已安裝 pandas, yfinance, requests, tqdm 套件。
自動化管線執行順序為：
1. 獲取台股清單 (生成 stock_list.csv)
2. 執行歷史價量管線 (生成 stock_price_full.csv)
3. 執行除權息抓取 (生成 corporate_actions.csv)
4. 合併特徵與計算因子 (生成最終 stock_price_final.csv)
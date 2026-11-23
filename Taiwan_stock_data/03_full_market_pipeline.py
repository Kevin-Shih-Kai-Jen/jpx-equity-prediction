import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

def sanitize_ohlc(df, stock_id):
    """檢查並清洗 OHLC 異常資料（例如 Close < Low 或 Close > High）"""
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    for col in ohlc_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 合理性條件：皆 > 0，且 Low <= min(Open, Close) 且 High >= max(Open, Close)
    invalid_mask = (
        (df['Open'] <= 0) | (df['High'] <= 0) | (df['Low'] <= 0) | (df['Close'] <= 0) |
        (df['High'] < df['Low']) |
        (df['Open'] < df['Low']) | (df['Open'] > df['High']) |
        (df['Close'] < df['Low']) | (df['Close'] > df['High'])
    )

    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        # 將異常列的 OHLC 設為 NA，避免錯誤價格污染特徵與 Target
        df.loc[invalid_mask, ohlc_cols] = np.nan

    return df, invalid_count

def process_single_stock(stock_id, market_type, start_date):
    """處理單一檔股票並轉為 JPX 格式"""
    # 相容 FinMind (twse/tpex) 與中文 (上市/上櫃)
    is_twse = str(market_type).lower() in ['twse', '上市']
    suffix = ".TW" if is_twse else ".TWO"
    ticker = f"{stock_id}{suffix}"

    df = yf.download(ticker, start=start_date, auto_adjust=False, progress=False)
    if df.empty:
        return None, 0

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    # 確保依日期排序並去除重複日期
    df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)

    # 清洗 OHLC 異常
    df, invalid_count = sanitize_ohlc(df, stock_id)

    # 欄位格式化
    df['SecuritiesCode'] = str(stock_id)
    df['Date_str'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df['RowId'] = pd.to_datetime(df['Date']).dt.strftime('%Y%m%d') + '_' + str(stock_id)

    # 外部欄位先填預設值，Target 留待 05_merge_factors.py 統一重算
    df['AdjustmentFactor'] = 1.0
    df['ExpectedDividend'] = np.nan
    df['SupervisionFlag'] = np.nan
    df['Target'] = np.nan

    df['Date'] = df['Date_str']
    columns_order = [
        'RowId', 'Date', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close',
        'Volume', 'AdjustmentFactor', 'ExpectedDividend', 'SupervisionFlag', 'Target'
    ]
    return df[columns_order], invalid_count

def run_market_pipeline(start_date="2017-01-01"):
    raw_dir = "data/raw_jpx_prices"
    os.makedirs(raw_dir, exist_ok=True)

    if not os.path.exists("stock_list.csv"):
        print("找不到 stock_list.csv，請先執行清單抓取腳本。")
        return

    stock_list = pd.read_csv("stock_list.csv", dtype={'stock_id': str})
    total_stocks = len(stock_list)
    print(f"啟動全市場抓取管線，共計 {total_stocks} 檔股票，起始日：{start_date}")

    failed_list = []
    anomaly_summary = []

    for _, row in tqdm(stock_list.iterrows(), total=total_stocks, desc="下載進度"):
        stock_id = row['stock_id']
        market_type = row.get('type', 'twse')
        save_path = os.path.join(raw_dir, f"stock_{stock_id}.csv")

        if os.path.exists(save_path):
            continue

        try:
            df, invalid_count = process_single_stock(stock_id, market_type, start_date)
            if df is not None and not df.empty:
                df.to_csv(save_path, index=False, encoding='utf-8', na_rep='NA')
                if invalid_count > 0:
                    anomaly_summary.append((stock_id, invalid_count))
            else:
                failed_list.append(stock_id)
        except Exception:
            failed_list.append(stock_id)

        time.sleep(0.1)

    print("\n所有單檔下載完成，準備合併為 stock_price_full.csv...")
    all_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.csv')]

    if all_files:
        df_list = [pd.read_csv(f, dtype={'SecuritiesCode': str}) for f in tqdm(all_files, desc="合併進度")]
        final_df = pd.concat(df_list, ignore_index=True)

        # 全域去重並依 Date, SecuritiesCode 排序
        final_df = final_df.drop_duplicates(subset=['Date', 'SecuritiesCode'], keep='last')
        final_df = final_df.sort_values(['Date', 'SecuritiesCode']).reset_index(drop=True)

        final_df.to_csv("stock_price_full.csv", index=False, encoding='utf-8', na_rep='NA')
        print(f"\n🎉 成功合併全市場資料！共 {len(final_df)} 筆")

    if anomaly_summary:
        print(f"⚠️ 已將 {len(anomaly_summary)} 檔股票的 OHLC 異常列清空為 NA，前 10 筆：{anomaly_summary[:10]}")
    if failed_list:
        print(f"⚠️ 以下股票代碼抓取失敗或無資料：{failed_list}")

if __name__ == "__main__":
    run_market_pipeline(start_date="2017-01-01")
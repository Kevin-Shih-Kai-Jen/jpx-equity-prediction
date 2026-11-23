import yfinance as yf
import pandas as pd
import numpy as np

def get_corporate_actions(sec_code):
    """
    【模組 A：取得公司行動資料】
    這裡模擬你從 FinMind 或 Goodinfo 抓下來的除權息與減資資料。
    台積電 (2330) 2024 年度的配息紀錄範例：
    """
    data = {
        'SecuritiesCode': [sec_code, sec_code],
        'AnnounceDate': ['2024-02-06', '2024-05-10'],  # 董事會宣告日 (當天盤後或隔天生效)
        'ExDate': ['2024-03-18', '2024-06-13'],        # 除息/除權日
        'CashDividend': [3.5, 4.0],                    # 現金股利
        'StockDividend': [0.0, 0.0]                    # 股票股利 (若配1元則為0.1股)
    }
    return pd.DataFrame(data)

def build_jpx_stock_price(ticker="2330.TW", start_date="2024-01-01"):
    print(f"正在從 yfinance 獲取 {ticker} 的歷史報價...")
    
    # 【模組 B：取得原始報價】
    df = yf.download(ticker, start=start_date, auto_adjust=False)
    if df.empty:
        print("查無資料。")
        return
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    
    sec_code = ticker.replace(".TW", "").replace(".TWO", "")
    df['SecuritiesCode'] = sec_code
    df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['RowId'] = df['Date'].dt.strftime('%Y%m%d') + '_' + sec_code
    
    # 確保依時間排序，這是時間序列填補的基礎
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 取得除權息事件表
    actions_df = get_corporate_actions(sec_code)
    
    # ==========================================
    # 【模組 C：欄位合併與轉換 (資料工程核心)】
    # ==========================================
    
    # 1. 處理 ExpectedDividend (預期股利)
    df['ExpectedDividend'] = np.nan
    
    # 1a. 宣告日：標記預期股利金額
    # 實務上宣告通常在盤後，所以從宣告日的「下一個交易日」開始產生預期
    announce_df = actions_df[['AnnounceDate', 'CashDividend']].rename(columns={'AnnounceDate': 'Date_str'})
    df = df.merge(announce_df, on='Date_str', how='left')
    # 將 CashDividend shift(1)，代表宣告日隔天才生效
    df['TriggeredDividend'] = df['CashDividend'].shift(1) 
    df['ExpectedDividend'] = np.where(df['TriggeredDividend'].notna(), df['TriggeredDividend'], df['ExpectedDividend'])
    
    # 1b. 除息日：預期股利實現，數字歸零
    ex_df = actions_df[['ExDate']].copy()
    ex_df['ExDate_Zero'] = 0
    ex_df = ex_df.rename(columns={'ExDate': 'Date_str'})
    df = df.merge(ex_df, on='Date_str', how='left')
    df['ExpectedDividend'] = np.where(df['ExDate_Zero'].notna(), df['ExDate_Zero'], df['ExpectedDividend'])
    
    # 1c. 神奇的 ffill (向前填補)：把宣告日到除息日間的 NaN 填上股利數字
    df['ExpectedDividend'] = df['ExpectedDividend'].ffill()
    
    # 1d. 把 0 轉回 NaN (平時沒有預期股利的狀態)
    df['ExpectedDividend'] = df['ExpectedDividend'].replace(0, np.nan)
    
    # 清理暫存欄位
    df = df.drop(columns=['CashDividend', 'TriggeredDividend', 'ExDate_Zero'])
    
    # 2. 處理 AdjustmentFactor (價格調整因子)
    # 預設為 1.0
    df['AdjustmentFactor'] = 1.0
    
    # 只有在「除權日(ExDate)」當天有數值
    factor_df = actions_df[['ExDate', 'StockDividend']].rename(columns={'ExDate': 'Date_str'})
    df = df.merge(factor_df, on='Date_str', how='left')
    
    # 台股股票股利以面額 10 元算。配 1 元 = 1/10 = 0.1 股
    # 因子公式：1 / (1 + 股票股利/10)
    df['calc_factor'] = 1 / (1 + (df['StockDividend'] / 10))
    df['AdjustmentFactor'] = np.where(df['calc_factor'].notna(), df['calc_factor'], 1.0)
    df = df.drop(columns=['StockDividend', 'calc_factor'])

    # 3. 處理 SupervisionFlag (處置股狀態)
    df['SupervisionFlag'] = np.nan # 待串接證交所處置股資料
    
    # ==========================================
    # 【模組 D：計算 Target 與輸出】
    # ==========================================
    
    df['AdjClose_t1'] = df['Adj Close'].shift(-1)
    df['AdjClose_t2'] = df['Adj Close'].shift(-2)
    df['Target'] = (df['AdjClose_t2'] / df['AdjClose_t1']) - 1
    df['Target'] = df['Target'].replace([np.inf, -np.inf], np.nan)
    
    df['Date'] = df['Date_str']
    columns_order = [
        'RowId', 'Date', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 
        'Volume', 'AdjustmentFactor', 'ExpectedDividend', 'SupervisionFlag', 'Target'
    ]
    
    final_df = df[columns_order]
    
    # 檢驗 2024-02 到 2024-03 之間的預期股利變化
    print("\n資料合併完成！檢驗 [宣告日 02-06] 到 [除息日 03-18] 之間的 ExpectedDividend 變化：")
    mask = (final_df['Date'] >= '2024-02-05') & (final_df['Date'] <= '2024-03-20')
    print(final_df.loc[mask, ['Date', 'ExpectedDividend', 'AdjustmentFactor']].head(20))
    
    final_df.to_csv(f'stock_price_integrated_{sec_code}.csv', index=False, na_rep='NA')
    print(f"\n檔案已匯出：stock_price_integrated_{sec_code}.csv")

if __name__ == "__main__":
    build_jpx_stock_price(ticker="2330.TW", start_date="2024-01-01")
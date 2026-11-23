import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def build_jpx_stock_price(ticker="2330.TW", start_date="2024-01-01"):
    print(f"正在從 yfinance 獲取 {ticker} 的歷史資料...")
    
    # 1. 獲取資料：嚴格設定 auto_adjust=False，確保抓到的是「原始價格」
    df = yf.download(ticker, start=start_date, auto_adjust=False)
    
    if df.empty:
        print("查無資料，請檢查網路或股票代碼。")
        return
        
    # 處理 yfinance 新版本可能回傳的多層級欄位 (MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()
    
    # 2. 欄位格式化與建立
    # 取出純數字代碼 (去掉 .TW)
    sec_code = ticker.replace(".TW", "").replace(".TWO", "")
    df['SecuritiesCode'] = sec_code
    
    # 處理時間格式
    df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['RowId'] = df['Date'].dt.strftime('%Y%m%d') + '_' + sec_code
    
    # 3. 處理 JPX 要求的外部資料 (在此雛形中先填補為 NA)
    df['AdjustmentFactor'] = np.nan   # 需另由除權息資料源計算
    df['ExpectedDividend'] = np.nan   # 需另爬取董事會宣告
    df['SupervisionFlag'] = np.nan    # 注意股/處置股標記
    
    # 4. Target 計算邏輯 (核心)
    # 確保資料依日期遞增排序
    df = df.sort_values('Date')
    
    # 建立 T+1 與 T+2 的「調整後收盤價 (Adj Close)」來算報酬，避免除權息造成的價格斷層影響模型
    df['AdjClose_t1'] = df['Adj Close'].shift(-1)
    df['AdjClose_t2'] = df['Adj Close'].shift(-2)
    
    # Target 定義：( T+2調整後收盤 / T+1調整後收盤 ) - 1
    df['Target'] = (df['AdjClose_t2'] / df['AdjClose_t1']) - 1
    
    # 處理停牌或資料最末端無法計算產生的無限大或缺失值，統一替換為 NA
    df['Target'] = df['Target'].replace([np.inf, -np.inf], np.nan)
    
    # 將原始 Date 欄位蓋成 YYYY-MM-DD 文字格式
    df['Date'] = df['Date_str']
    
    # 5. 嚴格對齊你指定的欄位順序
    columns_order = [
        'RowId', 'Date', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 
        'Volume', 'AdjustmentFactor', 'ExpectedDividend', 'SupervisionFlag', 'Target'
    ]
    
    # 確保不會漏掉欄位 (yfinance 抓下來的 Volume 預設為股數，符合你的需求)
    final_df = df[columns_order]
    
    # 6. 輸出為 CSV
    output_filename = f'stock_price_{sec_code}.csv'
    final_df.to_csv(output_filename, index=False, encoding='utf-8', na_rep='NA')
    
    print(f"\n轉換成功！檔案已儲存為 {output_filename}")
    print("\n資料預覽 (最後 5 筆，注意 Target 末兩筆應為 NA)：")
    print(final_df[['Date', 'Close', 'Target']].tail())

if __name__ == "__main__":
    # 以台積電 (2330) 2024 年至今的資料做測試
    build_jpx_stock_price(ticker="2330.TW", start_date="2024-01-01")


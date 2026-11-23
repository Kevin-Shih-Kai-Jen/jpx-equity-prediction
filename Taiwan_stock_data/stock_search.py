import pandas as pd

def search_stock():
    # 嘗試讀取股票清單 CSV
    try:
        df = pd.read_csv("stock_list.csv")
    except FileNotFoundError:
        print("錯誤：找不到 'stock_list.csv'。")
        print("請確定你已經執行過前一個抓取清單的程式，且檔案在目前的資料夾中。")
        return

    print("=" * 40)
    print(" 歡迎使用【台股代碼查詢小工具】")
    print(" (輸入 'q' 或 'exit' 即可離開)")
    print("=" * 40)

    while True:
        # 讓使用者輸入查詢字串
        query = input("\n請輸入『公司名稱』或『4碼代號』: ").strip()
        
        # 檢查是否要離開程式
        if query.lower() in ['q', 'exit']:
            print("結束查詢，掰掰！")
            break
            
        if not query:
            continue
            
        # 使用 str.contains 進行模糊比對 (支援查代碼或查名稱)
        mask = (
            df['stock_id'].astype(str).str.contains(query, case=False, na=False) | 
            df['stock_name'].astype(str).str.contains(query, case=False, na=False)
        )
        
        results = df[mask]
        
        if results.empty:
            print(f"❌ 找不到包含「{query}」的上市櫃股票，請換個關鍵字試試。")
        else:
            print(f"✅ 找到 {len(results)} 筆結果：")
            print("-" * 40)
            
            # 挑選要顯示的欄位 (確保這些欄位存在於你的 CSV 中)
            display_cols = ['stock_id', 'stock_name', 'type', 'industry_category']
            cols_to_show = [c for c in display_cols if c in results.columns]
            
            # 印出排版好的表格，隱藏 index
            print(results[cols_to_show].to_string(index=False))
            print("-" * 40)

if __name__ == "__main__":
    search_stock()
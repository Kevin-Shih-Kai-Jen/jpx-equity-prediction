import os
import requests
import pandas as pd

def get_taiwan_stock_list():
    print("正在取得台股全市場股票代碼表...")
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {"dataset": "TaiwanStockInfo"}
    token = os.getenv("FINMIND_TOKEN", "")
    if token:
        parameter["token"] = token

    response = requests.get(url, params=parameter, timeout=15)
    response.raise_for_status()
    data = response.json()

    if data.get("msg") != "success":
        print(f"獲取失敗：{data}")
        return

    df = pd.DataFrame(data["data"])
    df["stock_id"] = df["stock_id"].astype(str).str.strip()

    # 1. 只保留上市 (twse) 與上櫃 (tpex)
    df = df[df["type"].isin(["twse", "tpex"])]

    # 2. 排除 ETF、ETN、受益證券、存託憑證、大盤指數
    excluded_categories = {"ETF", "ETN", "受益證券", "存託憑證", "Index", "大盤"}
    df = df[~df["industry_category"].fillna("").isin(excluded_categories)]

    # 3. 僅保留普通股 4 碼代碼（1000~9999，排除 00xx ETF 與 91xx TDR）
    df = df[df["stock_id"].str.match(r"^[1-9]\d{3}$")]
    df = df[~df["stock_id"].str.match(r"^91\d{2}$")]

    # 4. 去除重複代碼並排序
    df = df.drop_duplicates(subset=["stock_id"], keep="first").sort_values("stock_id").reset_index(drop=True)

    df.to_csv("stock_list.csv", index=False, encoding="utf-8-sig")
    print(f"成功！共 {len(df)} 檔上市櫃普通股，已儲存為 stock_list.csv")
    print(df[["stock_id", "stock_name", "type", "industry_category"]].head())

if __name__ == "__main__":
    get_taiwan_stock_list()
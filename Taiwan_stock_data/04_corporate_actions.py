import os
import time
import requests
import pandas as pd
from tqdm import tqdm

from finmind_auth import auth_headers, check_finmind_token, load_finmind_token

class QuotaExceededError(Exception):
    pass

def get_numeric_col(df, col_name):
    if col_name not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)

def get_clean_date_col(df, col_name):
    if col_name not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    s = df[col_name].astype(str).str.strip()
    valid = s.str.match(r"^\d{4}-\d{2}-\d{2}$") & (s != "0000-00-00")
    return s.where(valid, pd.NA)

def fetch_finmind_dividend(stock_id, start_date="2017-01-01", token=""):
    url = "https://api.finmindtrade.com/api/v4/data"
    headers = auth_headers(token)
    params = {
        "dataset": "TaiwanStockDividend",
        "data_id": str(stock_id),
        "start_date": start_date
    }

    resp = requests.get(url, headers=headers, params=params, timeout=15)
    if resp.status_code in (402, 429):
        raise QuotaExceededError(f"HTTP {resp.status_code}: {resp.text[:150]}")
    resp.raise_for_status()

    payload = resp.json()
    if payload.get("status") in (402, 429):
        raise QuotaExceededError(f"API status {payload.get('status')}: {payload.get('msg')}")
    if payload.get("msg") != "success":
        raise RuntimeError(f"FinMind 回傳錯誤: {payload.get('msg')}")

    rows = payload.get("data", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def build_corporate_actions(start_date="2017-01-01"):
    if not os.path.exists("stock_list.csv"):
        print("請先執行 StocksSymbol.py 產生 stock_list.csv")
        return

    token = load_finmind_token()
    if not check_finmind_token(token):
        return

    raw_dir = "data/raw_corporate_actions"
    os.makedirs(raw_dir, exist_ok=True)

    stock_list = pd.read_csv("stock_list.csv", dtype={"stock_id": str})
    stock_ids = stock_list["stock_id"].astype(str).str.strip().drop_duplicates().tolist()

    cols = [
        "SecuritiesCode", "AnnounceDate", "StockExDate", "CashExDate",
        "CashDividend", "StockDividend", "Source"
    ]
    failed_list = []

    print(f"開始抓取全市場除權息資料，共 {len(stock_ids)} 檔...")

    while True:
        remaining = [
            sid for sid in stock_ids
            if not os.path.exists(os.path.join(raw_dir, f"actions_{sid}.csv"))
        ]
        if not remaining:
            break

        print(f"目前尚餘 {len(remaining)} 檔待抓取...")
        quota_hit = False

        for stock_id in tqdm(remaining, desc="除權息進度"):
            save_path = os.path.join(raw_dir, f"actions_{stock_id}.csv")
            try:
                df = fetch_finmind_dividend(stock_id, start_date=start_date, token=token)
                if df.empty:
                    pd.DataFrame(columns=cols).to_csv(save_path, index=False, encoding="utf-8")
                else:
                    stock_div = (
                        get_numeric_col(df, "StockEarningsDistribution")
                        + get_numeric_col(df, "StockStatutorySurplus")
                    )
                    cash_div = (
                        get_numeric_col(df, "CashEarningsDistribution")
                        + get_numeric_col(df, "CashStatutorySurplus")
                    )
                    clean_df = pd.DataFrame({
                        "SecuritiesCode": stock_id,
                        "AnnounceDate": "NA",
                        "StockExDate": get_clean_date_col(df, "StockExDividendTradingDate"),
                        "CashExDate": get_clean_date_col(df, "CashExDividendTradingDate"),
                        "CashDividend": cash_div,
                        "StockDividend": stock_div,
                        "Source": "FinMind TaiwanStockDividend"
                    })
                    clean_df = clean_df[clean_df["StockExDate"].notna() | clean_df["CashExDate"].notna()]
                    clean_df.to_csv(save_path, index=False, encoding="utf-8", na_rep="NA")
            except QuotaExceededError as e:
                print(f"\n⏳ {e}")
                print("已到達單小時上限，自動休眠 65 分鐘等待配額重置後繼續（請勿關閉視窗）...")
                quota_hit = True
                break
            except Exception as e:
                failed_list.append((stock_id, str(e)))

            time.sleep(0.3)

        if quota_hit:
            time.sleep(65 * 60)
            check_finmind_token(token)
        else:
            break

    # 只合併在目前 stock_list 內的檔案，排除舊清單殘留的興櫃股
    valid_set = set(stock_ids)
    all_files = [
        os.path.join(raw_dir, f"actions_{sid}.csv")
        for sid in valid_set
        if os.path.exists(os.path.join(raw_dir, f"actions_{sid}.csv"))
    ]

    df_list = []
    for f in tqdm(all_files, desc="合併進度"):
        tmp = pd.read_csv(f, dtype={"SecuritiesCode": str, "StockExDate": str, "CashExDate": str})
        if not tmp.empty:
            df_list.append(tmp)

    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df = final_df.drop_duplicates(
            subset=["SecuritiesCode", "StockExDate", "CashExDate", "CashDividend", "StockDividend"]
        )
        final_df = final_df.sort_values(["SecuritiesCode", "StockExDate", "CashExDate"]).reset_index(drop=True)
        final_df.to_csv("corporate_actions.csv", index=False, encoding="utf-8", na_rep="NA")
        print(f"\n🎉 成功建立 corporate_actions.csv！共 {len(final_df)} 筆紀錄。")
        print(f"其中含股票股利筆數：{((final_df['StockDividend'] > 0) & final_df['StockExDate'].notna()).sum()}")

    if failed_list:
        print(f"⚠️ 抓取失敗（共 {len(failed_list)} 檔），前 10 筆：{failed_list[:10]}")

if __name__ == "__main__":
    build_corporate_actions(start_date="2017-01-01")
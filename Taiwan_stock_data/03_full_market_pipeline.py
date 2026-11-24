"""
03 全市場歷史價量管線（FinMind 原始報價版）

為什麼不用 yfinance：
    Yahoo 會把台股「配股」當成股票分割，回溯調整歷史價格，且 Close 與 Open/High/Low
    的調整幅度有時不一致（例：1264 德麥 2017-07-03 Close < Low）。
    後面 05_merge_factors.py 會再乘一次 AdjustmentFactor，等於重複調整。
    FinMind TaiwanStockPrice 是交易所公布的「未調整原始報價」，除權日會正常跳空，
    與 corporate_actions.csv 的除權日完全對得上。

用法：
    python 03_full_market_pipeline.py                 # 只下載還沒抓過的股票，然後合併
    python 03_full_market_pipeline.py --force         # 全部重新下載
    python 03_full_market_pipeline.py --start 2017-01-01 --end 2026-11-20

輸出：
    data/raw_finmind_prices/stock_<代碼>.csv   單檔快取（新資料夾，舊的 Yahoo 資料不會被沿用）
    stock_price_full.csv                        全市場合併檔（給 05_merge_factors.py 使用）
    data/price_anomalies.csv                    清洗時發現的異常列（若有）
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from finmind_auth import auth_headers, check_finmind_token, load_finmind_token

for _s in (sys.stdout, sys.stderr):  # Windows cp950 主控台／輸出導向檔案時印 emoji 不會當掉
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 不論從哪個資料夾執行，stock_list.csv、data/ 等相對路徑都以本檔所在資料夾為準
os.chdir(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = "data/raw_finmind_prices"
API_URL = "https://api.finmindtrade.com/api/v4/data"
COLUMNS_ORDER = [
    "RowId", "Date", "SecuritiesCode", "Open", "High", "Low", "Close",
    "Volume", "AdjustmentFactor", "ExpectedDividend", "SupervisionFlag", "Target",
]
OHLC = ["Open", "High", "Low", "Close"]


class QuotaExceededError(Exception):
    pass


def fetch_finmind_price(stock_id, start_date, end_date, token):
    """抓單一股票的原始日報價（未還原）"""
    params = {"dataset": "TaiwanStockPrice", "data_id": str(stock_id), "start_date": start_date}
    if end_date:
        params["end_date"] = end_date
    headers = auth_headers(token)

    resp = requests.get(API_URL, headers=headers, params=params, timeout=30)
    if resp.status_code in (402, 429):
        raise QuotaExceededError(f"HTTP {resp.status_code}: {resp.text[:150]}")
    resp.raise_for_status()

    payload = resp.json()
    if payload.get("status") in (402, 429):
        raise QuotaExceededError(f"API status {payload.get('status')}: {payload.get('msg')}")
    if payload.get("msg") != "success":
        raise RuntimeError(f"FinMind 回傳錯誤: {payload.get('msg')}")
    return pd.DataFrame(payload.get("data", []))


def to_jpx_format(raw, stock_id):
    """FinMind 欄位 → JPX 欄位，並回傳 (整理後資料, 異常列)"""
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS_ORDER), pd.DataFrame()

    df = pd.DataFrame({
        "Date": pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d"),
        "Open": pd.to_numeric(raw["open"], errors="coerce"),
        "High": pd.to_numeric(raw["max"], errors="coerce"),
        "Low": pd.to_numeric(raw["min"], errors="coerce"),
        "Close": pd.to_numeric(raw["close"], errors="coerce"),
        "Volume": pd.to_numeric(raw["Trading_Volume"], errors="coerce").fillna(0).astype("int64"),
    })
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    # 無成交日：FinMind 的價格欄位是 0 → 改成 NA（不是錯誤，不列入異常）
    no_trade = (df[OHLC] <= 0).any(axis=1) | df[OHLC].isna().any(axis=1)
    df.loc[no_trade, OHLC] = np.nan

    # 原始報價理論上不會違反 OHLC 邏輯；若有，記錄下來並清成 NA，不讓它污染 Target
    o, h, l, c = (df[k] for k in OHLC)
    invalid = ~no_trade & ((h < l) | (o < l) | (o > h) | (c < l) | (c > h))
    anomalies = df[invalid].assign(SecuritiesCode=str(stock_id))
    df.loc[invalid, OHLC] = np.nan

    df["SecuritiesCode"] = str(stock_id)
    df["RowId"] = df["Date"].str.replace("-", "", regex=False) + "_" + str(stock_id)
    df["AdjustmentFactor"] = 1.0       # 由 05_merge_factors.py 依除權事件填入
    df["ExpectedDividend"] = np.nan
    df["SupervisionFlag"] = np.nan
    df["Target"] = np.nan              # 由 05_merge_factors.py 統一計算
    return df[COLUMNS_ORDER], anomalies


def download_all(stock_ids, start_date, end_date, token, force):
    os.makedirs(RAW_DIR, exist_ok=True)
    failed, anomalies = [], []

    while True:
        remaining = [
            sid for sid in stock_ids
            if force or not os.path.exists(os.path.join(RAW_DIR, f"stock_{sid}.csv"))
        ]
        force = False  # 只有第一輪強制重抓；配額休眠後只補剩下的
        if not remaining:
            break
        print(f"尚需下載 {len(remaining)} 檔...")

        quota_hit = False
        for sid in tqdm(remaining, desc="下載進度"):
            save_path = os.path.join(RAW_DIR, f"stock_{sid}.csv")
            try:
                raw = fetch_finmind_price(sid, start_date, end_date, token)
                df, bad = to_jpx_format(raw, sid)
                # 沒資料也寫一個只有表頭的檔，避免每次重跑都浪費配額
                df.to_csv(save_path, index=False, encoding="utf-8", na_rep="NA")
                if not bad.empty:
                    anomalies.append(bad)
            except QuotaExceededError as e:
                print(f"\n⏳ {e}\n已達 FinMind 每小時上限，休眠 65 分鐘後自動繼續（請勿關閉視窗）...")
                quota_hit = True
                break
            except Exception as e:
                failed.append((sid, str(e)[:120]))
            time.sleep(0.3)

        if quota_hit:
            time.sleep(65 * 60)
            check_finmind_token(token)
        else:
            break
    return failed, anomalies


def merge_all(stock_ids):
    files = [os.path.join(RAW_DIR, f"stock_{sid}.csv") for sid in stock_ids]
    files = [f for f in files if os.path.exists(f)]
    parts = []
    for f in tqdm(files, desc="合併進度"):
        tmp = pd.read_csv(f, dtype={"SecuritiesCode": str})
        if not tmp.empty:
            parts.append(tmp)
    if not parts:
        print("沒有任何報價資料可合併。")
        return None

    full = pd.concat(parts, ignore_index=True)
    full = full.drop_duplicates(subset=["Date", "SecuritiesCode"], keep="last")
    full = full.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    full.to_csv("stock_price_full.csv", index=False, encoding="utf-8", na_rep="NA")
    empty = len(files) - len(parts)
    print(f"\n🎉 stock_price_full.csv：{len(full):,} 筆、{full['SecuritiesCode'].nunique()} 檔股票"
          f"（{empty} 檔在區間內沒有報價）")
    return full


def run_market_pipeline(start_date="2017-01-01", end_date=None, force=False):
    if not os.path.exists("stock_list.csv"):
        print("找不到 stock_list.csv，請先執行 StocksSymbol.py。")
        return

    token = load_finmind_token()
    if not check_finmind_token(token):
        return

    stock_list = pd.read_csv("stock_list.csv", dtype={"stock_id": str}, encoding="utf-8-sig")
    stock_ids = stock_list["stock_id"].astype(str).str.strip().drop_duplicates().tolist()
    print(f"全市場原始報價管線：{len(stock_ids)} 檔，期間 {start_date} ~ {end_date or '最新'}"
          f"{'（強制重抓）' if force else ''}")

    failed, anomalies = download_all(stock_ids, start_date, end_date, token, force)
    merge_all(stock_ids)

    if anomalies:
        anom = pd.concat(anomalies, ignore_index=True)
        anom.to_csv("data/price_anomalies.csv", index=False, encoding="utf-8")
        print(f"⚠️ {len(anom)} 列 OHLC 邏輯異常，已清成 NA，明細見 data/price_anomalies.csv")
    if failed:
        print(f"⚠️ {len(failed)} 檔抓取失敗（重跑本程式會自動補抓），前 10 筆：{failed[:10]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="以 FinMind 原始報價建立 stock_price_full.csv")
    ap.add_argument("--start", default="2017-01-01", help="起始日 YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="結束日 YYYY-MM-DD（預設抓到最新）")
    ap.add_argument("--force", action="store_true", help="忽略快取，全部重新下載")
    args = ap.parse_args()
    run_market_pipeline(start_date=args.start, end_date=args.end, force=args.force)

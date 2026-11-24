import pandas as pd
import numpy as np

def compute_adjusted_target(df):
    """
    全向量化計算（相容 Pandas 1.x / 2.x / 3.x）：
    1. 依 SecuritiesCode + Date 升冪排序
    2. AdjustmentFactor 位於「除權日當天」，因此除權日前一日（含以前）價格乘上累積調整因子
    3. Target(t) = AdjustedClose(t+2) / AdjustedClose(t+1) - 1
    """
    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    cum_factor = (
        df.iloc[::-1]
        .groupby("SecuritiesCode", sort=False)["AdjustmentFactor"]
        .cumprod()
        .iloc[::-1]
    )
    df["_cum_factor"] = cum_factor

    price_multiplier = (
        df.groupby("SecuritiesCode", sort=False)["_cum_factor"]
        .shift(-1)
        .fillna(1.0)
    )
    df["_adj_close"] = df["Close"] * price_multiplier

    adj_t1 = df.groupby("SecuritiesCode", sort=False)["_adj_close"].shift(-1)
    adj_t2 = df.groupby("SecuritiesCode", sort=False)["_adj_close"].shift(-2)

    target = (adj_t2 / adj_t1) - 1.0
    target = target.where((adj_t1 > 0) & (adj_t2 > 0), np.nan)
    df["Target"] = target.replace([np.inf, -np.inf], np.nan)

    return df.drop(columns=["_cum_factor", "_adj_close"])

def validate_final_dataset(df):
    """完整執行修正單要求之四項交付前驗證"""
    print("\n" + "=" * 60)
    print("🔍 交付前全量驗證報告")
    print("=" * 60)

    # 1. 重複股票日期 & RowId 檢查
    dup_key = int(df.duplicated(subset=["Date", "SecuritiesCode"]).sum())
    dup_rowid = int(df.duplicated(subset=["RowId"]).sum())
    print(f"[1] 重複 (Date, SecuritiesCode)：{dup_key} 筆 | 重複 RowId：{dup_rowid} 筆 "
          f"{'✅' if (dup_key == 0 and dup_rowid == 0) else '❌'}")

    # 2. OHLC 合理性檢查
    valid_ohlc = df.dropna(subset=["Open", "High", "Low", "Close"])
    ohlc_err = int((
        (valid_ohlc["Open"] <= 0) | (valid_ohlc["High"] <= 0) |
        (valid_ohlc["Low"] <= 0) | (valid_ohlc["Close"] <= 0) |
        (valid_ohlc["High"] < valid_ohlc["Low"]) |
        (valid_ohlc["Open"] < valid_ohlc["Low"]) | (valid_ohlc["Open"] > valid_ohlc["High"]) |
        (valid_ohlc["Close"] < valid_ohlc["Low"]) | (valid_ohlc["Close"] > valid_ohlc["High"])
    ).sum())
    print(f"[2] OHLC 邏輯異常筆數：{ohlc_err} 筆 {'✅' if ohlc_err == 0 else '❌'}")

    # 3. 交易日缺口檢查（對照全市場交易日曆，檢查各股在首尾日期區間內缺少的天數）
    market_dates = np.sort(df["Date"].unique())
    date_to_idx = {d: i for i, d in enumerate(market_dates)}
    stock_span = df.groupby("SecuritiesCode")["Date"].agg(first="min", last="max", actual_days="count")
    stock_span["expected_days"] = (
        stock_span["last"].map(date_to_idx) - stock_span["first"].map(date_to_idx) + 1
    )
    stock_span["missing_days"] = stock_span["expected_days"] - stock_span["actual_days"]
    gapped_stocks = stock_span[stock_span["missing_days"] > 0]
    print(f"[3] 交易日缺口檢查：共 {len(gapped_stocks)} 檔在存續區間內有缺日（如停牌、減資暫停交易），"
          f"最大單股缺口 {int(stock_span['missing_days'].max())} 天")
    if not gapped_stocks.empty:
        top_gaps = gapped_stocks.sort_values("missing_days", ascending=False).head(5)
        print("    缺口前 5 檔供覆核：")
        for code, r in top_gaps.iterrows():
            print(f"      - {code}: 缺 {int(r['missing_days'])} 天 ({r['first']} ~ {r['last']})")

    # 4. Target 全量獨立重算比對
    recalc_df = compute_adjusted_target(df[["Date", "SecuritiesCode", "Close", "AdjustmentFactor"]].copy())
    orig_sorted = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    na_mismatch = int((orig_sorted["Target"].isna() != recalc_df["Target"].isna()).sum())
    both_valid = orig_sorted["Target"].notna() & recalc_df["Target"].notna()
    max_abs_diff = float((orig_sorted.loc[both_valid, "Target"] - recalc_df.loc[both_valid, "Target"]).abs().max()) if both_valid.any() else 0.0
    target_ok = (na_mismatch == 0) and (max_abs_diff < 1e-12)
    print(f"[4] Target 全量重算比對：NA 位置差異 {na_mismatch} 筆 | 最大絕對誤差 {max_abs_diff:.2e} "
          f"{'✅' if target_ok else '❌'}")

    # 5. 因子變化與極端報酬提示（幫助排查未涵蓋的現金減資或股票分割）
    non_one = int((df["AdjustmentFactor"] != 1.0).sum())
    extreme_targets = df[df["Target"].abs() > 0.15]
    print(f"[5] AdjustmentFactor != 1.0 筆數：{non_one} 筆 {'✅' if non_one > 0 else '⚠️ 需檢查來源'}")
    print(f"[6] 異常日報酬 (|Target| > 15%)：{len(extreme_targets)} 筆"
          "（多為大幅現金除息、現金減資或股票分割所致，建議抽查確認）")
    if not extreme_targets.empty:
        print(extreme_targets[["Date", "SecuritiesCode", "Close", "AdjustmentFactor", "Target"]].head(5).to_string(index=False))
    print("=" * 60)

def align_events_to_trading_days(events, df_price, max_days=14):
    """
    把除權事件對到「該股在除權日當天或之後的第一筆報價」。
    events 需有 SecuritiesCode, Date, StockDividend；回傳同欄位，Date 為實際套用因子的交易日。
    超過 max_days 天都找不到報價（例：長期停牌、區間外）的事件會被捨棄並印出。
    """
    min_d, max_d = df_price["Date"].min(), df_price["Date"].max()
    ev = events[(events["Date"] >= min_d) & (events["Date"] <= max_d)].copy()
    ev["_d"] = pd.to_datetime(ev["Date"])
    ev = ev.sort_values("_d")

    rows = df_price[["SecuritiesCode", "Date"]].drop_duplicates().rename(columns={"Date": "TradeDate"})
    rows["_d"] = pd.to_datetime(rows["TradeDate"])
    rows = rows.sort_values("_d")

    m = pd.merge_asof(ev, rows, on="_d", by="SecuritiesCode",
                      direction="forward", tolerance=pd.Timedelta(days=max_days))
    exact = int((m["TradeDate"] == m["Date"]).sum())
    shifted = m[m["TradeDate"].notna() & (m["TradeDate"] != m["Date"])]
    dropped = m[m["TradeDate"].isna()]
    print(f"除權事件比對：區間內共 {len(m)} 筆，除權日當天對上 {exact} 筆，"
          f"順延到下一個交易日 {len(shifted)} 筆，{max_days} 天內找不到報價 {len(dropped)} 筆")
    if not shifted.empty:
        print("    順延明細（前 10 筆）：" + "、".join(
            f"{r.SecuritiesCode} {r.Date}→{r.TradeDate}" for r in shifted.head(10).itertuples()))
    if not dropped.empty:
        print("    找不到報價（前 10 筆）：" + "、".join(
            f"{r.SecuritiesCode} {r.Date}" for r in dropped.head(10).itertuples()))

    m = m.dropna(subset=["TradeDate"])
    # 兩個事件順延到同一天時，股票股利加總（與同日多筆事件的處理一致）
    return (m.groupby(["SecuritiesCode", "TradeDate"], as_index=False)["StockDividend"].sum()
             .rename(columns={"TradeDate": "Date"}))


def merge_adjustment_factors():
    print("正在讀取全市場報價資料 (stock_price_full.csv)...")
    df_price = pd.read_csv("stock_price_full.csv", dtype={"SecuritiesCode": str})

    print("正在讀取公司行動資料 (corporate_actions.csv)...")
    df_actions = pd.read_csv(
        "corporate_actions.csv",
        dtype={"SecuritiesCode": str, "StockExDate": str, "CashExDate": str}
    )

    df_actions["StockDividend"] = pd.to_numeric(df_actions["StockDividend"], errors="coerce").fillna(0.0)

    # 嚴格只使用 StockExDate 與 StockDividend > 0 的除權事件
    df_splits = df_actions[(df_actions["StockDividend"] > 0) & df_actions["StockExDate"].notna()].copy()
    df_splits = (
        df_splits.groupby(["SecuritiesCode", "StockExDate"], as_index=False)["StockDividend"]
        .sum()
        .rename(columns={"StockExDate": "Date"})
    )

    # 除權日遇到休市（颱風假等）或該股當天沒有資料時，順延到該股下一個有資料的交易日，
    # 否則因子會整筆遺失（例：3293 鈊象 2024-07-24 颱風休市，配股 10 元沒被調整）
    df_splits = align_events_to_trading_days(df_splits, df_price)

    print("開始合併 AdjustmentFactor...")
    merged_df = pd.merge(
        df_price,
        df_splits[["SecuritiesCode", "Date", "StockDividend"]],
        on=["SecuritiesCode", "Date"],
        how="left"
    )

    merged_df["calc_factor"] = 1.0 / (1.0 + (merged_df["StockDividend"] / 10.0))
    merged_df["AdjustmentFactor"] = merged_df["calc_factor"].fillna(1.0)
    merged_df = merged_df.drop(columns=["StockDividend", "calc_factor"])

    # 缺乏可靠宣告日與對應監理狀態，依規範明確維持 NA（不可填 0 或 False）
    merged_df["ExpectedDividend"] = np.nan
    merged_df["SupervisionFlag"] = np.nan

    print("按股票 + 交易日向量化重算 Target...")
    merged_df = compute_adjusted_target(merged_df)

    merged_df = merged_df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    columns_order = [
        "RowId", "Date", "SecuritiesCode", "Open", "High", "Low", "Close",
        "Volume", "AdjustmentFactor", "ExpectedDividend", "SupervisionFlag", "Target"
    ]
    final_df = merged_df[columns_order]

    validate_final_dataset(final_df)

    print("\n正在儲存為 stock_price_final.csv...")
    final_df.to_csv("stock_price_final.csv", index=False, encoding="utf-8", na_rep="NA")
    print(f"🎉 全部處理完畢！最終資料集共計 {len(final_df)} 筆。")

if __name__ == "__main__":
    merge_adjustment_factors()
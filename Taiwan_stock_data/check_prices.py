"""
台股 JPX 格式資料檢查程式
================================
用法：
    python check_prices.py stock_price_final.csv
    python check_prices.py data/raw_finmind_prices        # 也可直接檢查單檔資料夾
    python check_prices.py stock_price_final.csv --actions corporate_actions.csv --stock-list stock_list.csv

輸出：
    1. 終端機報告（每項檢查 ✅ / ⚠️ / ❌）
    2. check_issues.csv：每一筆有問題的資料列（股票、日期、檢查項目、說明）
    3. 結束碼：有任何 ❌ 時回傳 1，可接在 pipeline 後面當作「不通過就不能訓練」的關卡

與 05_merge_factors.py 內建驗證的差別：
    - Target 用「另一條公式」獨立重算，而不是呼叫同一個 compute_adjusted_target()，
      否則程式寫錯也會自己驗證通過。
    - 多檢查「價格是否為原始報價」（台股升降單位）、「漲跌幅是否超過 10%」、
      「除權日價格是否已被 Yahoo 調整過（重複調整）」，這三項正是修正單上
      Close < Low 與 Target 對不上的根本原因。

AdjustmentFactor 慣例：依 README，因子記在「除權日當天」，
    調整後收盤 AdjClose(d) = Close(d) × Π(d 之後每天的因子)
    → Target(t) = Close(t+2) / (Close(t+1) × Factor(t+2)) − 1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

EXPECTED_COLS = [
    "RowId", "Date", "SecuritiesCode", "Open", "High", "Low", "Close",
    "Volume", "AdjustmentFactor", "ExpectedDividend", "SupervisionFlag", "Target",
]
PRICE_COLS = ["Open", "High", "Low", "Close"]

LIMIT = 0.10            # 台股漲跌幅限制
LIMIT_TOL = 0.005       # 四捨五入到升降單位的容許誤差
IPO_FREE_DAYS = 5       # 新上市前 5 日無漲跌幅限制
GAP_DAYS = 3            # 停牌超過 3 個交易日後恢復交易（如減資）不套用漲跌幅檢查
TARGET_TOL = 1e-6      # CSV 存檔的浮點誤差遠小於此值
OFF_TICK_STOCK_RATIO = 0.05   # 單檔超過 5% 價格不在升降單位上 → 判定為調整後價格
EVENT_SHIFT_DAYS = 14   # 除權日遇休市時，順延到該股下一筆報價（與 05_merge_factors.py 相同）
# 證交所曾在補班的週六開市（2019 年起補班日不再交易），這些日期是真實交易日
TW_SATURDAY_TRADING_DAYS = {"2017-02-18", "2017-06-03", "2017-09-30", "2018-03-31", "2018-12-22"}


class Report:
    def __init__(self):
        self.issues = []
        self.summary = []

    def add_rows(self, check, level, rows, detail):
        if rows is None or len(rows) == 0:
            return
        part = pd.DataFrame({
            "check": check,
            "level": level,
            "SecuritiesCode": rows["SecuritiesCode"].values if "SecuritiesCode" in rows else "",
            "Date": rows["Date"].values if "Date" in rows else "",
            "detail": detail if isinstance(detail, str) else np.asarray(detail),
        })
        self.issues.append(part)

    def line(self, level, text):
        icon = {"OK": "✅", "WARN": "⚠️ ", "ERROR": "❌", "INFO": "ℹ️ "}[level]
        self.summary.append(level)
        print(f"{icon} {text}")

    @property
    def has_error(self):
        return "ERROR" in self.summary


def tick_size(p):
    """台股（上市/上櫃普通股）升降單位"""
    return np.select(
        [p < 10, p < 50, p < 100, p < 500, p < 1000],
        [0.01, 0.05, 0.1, 0.5, 1.0],
        default=5.0,
    )


def load_prices(path):
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith(".csv"))
        print(f"讀取資料夾 {path}，共 {len(files)} 個檔案...")
        df = pd.concat(
            (pd.read_csv(os.path.join(path, f), dtype={"SecuritiesCode": str}) for f in files),
            ignore_index=True,
        )
    else:
        print(f"讀取 {path} ...")
        df = pd.read_csv(path, dtype={"SecuritiesCode": str})
    print(f"共 {len(df):,} 筆、{df['SecuritiesCode'].nunique():,} 檔股票\n")
    return df


def load_actions(path):
    if not path or not os.path.exists(path):
        return None
    a = pd.read_csv(path, dtype={"SecuritiesCode": str, "StockExDate": str, "CashExDate": str})
    for c in ["StockDividend", "CashDividend"]:
        a[c] = pd.to_numeric(a[c], errors="coerce").fillna(0.0)
    return a


# ---------------------------------------------------------------------------
def align_events(ev, df):
    """ev 有 SecuritiesCode, Date → 加上 TradeDate：該股在 Date 當天或之後第一筆報價的日期（找不到為 NA）"""
    ev = ev.copy()
    ev["_d"] = pd.to_datetime(ev["Date"])
    rows = df[["SecuritiesCode", "Date"]].drop_duplicates().rename(columns={"Date": "TradeDate"})
    rows["_d"] = pd.to_datetime(rows["TradeDate"])
    m = pd.merge_asof(ev.sort_values("_d"), rows.sort_values("_d"), on="_d", by="SecuritiesCode",
                      direction="forward", tolerance=pd.Timedelta(days=EVENT_SHIFT_DAYS))
    return m.drop(columns="_d")


def check_schema(df, r):
    print("【1】欄位與鍵值")
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLS]
    if missing:
        r.line("ERROR", f"缺少欄位：{missing}")
        return False
    r.line("OK" if not extra else "WARN", f"欄位齊全{'' if not extra else f'，多出欄位 {extra}'}")

    dates = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    bad_date = df[dates.isna()]
    r.add_rows("日期格式", "ERROR", bad_date, "Date 不是 YYYY-MM-DD")
    weekend = df[(dates.dt.dayofweek >= 5) & ~df["Date"].isin(TW_SATURDAY_TRADING_DAYS)]
    r.add_rows("週末日期", "ERROR", weekend, "週末不應有交易資料（補班交易日除外）")

    expect_rowid = dates.dt.strftime("%Y%m%d") + "_" + df["SecuritiesCode"].astype(str)
    bad_rowid = df[expect_rowid != df["RowId"]]
    r.add_rows("RowId", "ERROR", bad_rowid, "RowId 與 Date_SecuritiesCode 不一致")

    dup = df[df.duplicated(["Date", "SecuritiesCode"], keep=False)]
    r.add_rows("重複鍵", "ERROR", dup, "同一股票同一天出現多筆")

    n = len(bad_date) + len(weekend) + len(bad_rowid) + len(dup)
    r.line("OK" if n == 0 else "ERROR",
           f"日期格式錯誤 {len(bad_date)}、週末 {len(weekend)}、RowId 不符 {len(bad_rowid)}、"
           f"重複 (Date, Code) {len(dup)} 筆")
    return True


def check_ohlc(df, r):
    print("\n【2】OHLC 合理性")
    o, h, l, c = (df[k] for k in PRICE_COLS)
    na_cnt = df[PRICE_COLS].isna().sum(axis=1)
    partial = df[(na_cnt > 0) & (na_cnt < 4)]
    r.add_rows("OHLC 部分缺值", "ERROR", partial, "同一列 OHLC 只有部分是 NA")

    full = na_cnt == 0
    bad = full & (
        (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0) | (h < l)
        | (o < l) | (o > h) | (c < l) | (c > h)
    )
    bad_rows = df[bad]
    detail = (
        "O=" + bad_rows["Open"].round(4).astype(str) + " H=" + bad_rows["High"].round(4).astype(str)
        + " L=" + bad_rows["Low"].round(4).astype(str) + " C=" + bad_rows["Close"].round(4).astype(str)
    )
    r.add_rows("OHLC 邏輯", "ERROR", bad_rows, detail)

    vol = pd.to_numeric(df["Volume"], errors="coerce")
    bad_vol = df[(vol < 0) | (vol.notna() & (vol % 1 != 0))]
    r.add_rows("Volume", "ERROR", bad_vol, "Volume 為負數或非整數")

    all_na = int((na_cnt == 4).sum())
    r.line("OK" if len(bad_rows) == 0 else "ERROR", f"Close/Open 超出 High-Low 或價格 ≤ 0：{len(bad_rows)} 筆")
    r.line("OK" if len(partial) == 0 else "ERROR", f"OHLC 部分缺值：{len(partial)} 筆")
    r.line("OK" if len(bad_vol) == 0 else "ERROR", f"Volume 負數/非整數：{len(bad_vol)} 筆")
    r.line("INFO", f"OHLC 全為 NA（已被清洗或停牌）：{all_na:,} 筆")


def check_tick(df, r):
    print("\n【3】是否為原始報價（台股升降單位）")
    valid = df.dropna(subset=PRICE_COLS)
    p = valid[PRICE_COLS].to_numpy(dtype=float)
    t = tick_size(p)
    q = p / t
    off = np.abs(q - np.round(q)) > 1e-3
    row_off = off.any(axis=1)
    ratio = pd.Series(row_off).groupby(valid["SecuritiesCode"].values).mean()
    adjusted = ratio[ratio > OFF_TICK_STOCK_RATIO].sort_values(ascending=False)

    if len(adjusted):
        first_rows = (
            valid[valid["SecuritiesCode"].isin(adjusted.index) & row_off]
            .groupby("SecuritiesCode").head(1)
        )
        r.add_rows("非原始報價", "ERROR", first_rows,
                   "價格不在台股升降單位上（例如 208.2645），代表是調整後價格，"
                   "再套 AdjustmentFactor 會重複調整")
    # 其餘股票中零星不在升降單位上的列，可能是個別錯誤報價
    stray = valid[row_off & ~valid["SecuritiesCode"].isin(adjusted.index).values]
    r.add_rows("零星非升降單位", "WARN", stray, "這一列的價格不在台股升降單位上，請抽查")
    r.line("OK" if len(adjusted) == 0 else "ERROR",
           f"{len(adjusted)} / {ratio.size} 檔股票有超過 {OFF_TICK_STOCK_RATIO:.0%} 的價格不在升降單位上"
           f"（整體 {row_off.mean():.1%} 的列）")
    if len(adjusted):
        print("    例：" + "、".join(f"{k}({v:.0%})" for k, v in adjusted.head(8).items()))
    r.line("OK" if len(stray) == 0 else "WARN", f"其他股票中零星不在升降單位上的列：{len(stray)} 筆")
    return len(adjusted) == 0


def check_price_limit(df, actions, r):
    print("\n【4】漲跌幅（±10%）")
    d = df[["SecuritiesCode", "Date", "Close"]].copy()
    d["_pos"] = d.groupby("SecuritiesCode").cumcount()

    market_dates = np.sort(d["Date"].unique())
    idx = pd.Series(np.arange(len(market_dates)), index=market_dates)
    d["_di"] = d["Date"].map(idx).values
    d["_prev_close"] = d.groupby("SecuritiesCode")["Close"].shift(1)
    d["_prev_di"] = d.groupby("SecuritiesCode")["_di"].shift(1)
    d["_ret"] = d["Close"] / d["_prev_close"] - 1

    excl = (d["_pos"] < IPO_FREE_DAYS) | ((d["_di"] - d["_prev_di"]) > GAP_DAYS)
    n_event = 0
    if actions is not None:
        ev = pd.concat([
            actions[["SecuritiesCode", "StockExDate"]].rename(columns={"StockExDate": "Date"}),
            actions[["SecuritiesCode", "CashExDate"]].rename(columns={"CashExDate": "Date"}),
        ]).dropna().drop_duplicates()
        ev = ev[(ev["Date"] >= d["Date"].min()) & (ev["Date"] <= d["Date"].max())]
        # 除權息日遇休市時，跳空發生在順延後的第一個交易日
        ev = (align_events(ev, d).dropna(subset=["TradeDate"])[["SecuritiesCode", "TradeDate"]]
              .rename(columns={"TradeDate": "Date"}).drop_duplicates())
        ev["_event"] = True
        d = d.merge(ev, on=["SecuritiesCode", "Date"], how="left")
        is_event = d["_event"].notna()
        n_event = int((is_event & (d["_ret"].abs() > LIMIT + LIMIT_TOL)).sum())
        excl = excl.values | is_event.values

    over = d[(d["_ret"].abs() > LIMIT + LIMIT_TOL) & ~excl]
    detail = ("前收 " + over["_prev_close"].round(4).astype(str) + " → 收 "
              + over["Close"].round(4).astype(str) + "（" + (over["_ret"] * 100).round(2).astype(str) + "%）")
    r.add_rows("漲跌幅超過 10%", "ERROR", over, detail)
    r.line("OK" if len(over) == 0 else "ERROR",
           f"非除權息日、非新上市、非停牌復牌，收盤漲跌超過 ±{LIMIT:.0%}：{len(over)} 筆")
    if actions is None:
        print("    （沒有提供 --actions，除權息日沒被排除，可能有誤報）")
    elif n_event:
        print(f"    另有 {n_event} 筆發生在除權息日（已排除，屬正常）")


def check_factors(df, actions, r):
    print("\n【5】除權事件與 AdjustmentFactor")
    af = pd.to_numeric(df["AdjustmentFactor"], errors="coerce")
    bad_af = df[af.isna() | (af <= 0) | (af > 1)]
    r.add_rows("AdjustmentFactor", "ERROR", bad_af, "因子必須在 (0, 1]")
    non_one = int((af != 1.0).sum())
    r.line("OK" if len(bad_af) == 0 else "ERROR", f"因子值域異常：{len(bad_af)} 筆；因子 ≠ 1 的列：{non_one:,} 筆")

    if actions is None:
        r.line("WARN", "沒有提供 --actions，跳過除權日比對與重複調整檢查")
        return

    lo, hi = df["Date"].min(), df["Date"].max()
    codes = set(df["SecuritiesCode"].unique())
    ev = actions[(actions["StockDividend"] > 0) & actions["StockExDate"].notna()]
    ev = (ev.groupby(["SecuritiesCode", "StockExDate"], as_index=False)["StockDividend"].sum()
            .rename(columns={"StockExDate": "Date"}))
    ev = ev[(ev["Date"] >= lo) & (ev["Date"] <= hi) & ev["SecuritiesCode"].isin(codes)]
    # 除權日遇休市（颱風假等）時，因子應出現在該股下一筆報價；同一天的事件股利加總
    ev = align_events(ev, df)
    no_row = ev[ev["TradeDate"].isna()]
    r.add_rows("除權日無報價", "WARN", no_row,
               f"除權日起 {EVENT_SHIFT_DAYS} 天內找不到該股票的報價，因子無法套用")
    shifted = int((ev["TradeDate"].notna() & (ev["TradeDate"] != ev["Date"])).sum())
    ev = (ev.dropna(subset=["TradeDate"])
            .groupby(["SecuritiesCode", "TradeDate"], as_index=False)["StockDividend"].sum()
            .rename(columns={"TradeDate": "Date"}))
    ev["expected_factor"] = 1.0 / (1.0 + ev["StockDividend"] / 10.0)

    d = df[["SecuritiesCode", "Date", "Close", "AdjustmentFactor"]].sort_values(["SecuritiesCode", "Date"]).copy()
    d["_prev_close"] = d.groupby("SecuritiesCode")["Close"].shift(1)
    hit = ev.merge(d, on=["SecuritiesCode", "Date"], how="inner")
    wrong_f = hit[(hit["AdjustmentFactor"] - hit["expected_factor"]).abs() > 1e-9]
    r.add_rows("因子不符", "ERROR", wrong_f,
               "除權日因子 " + wrong_f["AdjustmentFactor"].round(6).astype(str)
               + "，應為 " + wrong_f["expected_factor"].round(6).astype(str))

    # 重複調整：原始報價在除權日的參考價是「前收 × 因子」，漲跌幅限制以參考價計算。
    # 若收盤相對參考價超出 ±10%（原始報價不可能），但相對前收在 ±10% 內，代表來源已經調整過。
    # （只看「比較接近 1 還是因子」會把除權當天上漲的股票誤判，所以改用漲跌幅限制判斷）
    big = hit[(hit["expected_factor"] < 0.97) & hit["_prev_close"].notna() & hit["Close"].notna()].copy()
    vs_ref = big["Close"] / (big["_prev_close"] * big["expected_factor"]) - 1
    vs_prev = big["Close"] / big["_prev_close"] - 1
    already = (vs_ref.abs() > LIMIT + LIMIT_TOL) & (vs_prev.abs() <= LIMIT + LIMIT_TOL)
    dbl = big[already.values]
    r.add_rows("重複調整", "ERROR", dbl,
               "除權日價格沒有跳空（前收 " + dbl["_prev_close"].round(4).astype(str) + " → 收 "
               + dbl["Close"].round(4).astype(str) + "），來源已調整過，再乘因子 "
               + dbl["expected_factor"].round(4).astype(str) + " 會重複調整")

    r.line("OK" if len(no_row) == 0 else "WARN",
           f"區間內除權事件對上報價 {len(hit)} 筆（其中 {shifted} 筆因休市順延），找不到報價 {len(no_row)} 筆")
    r.line("OK" if len(wrong_f) == 0 else "ERROR", f"除權日因子與 1/(1+配股/10) 不符：{len(wrong_f)} 筆")
    r.line("OK" if len(dbl) == 0 else "ERROR",
           f"可判斷的除權事件 {len(big)} 筆中，價格已被來源調整（會重複調整）：{len(dbl)} 筆")
    if len(ev) and non_one == 0:
        r.line("ERROR", "有除權事件，但檔案裡的 AdjustmentFactor 全部是 1.0")


def check_target(df, r):
    print("\n【6】Target 獨立重算")
    d = df[["SecuritiesCode", "Date", "Close", "AdjustmentFactor", "Target"]].sort_values(
        ["SecuritiesCode", "Date"]).reset_index(drop=True)
    g = d.groupby("SecuritiesCode", sort=False)
    c1 = g["Close"].shift(-1)
    c2 = g["Close"].shift(-2)
    f2 = g["AdjustmentFactor"].shift(-2)
    # 另一條公式：Target(t) = Close(t+2) / (Close(t+1) × Factor(t+2)) − 1
    recalc = c2 / (c1 * f2) - 1
    recalc = recalc.where((c1 > 0) & (c2 > 0))
    stored = pd.to_numeric(d["Target"], errors="coerce")

    na_diff = d[stored.isna() != recalc.isna()]
    r.add_rows("Target NA 位置", "ERROR", na_diff,
               "檔案 Target=" + stored[na_diff.index].astype(str) + "，重算=" + recalc[na_diff.index].astype(str))

    both = stored.notna() & recalc.notna()
    diff = (stored - recalc).abs()
    bad = d[both & (diff > TARGET_TOL)]
    r.add_rows("Target 數值", "ERROR", bad,
               "檔案 " + (stored[bad.index] * 100).round(6).astype(str) + "%，重算 "
               + (recalc[bad.index] * 100).round(6).astype(str) + "%")

    max_diff = float(diff[both].max()) if both.any() else 0.0
    r.line("OK" if len(bad) == 0 and len(na_diff) == 0 else "ERROR",
           f"數值不一致 {len(bad):,} 筆、NA 位置不一致 {len(na_diff):,} 筆（最大誤差 {max_diff:.2e}）")
    ext = int((stored.abs() > 0.105).sum())
    r.line("INFO", f"|Target| > 10.5% 的列：{ext:,} 筆（理論上只會出現在除權息/新上市/復牌附近）")


def check_calendar(df, r):
    print("\n【7】交易日曆與缺口")
    per_day = df.groupby("Date")["SecuritiesCode"].nunique()
    med = per_day.rolling(21, center=True, min_periods=5).median()
    thin = per_day[per_day < 0.5 * med]
    if len(thin):
        r.add_rows("稀疏交易日", "WARN", thin.reset_index().assign(SecuritiesCode=""),
                   "當天股票數不到附近交易日中位數的一半，可能是非交易日或下載缺漏")
    r.line("OK" if len(thin) == 0 else "WARN", f"股票數異常少的日期：{len(thin)} 天")

    dates = np.sort(df["Date"].unique())
    idx = pd.Series(np.arange(len(dates)), index=dates)
    span = df.groupby("SecuritiesCode")["Date"].agg(first="min", last="max", n="count")
    span["expected"] = span["last"].map(idx) - span["first"].map(idx) + 1
    span["missing"] = span["expected"] - span["n"]
    gapped = span[span["missing"] > 0]
    r.line("OK" if len(gapped) == 0 else "INFO",
           f"{len(gapped)} 檔在上市期間內有缺日（停牌、減資屬正常），最多缺 {int(span['missing'].max())} 天")


def check_constant_cols(df, r):
    print("\n【8】預留欄位")
    for col, note in [("ExpectedDividend", "應為 NA（不可填 0）"), ("SupervisionFlag", "應為 NA（不可填 False）")]:
        s = df[col]
        zeros = int((s.astype(str).isin(["0", "0.0", "False", "false"])).sum())
        r.line("OK" if zeros == 0 else "ERROR", f"{col}：NA {s.isna().mean():.1%}，被填成 0/False {zeros} 筆（{note}）")


def check_coverage(df, stock_list_path, r):
    if not stock_list_path or not os.path.exists(stock_list_path):
        return
    print("\n【9】股票清單覆蓋")
    sl = pd.read_csv(stock_list_path, dtype={"stock_id": str}, encoding="utf-8-sig")
    want = set(sl["stock_id"])
    have = set(df["SecuritiesCode"].unique())
    missing = sorted(want - have)
    extra = sorted(have - want)
    r.line("OK" if not missing else "WARN",
           f"清單 {len(want)} 檔，報價有 {len(have & want)} 檔；清單中沒有報價 {len(missing)} 檔，"
           f"報價有但清單沒有 {len(extra)} 檔")
    if missing:
        print("    例：" + "、".join(missing[:10]))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="檢查台股 JPX 格式報價資料")
    ap.add_argument("path", help="stock_price_final.csv，或放單檔 CSV 的資料夾")
    ap.add_argument("--actions", default="corporate_actions.csv", help="公司行動檔（預設 corporate_actions.csv）")
    ap.add_argument("--stock-list", default="stock_list.csv", help="股票清單（預設 stock_list.csv）")
    ap.add_argument("--out", default="check_issues.csv", help="問題明細輸出檔")
    args = ap.parse_args()

    df = load_prices(args.path)
    df["SecuritiesCode"] = df["SecuritiesCode"].astype(str)
    for c in PRICE_COLS + ["AdjustmentFactor", "Target"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    actions = load_actions(args.actions)

    r = Report()
    if not check_schema(df, r):
        sys.exit(1)
    check_ohlc(df, r)
    check_tick(df, r)
    check_price_limit(df, actions, r)
    check_factors(df, actions, r)
    check_target(df, r)
    check_calendar(df, r)
    check_constant_cols(df, r)
    check_coverage(df, args.stock_list, r)

    print("\n" + "=" * 60)
    if r.issues:
        issues = pd.concat(r.issues, ignore_index=True)
        issues.to_csv(args.out, index=False, encoding="utf-8-sig")
        counts = issues.groupby(["level", "check"]).size().sort_index()
        print(f"問題明細已寫入 {args.out}（共 {len(issues):,} 列）")
        print(counts.to_string())
    print("=" * 60)
    print("❌ 未通過：請修正後再訓練" if r.has_error else "✅ 全部通過")
    sys.exit(1 if r.has_error else 0)


if __name__ == "__main__":
    main()

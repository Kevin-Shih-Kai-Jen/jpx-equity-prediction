"""TOPIX 日資料檢查器（Python 3.10+；依賴 pandas、numpy）。

    from topix_validator import validate_topix
    validate_topix(topix)  # 合法回傳 True；不合法拋出詳細 ValueError
    valid = validate_topix(topix, raise_on_error=False)  # 只要 True / False

沿用自己的 check_table：
    validate_topix(topix, check_table_func=check_table)

支援 DataFrame、CSV 路徑，以及 Date 放在 DatetimeIndex 的資料。
不修改原表、不排序、不補值、不刪除資料。日期字串僅接受 YYYY-MM-DD。
預設檢查 2017-01-01 至 2021-12-31；多抓暖機資料時自行調整 start。
Volume 若存在便檢查，零可接受。其他額外欄位不檢查。

注意：通過代表符合指定規則，不代表已核實指數來源、價格真實性或
交易日完整性。若要檢查缺漏／多餘日期，傳入官方 expected_dates；
不可用一般工作日 pd.bdate_range 取代日本交易日曆。
"""

from __future__ import annotations

from datetime import date, datetime
from os import PathLike
import re
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


Requirements = dict[str, tuple[list[type], list[Callable[[Any], bool]]]]
CheckResults = dict[str, list[tuple[int, bool, TypeError | ValueError | None]]]
TableChecker = Callable[[pd.DataFrame, Requirements], CheckResults]
NUMERIC_TYPES = [int, float, np.integer, np.floating]
PRICE_COLUMNS = ("Open", "High", "Low", "Close")


class TopixValidationError(ValueError):
    """資料不符合規則；errors 保存完整錯誤，訊息最多顯示前 20 個。"""

    def __init__(self, errors: list[str]):
        self.errors = errors
        details = "\n".join(f"- {error}" for error in errors[:20])
        if len(errors) > 20:
            details += f"\n- 另有 {len(errors) - 20} 個問題，見 exception.errors。"
        super().__init__(f"TOPIX 資料檢查失敗（{len(errors)} 個問題）：\n{details}")


def _finite_number(value: Any) -> bool:
    if not isinstance(value, tuple(NUMERIC_TYPES)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _positive(value: Any) -> bool:
    return _finite_number(value) and value > 0


def _volume(value: Any) -> bool:
    return _finite_number(value) and value >= 0 and value == int(value)


def _daily_date(value: Any) -> pd.Timestamp:
    """明確解析日日期，避免整數被默認解讀為 Unix nanoseconds。"""
    if isinstance(value, str):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
            raise ValueError("日期字串必須為 YYYY-MM-DD")
    elif not isinstance(value, (date, datetime, pd.Timestamp, np.datetime64)):
        raise ValueError("日期必須為 YYYY-MM-DD 字串或日期物件")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("日期不能缺失")
    if stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ValueError("日資料不能帶時區或非零時間")
    return stamp


def build_topix_requirements(
    start: str = "2017-01-01",
    end: str = "2021-12-31",
    *,
    include_volume: bool = False,
) -> Requirements:
    """可直接傳入使用者原有 check_table 的單值規則。Date 須先解析。"""
    first, last = _daily_date(start), _daily_date(end)
    if first > last:
        raise ValueError("start 不得晚於 end")

    def date_in_range(value: Any) -> bool:
        try:
            return first <= _daily_date(value) <= last
        except (TypeError, ValueError, OverflowError):
            return False

    requirements: Requirements = {
        "Date": ([pd.Timestamp, datetime], [date_in_range]),
        **{col: (NUMERIC_TYPES.copy(), [_positive]) for col in PRICE_COLUMNS},
    }
    if include_volume:
        requirements["Volume"] = (NUMERIC_TYPES.copy(), [_volume])
    return requirements


TOPIX_REQUIREMENTS = build_topix_requirements()


def check_series(
    acceptable_types: list[type],
    series: pd.Series,
    rules: list[Callable[[Any], bool]],
) -> list[tuple[int, bool, TypeError | ValueError | None]]:
    """內建後備版本；位置從 0 開始，True 表示通過。"""
    results = []
    for position, value in enumerate(series):
        error: TypeError | ValueError | None = None
        if not isinstance(value, tuple(acceptable_types)):
            error = TypeError(f"不接受型別 {type(value).__name__}：{value!r}")
        else:
            for rule in rules:
                try:
                    passed = bool(rule(value))
                except (TypeError, ValueError, OverflowError) as exc:
                    error = ValueError(f"規則 {rule.__name__} 失敗：{exc}")
                    break
                if not passed:
                    error = ValueError(f"不符合 {rule.__name__}：{value!r}")
                    break
        results.append((position, error is None, error))
    return results


def check_table(table: pd.DataFrame, series_requirements: Requirements) -> CheckResults:
    """與使用者提供的 check_table 相同介面，讓檔案可獨立使用。"""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("Expected a DataFrame")
    if table.empty:
        raise ValueError("Table is empty")
    results = {}
    for col_name, (acceptable_types, rules) in series_requirements.items():
        if col_name not in table.columns:
            raise ValueError(f"Missing required column: {col_name}")
        results[col_name] = check_series(acceptable_types, table[col_name], rules)
    return results


def validate_topix(
    table: pd.DataFrame | str | PathLike,
    *,
    start: str = "2017-01-01",
    end: str = "2021-12-31",
    raise_on_error: bool = True,
    check_table_func: TableChecker | None = None,
    expected_dates: Iterable[Any] | None = None,
) -> bool:
    """一次完成單值、日期與 OHLC 關係檢查。

    合法回傳 True。不合法預設拋出 TopixValidationError（ValueError 子類）；
    raise_on_error=False 時回傳 False。檔案讀取／參數錯誤仍會拋出例外。

    check_table_func：可傳入自己的 check_table。回傳格式須與本檔相同，
    每個欄位每列有 (position, passed, error)，passed=True 代表合法。
    傳給 checker 的副本使用從 0 開始的 RangeIndex，方便定位。

    expected_dates：可選的完整交易日期序列；會截取 start/end 期間比較。
    未提供時不聲稱資料已涵蓋完整期間。此函式不檢查 MarketReturn。
    """
    requirements = build_topix_requirements(start, end)
    first, last = _daily_date(start), _daily_date(end)
    errors: list[str] = []

    def finish() -> bool:
        if errors and raise_on_error:
            raise TopixValidationError(errors)
        return not errors

    if isinstance(table, (str, PathLike)):
        table = pd.read_csv(table)
    if not isinstance(table, pd.DataFrame):
        errors.append("輸入必須為 DataFrame 或 CSV 路徑")
        return finish()
    if table.empty:
        errors.append("資料表為空")
        return finish()
    if not table.columns.is_unique:
        errors.append("欄位名稱重複")
        return finish()

    data = table.copy(deep=True)
    if "Date" not in data.columns and isinstance(data.index, pd.DatetimeIndex):
        data.insert(0, "Date", data.index)
    data = data.reset_index(drop=True)
    missing = [col for col in ("Date", *PRICE_COLUMNS) if col not in data.columns]
    if missing:
        errors.append(f"缺少必要欄位：{missing}")
        return finish()

    parsed_dates = []
    for position, value in enumerate(data["Date"]):
        try:
            parsed_dates.append(_daily_date(value))
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f"Date，第 {position + 1} 筆資料：{value!r}；{exc}")
    if errors:
        return finish()
    data["Date"] = pd.Series(parsed_dates)

    if "Volume" in data.columns:
        requirements["Volume"] = (NUMERIC_TYPES.copy(), [_volume])
    checker = check_table if check_table_func is None else check_table_func
    results = checker(data, requirements)
    # 不靜默接受不相容／缺漏的 checker 結果。
    for column in requirements:
        if column not in results or len(results[column]) != len(data):
            raise TypeError(f"check_table_func 未回傳 {column} 的逐列結果")
        for position, passed, error in results[column]:
            if not passed:
                errors.append(f"{column}，位置 {position}（從 0 起）：{error}")
    if errors:
        return finish()

    dates = data["Date"]
    duplicate_dates = dates[dates.duplicated(keep=False)]
    if not duplicate_dates.empty:
        examples = duplicate_dates.dt.strftime("%Y-%m-%d").unique()[:10].tolist()
        errors.append(f"日期重複：{examples}")
    if not dates.is_monotonic_increasing:
        errors.append("日期未由舊到新排列；計算報酬前請 sort_values('Date')")

    bad_ohlc = (
        (data["Low"] > data["High"])
        | (data["Open"] < data["Low"])
        | (data["Open"] > data["High"])
        | (data["Close"] < data["Low"])
        | (data["Close"] > data["High"])
    )
    for position in data.index[bad_ohlc]:
        values = data.loc[position, list(PRICE_COLUMNS)].to_dict()
        errors.append(f"第 {position + 1} 筆，{dates.iloc[position].date()}：OHLC 關係錯誤 {values}")

    if expected_dates is not None:
        if isinstance(expected_dates, (str, bytes)):
            raise TypeError("expected_dates 必須是日期序列，不是單一字串")
        expected = pd.DatetimeIndex([_daily_date(d) for d in expected_dates])
        expected = expected[(expected >= first) & (expected <= last)].unique()
        if expected.empty:
            raise ValueError("expected_dates 在指定期間內沒有日期")
        actual = pd.DatetimeIndex(dates)
        for label, difference in (
            ("缺少交易日", expected.difference(actual)),
            ("多出非預期日期", actual.difference(expected)),
        ):
            if len(difference):
                errors.append(f"{label} {len(difference)} 日，例：{difference[:10].strftime('%Y-%m-%d').tolist()}")
    return finish()

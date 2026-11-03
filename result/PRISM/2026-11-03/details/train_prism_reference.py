"""單檔版本：直接放到 JPX/codes/module/train_prism.py 執行。

不需要安裝前面的程式包，也不匯入其他 models.py 或 training.py。
模型、序列整理、訓練、驗證與績效計算都放在這一個檔案。
預設讀取原專案 config.COL_NAMES 與 data/stock_prices.csv。
COL_NAMES 若含 Target、日期或股票識別欄位，會自動排除，不送入模型。
不覆寫任何既有程式或 CSV；結果另存於 JPX/runs/prism_single/。

閱讀順序：最上方「設定」→ main() → run_experiment()。
下面的模型與輔助函式可以先在編輯器裡摺疊，需要理解時再展開。
預設是正式訓練；先試跑可把 EPOCHS 改成 1、MAX_SPLITS 改成 1。

已通過 CPU 合成資料測試，包含直接執行這個單檔的兩階段訓練。
尚未執行你的完整 JPX CSV，或在 Mac/MPS 上實測。
"""

from copy import deepcopy
from datetime import datetime, timedelta
from numbers import Integral
from pathlib import Path
import json
import math
import sys

try:
    import numpy as np
    import pandas as pd
    import scipy
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    import torch.nn.functional as F
except ModuleNotFoundError as error:
    raise SystemExit(
        f"缺少 Python 套件：{error.name}\n"
        f'請用同一個 Python 安裝："{sys.executable}" -m pip install '
        "torch numpy pandas scipy"
    ) from error


# ======================================================================
# 1. 設定：平常需要調整的項目集中在這裡
# ======================================================================

# 檔案位於 JPX/codes/module/，向上兩層就是 JPX。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "stock_prices.csv"

# None = 讀取 codes/config.py 的 COL_NAMES，再排除標籤與識別欄位。
# 只處理模型輸入清單，不更動 config.py 或原始資料。
# 若希望連 config 都不用，可在這裡明確填入你要用的欄位清單。
# 例如：["Open", "High", "Low", "Close", "Volume"]
FEATURE_COLUMNS = None

LOOKBACKS = [42]  # 每檔股票最近幾筆歷史觀測
EPOCHS = 50  # Stage 1、Stage 2 各自最多訓練幾輪
PATIENCE = 10  # 驗證 loss 連續幾輪未足夠改善就停止
MIN_CHANGE = 1e-5  # 重設 patience 所需的改善幅度
LEARNING_RATE = 1e-4
VALIDATION_FRACTION = 0.2  # 訓練期間尾端約 20% 的日期用來選 epoch

START_DATE = "2017-01-01"
END_DATE = "2021-07-01"  # 不包含這一天
SPLIT_DAYS = 180  # 沿用原 expanding-window 的日曆天數間隔
MAX_SPLITS = None  # None 跑全部；設 1 只跑第一個視窗
TARGET_HORIZON = 2  # 標籤結束日：同一股票往後兩筆觀測

N_LONG = 200  # 每個回測日期買入幾檔股票
N_SHORT = 200  # 每個回測日期放空幾檔股票
DEVICE = "auto"  # auto = 有 CUDA 就用 CUDA，否則 CPU
SEED = 42

# 每次建立有時間戳記的新資料夾；不覆蓋先前結果。
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "prism_single"


# ======================================================================
# 2. 主流程：直接執行本檔時，從這個 main() 開始
# ======================================================================


def main():
    """讀資料 → 檢查日期窗口 → 訓練兩階段 → 存預測與績效。"""
    # 1. 確認設定，讀入你選定的特徵與原始資料。
    if EPOCHS < 1 or PATIENCE < 1 or MIN_CHANGE < 0:
        raise ValueError("EPOCHS / PATIENCE 必須為正，MIN_CHANGE 不可為負")
    if not np.isfinite(LEARNING_RATE) or LEARNING_RATE <= 0:
        raise ValueError("LEARNING_RATE 必須是正的有限數值")
    if not LOOKBACKS:
        raise ValueError("LOOKBACKS 不可以是空清單")

    columns = get_feature_columns()
    device = choose_device(DEVICE)
    print(f"專案位置：{PROJECT_ROOT}", flush=True)
    print(f"運算裝置：{device}", flush=True)
    data = load_stock_data(columns)

    # 2. 使用 [起日, 迄日)，相鄰視窗不重複計算邊界那一天。
    splits = create_expanding_splits(START_DATE, END_DATE, SPLIT_DAYS)
    if MAX_SPLITS is not None:
        if not isinstance(MAX_SPLITS, int) or MAX_SPLITS < 1:
            raise ValueError("MAX_SPLITS 必須是正整數或 None")
        splits = splits[:MAX_SPLITS]

    # 3. 先檢查全部窗口，避免訓練很久後才發現資料不足。
    report = preflight_data(
        data,
        columns,
        LOOKBACKS,
        splits,
        validation_fraction=VALIDATION_FRACTION,
        target_horizon=TARGET_HORIZON,
        n_long=N_LONG,
        n_short=N_SHORT,
        end_inclusive=False,
    )

    # 4. 結果另外存放，不更動你的原始檔案。
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = OUTPUT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)
    report.to_csv(output / "preflight.csv", index=False)
    print(f"結果資料夾：{output}", flush=True)

    # 5. 每個視窗都先訓練 Stage 1，再固定它的編碼器訓練 Stage 2。
    summary = run_experiment(
        data,
        columns,
        LOOKBACKS,
        splits,
        device=device,
        epochs=EPOCHS,
        patience=PATIENCE,
        min_change=MIN_CHANGE,
        validation_fraction=VALIDATION_FRACTION,
        learning_rate=LEARNING_RATE,
        target_horizon=TARGET_HORIZON,
        checkpoint_dir=output,
        n_long=N_LONG,
        n_short=N_SHORT,
        seed=SEED,
        end_inclusive=False,
    )
    print(f"訓練與回測完成：{output}", flush=True)
    return summary


def get_feature_columns() -> list[str]:
    """從設定的欄位清單取出模型特徵，保留原本的欄位順序。

    COL_NAMES 可以同時包含特徵、Target 與資料識別欄位。
    Target 仍由 dataset 作為標籤；Date、SecuritiesCode 仍用於分組與對齊。
    這些欄位只從輸入特徵中排除，不從原始資料刪掉。
    """
    if FEATURE_COLUMNS is None:
        config_path = PROJECT_ROOT / "codes" / "config.py"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"找不到 {config_path}。"
                "也可以直接在本檔上方填寫 FEATURE_COLUMNS。"
            )

        # 讓直接執行 module 裡的檔案，也能找到 JPX/codes/config.py。
        sys.path.insert(0, str(PROJECT_ROOT))
        from codes.config import COL_NAMES

        selected = COL_NAMES
    else:
        selected = FEATURE_COLUMNS

    if isinstance(selected, (str, set)):
        raise TypeError("特徵請使用有固定順序的 list / tuple")
    columns = list(selected)
    if not columns or any(not isinstance(name, str) for name in columns):
        raise ValueError("特徵必須是非空的欄位名稱清單")
    if len(set(columns)) != len(columns):
        raise ValueError("特徵欄位不可重複")
    # 分開特徵與其他用途的欄位，避免把真實答案 Target 當作輸入。
    metadata_columns = {"Target", "Date", "RowId", "SecuritiesCode"}
    excluded = [name for name in columns if name in metadata_columns]
    columns = [name for name in columns if name not in metadata_columns]
    if excluded:
        print(f"已從模型輸入排除標籤／識別欄位：{excluded}", flush=True)
    if not columns:
        raise ValueError(
            "排除標籤與識別欄位後，沒有可用特徵。"
            "請在本檔 FEATURE_COLUMNS 設定實際的特徵欄位。"
        )
    return columns


def load_stock_data(
    columns,
) -> pd.DataFrame:
    """直接讀 CSV，沿用原本的價格調整方式，不匯入舊資料處理模組。"""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"找不到股票資料：{DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    required = {
        "Date",
        "SecuritiesCode",
        "Target",
        "AdjustmentFactor",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        *columns,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"CSV 缺少這些欄位：{missing}")
    if data.empty:
        raise ValueError("CSV 沒有資料")

    if "ExpectedDividend" in data:
        data["ExpectedDividend"] = data["ExpectedDividend"].fillna(0)
    data = adjust_prices(data)
    for column in [*columns, "Target"]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    print(f"讀入 {len(data):,} 列，輸入特徵數 = {len(columns)}", flush=True)
    print(f"特徵順序：{columns}", flush=True)
    return data


def adjust_prices(
    data,
) -> pd.DataFrame:
    """沿用你原函式的 prior-factor 價格調整，不改 Target。

    每檔股票以「到前一筆為止」的累積 AdjustmentFactor 調整。
    OHLC 除以 prior_factor；Volume 乘以 prior_factor。
    這不是股利再投資報酬，ExpectedDividend 不參與此價格換算。
    """
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    data = data.sort_values(["SecuritiesCode", "Date"])

    # 原函式對無效 factor 使用 1；這裡保留相同型別與有限值規則。
    def clean_factor(value):
        valid_types = (int, float, np.float32, np.float64)
        if type(value) not in valid_types or pd.isna(value):
            return 1.0
        return value if np.isfinite(value) else 1.0

    data["AdjustmentFactor"] = data["AdjustmentFactor"].map(clean_factor)
    if (data["AdjustmentFactor"] <= 0).any():
        raise ValueError("AdjustmentFactor 必須大於 0")

    prior_factor = data.groupby("SecuritiesCode")[
        "AdjustmentFactor"
    ].transform(lambda values: values.cumprod().shift(1, fill_value=1.0))
    if not np.isfinite(prior_factor).all() or (prior_factor <= 0).any():
        raise ValueError("累積 AdjustmentFactor 必須是正的有限數值")
    for column in ["Open", "High", "Low", "Close"]:
        data[column] = data[column] / prior_factor
    data["Volume"] = data["Volume"] * prior_factor
    return data


def create_expanding_splits(
    start_date,
    end_date,
    split_days,
) -> list[tuple]:
    """沿用原 utils 的 expanding-window 規則，日期是 [start, end)。

    第一輪訓練 split_days 天，接著回測 split_days 天。
    下一輪把上一段回測期間納入歷史訓練範圍，並重新訓練模型。
    最後不足一個完整回測區間時停止，與原 utils 相同。
    """
    if not isinstance(split_days, int) or split_days < 1:
        raise ValueError("SPLIT_DAYS 必須是正整數")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start >= end:
        raise ValueError("START_DATE 必須早於 END_DATE")

    interval = timedelta(days=split_days)
    train_end = start + interval
    splits = []
    while train_end + interval <= end:
        test_end = train_end + interval
        splits.append((start, train_end, train_end, test_end))
        train_end = test_end
    return normalize_splits(splits, end_inclusive=False)


def run_experiment(
    df,
    col_names,
    time_spans,
    splits,
    *,
    device="cpu",
    epochs=50,
    patience=10,
    min_change=1e-5,
    validation_fraction=0.2,
    learning_rate=1e-4,
    target_horizon=2,
    checkpoint_dir="checkpoints/prism_vq",
    n_long=200,
    n_short=200,
    seed=42,
    end_inclusive=True,
):
    """執行多個 lookback / expanding windows，儲存權重與回測結果。

    主要輸入：
        df          : 已整理的股票表格，包含 Date、SecuritiesCode、Target。
        col_names   : 特徵欄位；順序會直接決定輸入 Tensor 的特徵順序。
        time_spans  : lookback 筆數，例如 [5, 20, 42]，或單一整數。
        splits      : 各輪 (train_start, train_end, test_start, test_end)。
        target_horizon : Target 完整得知需要往後幾筆股票觀測；預設 2。
        checkpoint_dir : 本次實驗的輸出資料夾。
        n_long / n_short : 每個回測日期買入 / 放空的股票數量。

    本版本每個外層視窗都建立新模型；expanding window 本身不強制重訓。
    Stage 1 與 Stage 2 各自只用內層驗證集選最佳 epoch。
    回傳一個 DataFrame：每個 lookback / split 一列的績效摘要。
    """

    # 1. 統一參數格式，並檢查外層回測區間沒有重疊。
    # 直接呼叫時保留包含迄日的介面；專案 main 會明確傳 False。
    time_spans = (
        [int(time_spans)]
        if isinstance(time_spans, (int, np.integer))
        else list(time_spans)
    )
    splits = normalize_splits(splits, end_inclusive=end_inclusive)
    if not splits or not time_spans:
        raise ValueError("Need splits and lookbacks")

    # 2. 準備輸出資料夾與共用的 prediction loss。
    # MSELoss 預設回傳平均值；模型的 auxiliary loss 在 本檔的訓練函式 中另加。
    output = Path(checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    loss_fn, summaries = nn.MSELoss(), []

    # 3. 每個 lookback 建立自己的歷史窗口；例如 42 表示最近 42 筆觀測。
    for lookback in time_spans:
        dataset = StockSequenceDataset(
            df, col_names, lookback, target_horizon
        )
        test_frames = []

        # 4. 依時間順序跑外層視窗。每個視窗重新建立 Stage 1、Stage 2。
        for split_index, split in enumerate(splits, start=1):
            torch.manual_seed(seed + int(lookback) * 1000 + split_index)

            # fit 更新權重；valid 決定何時停止；test 只用來回測。
            fit_loader, valid_loader, test_loader, validation_start = (
                split_daily_loaders(
                    dataset,
                    split,
                    validation_fraction,
                    end_inclusive=end_inclusive,
                )
            )
            validate_daily_loaders(
                fit_loader,
                valid_loader,
                test_loader,
                n_long=n_long,
                n_short=n_short,
            )
            tag = f"lookback_{lookback}_split_{split_index:02d}"

            # 記住特徵順序與日期，未來才能用相同設定重建模型輸入。
            metadata = dict(
                feature_columns=list(col_names),
                num_features=len(col_names),
                lookback=int(lookback),
                split=[str(v.date()) for v in split],
                validation_start=str(validation_start.date()),
                target_horizon=target_horizon,
                seed=seed,
                window_mode="last_observed_rows_per_stock",
                experiment_mode="fresh_models_per_split",
                end_inclusive=end_inclusive,
            )
            (output / f"{tag}.config.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"\n{tag}: inner validation from {validation_start.date()}, "
                f"outer test from {split[2].date()}",
                flush=True,
            )

            # 5. 先訓練 Stage 1，讓編碼器與 codebook 學習。
            # 第二個參數 0 對應保留的 prior_factor_dim；目前模型沒有使用它。
            model_1 = Stage_1(len(col_names), 0).to(device)
            optimizer_1 = torch.optim.AdamW(
                model_1.parameters(), lr=learning_rate
            )
            result_1 = train(
                model_1,
                fit_loader,
                loss_fn,
                optimizer_1,
                epochs=epochs,
                device=device,
                patience=patience,
                min_change=min_change,
                validation_loader=valid_loader,
                checkpoint_path=output / f"{tag}.stage1.pt",
                checkpoint_metadata=metadata,
            )

            # 6. 用 Stage 1 的最佳權重建立 Stage 2。
            # Stage_2 內部會凍結 RevIN、SpatialEncoder、VectorQuantizer。
            model_2 = Stage_2(model_1, num_features=len(col_names)).to(device)

            # 只把 requires_grad=True 的參數交給 Stage 2 optimizer。
            optimizer_2 = torch.optim.AdamW(
                [p for p in model_2.parameters() if p.requires_grad],
                lr=learning_rate,
            )
            result_2 = train(
                model_2,
                fit_loader,
                loss_fn,
                optimizer_2,
                epochs=epochs,
                device=device,
                patience=patience,
                min_change=min_change,
                validation_loader=valid_loader,
                checkpoint_path=output / f"{tag}.stage2.pt",
                checkpoint_metadata=metadata,
            )

            # 7. 訓練與選 epoch 都結束後，才評估這個視窗的 outer test。
            test_loss, test_aux, predictions = evaluation(
                model_2, test_loader, loss_fn, device
            )
            predictions["Split"], predictions["Lookback"] = (
                split_index,
                lookback,
            )

            # 預測表已帶日期與股票代碼，績效計算直接使用這份對齊後的表。
            predictions.to_csv(output / f"{tag}.predictions.csv", index=False)
            metrics = performance_from_predictions(
                predictions, n_long=n_long, n_short=n_short
            )
            summaries.append(
                dict(
                    lookback=int(lookback),
                    split=split_index,
                    stage1_best_epoch=result_1["best_epoch"],
                    stage2_best_epoch=result_2["best_epoch"],
                    test_prediction_loss=test_loss,
                    test_model_loss=test_aux,
                    **metrics,
                )
            )
            pd.DataFrame(summaries).to_csv(
                output / "split_summary.csv", index=False
            )
            test_frames.append(predictions)

        # 8. 同一 lookback 的所有視窗跑完後，合併各段未來期間預測。
        combined = pd.concat(test_frames, ignore_index=True)
        if combined.duplicated(["Date", "SecuritiesCode"]).any():
            raise ValueError("Repeated stock/date across outer tests")
        combined.to_csv(
            output / f"lookback_{lookback}.all_predictions.csv", index=False
        )
        metrics = performance_from_predictions(
            combined, n_long=n_long, n_short=n_short
        )
        # 重新用所有日期計算 Sharpe，不能直接平均各視窗的 Sharpe。

        # JSON 用 null 表示未定義的績效值，例如日報酬恆定時的 Sharpe。
        safe = {
            k: float(v) if np.isfinite(v) else None
            for k, v in metrics.items()
        }
        (output / f"lookback_{lookback}.metrics.json").write_text(
            json.dumps(safe, indent=2), encoding="utf-8"
        )
    return pd.DataFrame(summaries)


# ======================================================================
# 3. 模型：Stage 1、Stage 2 及其內部組件
# ======================================================================


class RevIN(nn.Module):
    """各股票、各特徵沿時間軸標準化，並保留可學習的縮放與平移。"""

    def __init__(
        self,
        num_features: int,
        small: float = 1e-5,
    ):
        """num_features 是特徵數；a / intercept 各有一個參數對應一個特徵。"""
        super().__init__()
        self.variance = None
        self.mean = None
        self.small = small
        self.a = nn.Parameter(torch.ones(num_features))
        self.intercept = nn.Parameter(torch.zeros(num_features))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """等同呼叫 normalize；輸入與輸出都是 (N, T, C)。"""
        return self.normalize(x)

    def normalize(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """用「本次輸入」的時間平均與變異數進行標準化。

        mean / variance: (N, 1, C)，會自動廣播到 (N, T, C)。
        a / intercept: (C,)，也會廣播到所有股票與時間位置。
        """
        self.mean, self.variance = self._get_paramans(x)

        return (
            self.a * ((x - self.mean)) / ((self.variance + self.small) ** 0.5)
            + self.intercept
        )

    def denormalize(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """用上一次 normalize 的統計量還原相同特徵空間的資料。

        必須先 normalize；新一批 normalize 會覆蓋舊統計量。
        目前兩階段的單一股票報酬預測不呼叫這個函式。
        """
        if self.mean is None or self.variance is None:
            raise ValueError(
                "Data should be normalized first in order to get denormalized"
            )

        return (x - self.intercept) / self.a * (
            self.variance + self.small
        ) ** 0.5 + self.mean

    def _get_paramans(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """沿時間軸 dim=1 計算每檔股票、每項特徵的統計量。

        keepdim=True 保留時間維度 1，使結果可以與原輸入廣播。
        detach 讓 mean / variance 在這次反向傳播中被視為固定值。
        unbiased=False 使用除以觀測數的變異數，也可處理單筆窗口。
        """
        return (
            torch.mean(x, dim=1, keepdim=True).detach(),
            torch.var(x, dim=1, keepdim=True, unbiased=False).detach(),
        )


class SpatialEncoder(nn.Module):
    """先用 GRU 摘要每檔股票的歷史，再用 attention 比較同日股票。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_heads: int = 2,
        dim_feedforward: int = 512,
        num_encoder_layers: int = 1,
        output_size: int = 128,
    ):
        """設定時間摘要與股票間 attention 的維度。

        流程：
            MLP：              (N, T, C) → (N, T, C)
            GRU 最後層狀態：   (N, H)
            加入 batch 維度：  (1, N, H)
            股票間 attention： (1, N, H)
            移除 batch 維度：  (N, H)
            輸出投影：         (N, output_size)

        這裡 attention 的 token 是股票，因此輸入的 N 檔股票必須來自同一天。
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=input_size),
            nn.LayerNorm(normalized_shape=input_size),
            nn.LeakyReLU(),
        )

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            nhead=num_heads,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_encoder_layers
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, output_size),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """把歷史特徵 (N, T, input_size) 編碼成 (N, output_size)。"""
        x = self.mlp(x)

        # GRU 的 hidden: (num_layers, number_of_stocks, hidden_size)。
        _, hidden = self.gru(x)  # shape = (layer num, stock num, features)

        # hidden[-1] 取最後一層，不是最後一檔股票。
        # (N, H) 加上 batch 維度後為 (1, N, H)，股票成為 attention tokens。
        x = hidden[-1].unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        x = self.output(x)

        return x


class VectorQuantizer(nn.Module):
    """把每檔股票的連續表示，對應到 codebook 中最相近的向量。"""

    def __init__(
        self,
        num_embed: int = 512,
        embed_dim: int = 128,
        beta: float = 0.25,
        distance: str = "l2",
    ):
        """建立 (num_embed, embed_dim) 的可訓練 codebook。

        beta 乘在 commitment loss；distance 支援 "l2" 或 "cos"。
        這裡保留你目前的 VQ loss 設計，不宣稱等同原作者的 contrastive 實作。
        """
        super().__init__()
        self.vq_code = nn.Parameter(
            torch.randn(num_embed, embed_dim), requires_grad=True
        )

        self.beta = beta
        self.distance = distance

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """回傳量化表示 (N, embed_dim) 與 scalar auxiliary loss。

        前向數值等於最近 codeword；透過 STE，預測 loss 的梯度可回到 encoder。
        codebook 本身另外由 VQ loss 學習。
        """
        nearest_code = self.find_nearest_code(x)

        # STE = Straight Through Estimator，讓量化後的任務梯度回到 encoder。

        # 前向：x + code - x = code；反向：detach 分支不傳梯度，所以梯度回到 x。
        ste = x + (nearest_code - x).detach()
        loss = self.calculate_loss(x, nearest_code)

        return ste, loss

    def calculate_loss(
        self,
        x: torch.Tensor,
        nearest_code: torch.Tensor,
    ) -> torch.Tensor:
        """合併目前實作的三個 VQ 訓練項，回傳 scalar Tensor。

        contrastive：每檔股票把最相似 code 的 log probability 取負後平均。
            這會鼓勵該 code 相對其他 code 更容易被選中，並不保證所有 code
            都會被使用，也不等同「以每個 code 為中心挑股票」的作者實作。
        commitment：固定選中 code，讓 encoder 的表示靠近它。
        codebook：固定 encoder 表示，讓選中 code 靠近它。

        總和 = beta * commitment_loss + codebook_loss + contrastive_loss。
        """
        # Contrastive loss

        # scores: (N, num_embed)；argmax 回傳每一列最高分 code 的索引 (N,)。
        scores = self.find_similarity_score(x)
        selected_index = scores.argmax(dim=1)

        # log_softmax 仍處理完整 scores 矩陣，shape 不會因為前面 argmax 改變。
        log_prob = F.log_softmax(scores, dim=1)

        stock_indicies = torch.arange(
            start=0, end=x.shape[0], device=x.device
        )

        # 同時給列索引與欄索引：每檔股票只取它選中的那一格，結果是 (N,)。
        selected_log_probs = log_prob[stock_indicies, selected_index]
        contrastive_loss = -selected_log_probs.mean()

        # Commitment loss

        # detach code：這一項只調整 encoder 表示，不更新選中的 code。
        commitment_loss = F.mse_loss(x, nearest_code.detach())

        # Codebook loss

        # detach x：這一項只調整 codebook，不更新 encoder。
        codebook_loss = F.mse_loss(nearest_code, x.detach())

        return self.beta * commitment_loss + codebook_loss + contrastive_loss

    def find_nearest_code(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """對每檔股票選一個最高分 code，回傳 (N, embed_dim)。

        _calculate_distance 雖然叫 distance，實際統一回傳「越大越近」的分數。
        量化可能幫助壓縮表示，但不能保證去除金融雜訊。
        """
        distance = self._calculate_distance(x)
        nearest_code_index = torch.argmax(distance, dim=-1)

        return self.vq_code[nearest_code_index]

    def find_similarity_score(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """回傳每檔股票對每個 code 的分數，shape 為 (N, num_embed)。

        l2：負的歐氏距離，沒有平方；cos：cosine similarity。
        兩種設定都是分數越大越相近。
        """
        distance = self._calculate_distance(x)

        return distance

    def _calculate_distance(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """計算全部股票與全部 code 的配對分數。

        x: (N, embed_dim)，vq_code: (num_embed, embed_dim)。
        輸出: (N, num_embed)，每一列是一檔股票對所有 code 的分數。
        l2 使用 -torch.cdist；cos 使用正規化後的矩陣乘法。
        """

        if self.distance == "l2":
            distance = -torch.cdist(x, self.vq_code)
        elif self.distance == "cos":
            x = F.normalize(x, p=2, dim=-1)
            code = F.normalize(self.vq_code, p=2, dim=-1)

            # (N, D) @ (R, D).T = (N, D) @ (D, R) = (N, R)
            # Every stocks corresponding nearest codes
            distance = x @ code.T
        else:
            raise ValueError(f"Unsupported distance: {self.distance}")

        return distance


class PredictionLinear(nn.Module):
    """Stage 1 的預測頭；每個 codeword 對應一個股票預測值。"""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.output = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """輸入 (N, input_dim)，回傳 (N,)。

        squeeze(-1) 只移除最後的單一輸出維度；N=1 時仍保留股票維度。
        """
        return self.output(x).squeeze(-1)


class Stage_1(nn.Module):
    """共同訓練歷史編碼器、codebook 與第一階段預測頭。"""

    def __init__(
        self,
        num_features: int,
        prior_factor_dim: int,
    ):
        """num_features 決定輸入特徵數；prior_factor_dim 保留但目前未使用。

        目前使用 512 個 128 維 codeword，並直接預測每檔股票的 Target。
        """
        super().__init__()
        # (N, T, num_features) -> (N, T, num_features)
        self.revin = RevIN(num_features=num_features)

        # (N, T, num_features) -> (N, 128)
        self.spatial_encoder = SpatialEncoder(
            input_size=num_features,
            hidden_size=128,
            num_layers=1,
            num_heads=2,
            dim_feedforward=512,
            num_encoder_layers=1,
            output_size=128,
        )

        # (N, 128) -> code (N, 128), loss ()
        self.vector_quantizer = VectorQuantizer(
            num_embed=512, embed_dim=128, beta=0.25, distance="cos"
        )

        # 目前直接用 128 維 code 預測，沒有拼接 prior factors。
        self.output = PredictionLinear(
            input_dim=128, hidden_dim=64, dropout=0.1
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """輸入 x: (N, T, num_features)。

        回傳 prediction: (N,) 與 VQ loss: ()。
        訓練器再加入 prediction MSE，一起反向傳播。
        """

        # 1. 沿時間標準化：(N, T, num_features)，shape 保持不變。
        x = self.revin(x)

        # 2. 摘要歷史並比較同日股票，得到 (N, 128)。
        x = self.spatial_encoder(x)

        # 3. 量化為 codeword，並計算 VQ 輔助 loss。
        code, loss = self.vector_quantizer(x)

        # 4. 每個 codeword 產生一個 Target 預測，得到 (N,)。
        x = self.output(code)

        return x, loss


class RoPESelfAttention(nn.Module):
    """在每個 head 的 Q、K 上做 RoPE，再計算 self-attention。"""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 2,
        dropout: float = 0.1,
    ):
        """d_model 必須可被 nhead 整除，而且每個 head 的維度必須為偶數。

        本類目前沒有 attention-weight dropout；dropout 參數保留於介面。
        外層 TemporalEncoderLayer 另有 attention 輸出 dropout。
        """
        super().__init__()

        if nhead <= 0 or d_model % nhead != 0:
            raise ValueError("d_model must be divisible by a positive nhead")

        if (d_model // nhead) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.WQ = nn.Linear(d_model, d_model, bias=False)
        self.WK = nn.Linear(d_model, d_model, bias=False)
        self.WV = nn.Linear(d_model, d_model, bias=False)

        self.nhead = nhead
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """輸入與輸出都是 (N, sequence_length, d_model)。

        步驟：Q/K/V 投影 → 拆 heads → 旋轉 Q/K → attention → 合併 heads。
        V 不做這個位置旋轉。
        """
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        Q = self.split_head(Q)
        K = self.split_head(K)
        V = self.split_head(V)

        rotated_Q = self.RoPE(Q)
        rotated_K = self.RoPE(K)

        attention_value = self.calculate_attention(rotated_Q, rotated_K, V)
        mearged_scores = self.merge_heads(attention_value)
        final_output = self.output(mearged_scores)
        return final_output

    def split_head(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """把最後一維切成多個 head，再把 head 維度移到前面。

        (N, S, d_model) → (N, S, nhead, head_dim) → (N, nhead, S, head_dim)。
        head_dim = d_model // nhead；必須先 reshape 再 transpose，才不會混到時間。
        """
        N, S, D = x.shape
        if D % self.nhead != 0:
            raise ValueError("d_model must be divisible by n_head")

        reshaped_x = x.reshape(N, S, self.nhead, D // self.nhead)

        # (N, S, D) -> (N, 2, S, D / 2)
        # --> Instead of (2, N, S, D / 2).
        # It satisfies some functions and easier to track the stock's overall info
        reshaped_x = reshaped_x.transpose(1, 2)

        return reshaped_x

    def RoPE(
        self,
        x: torch.Tensor,
        freq_base: float = 10000.0,
    ) -> torch.Tensor:
        """旋轉一個 Q 或 K Tensor；每次呼叫只處理一個 Tensor。

        輸入、輸出都是 (N, number_of_heads, sequence_length, head_dim)。
        每兩個相鄰 feature 分成一組，每組使用自己的固定頻率。
        位置改變的是「角度 = 位置 × 頻率」，不是每天重新選一組頻率。
        分組只方便運算，兩個旋轉後分量都會放回，輸出維度不會減半。
        """
        sequence_shape = x.shape[-2]
        head_dim = x.shape[-1]

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        # 1. 取 feature 配對索引 0, 2, 4, ...；head_dim=32 時，共 16 組。
        freq_sequence = torch.arange(
            start=0,
            end=head_dim,
            step=2,
            device=x.device,
            dtype=torch.float32,
        )

        # 每一組對應一個固定角頻率；shape: (head_dim // 2,)。
        freq_sequence = freq_base ** -(freq_sequence / head_dim)

        # 2. 位置是 0, 1, ..., S-1；包含第一個 code token 的位置 0。
        positions = torch.arange(
            start=0,
            end=sequence_shape,
            step=1,
            device=x.device,
            dtype=torch.float32,
        )

        # (S, 1) @ (1, head_dim // 2) → (S, head_dim // 2)。
        # 第 m 列是位置 m 乘上全部配對頻率，不是頻率每天不同。
        angles = positions[:, None] @ freq_sequence[None, :]

        cos = torch.cos(angles).to(dtype=x.dtype)
        sin = torch.sin(angles).to(dtype=x.dtype)

        # 3. 每組保留兩個分量；first / second 都是 (N, heads, S, head_dim // 2)。
        first = x[..., 0::2]
        second = x[..., 1::2]

        # 這裡是逐元素乘法，不是矩陣乘法。
        # cos / sin 的 (S, head_dim // 2) 會廣播到所有股票與 heads。
        rotated_first = first * cos - second * sin
        rotated_second = first * sin + second * cos

        # 4. stack 在最後新增「每組兩個分量」維度：(..., head_dim // 2, 2)。
        rotated_x = torch.stack([rotated_first, rotated_second], dim=-1)

        # 合併最後兩維，把每組的兩個分量交錯放回，恢復原本 head_dim。
        return torch.flatten(rotated_x, start_dim=-2)

    def calculate_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        # scores:
        # (number_of_stocks, number_of_heads, query_sequence_length, key_sequence_length)
        #
        # output:
        # (number_of_stocks, number_of_heads, query_sequence_length, features_per_head)
        """用 Q 與 K 的相似程度決定如何加權 V。

        Q / K / V: (number_of_stocks, number_of_heads, sequence_length, head_dim)。
        scores: (number_of_stocks, number_of_heads, query_length, key_length)。
        輸出: (number_of_stocks, number_of_heads, query_length, head_dim)。
        目前沒有 causal mask；所有輸入 token 都是預測時已知的歷史與 code。
        """
        attention_scores = Q @ K.transpose(-2, -1)

        # 除以 sqrt(head_dim)，避免內積因維度增加而讓 softmax 過度集中。
        attention_scores = attention_scores / (Q.shape[-1] ** 0.5)

        # 對每個 query，沿最後的 key 維度分配權重，再加權 V。
        weights = torch.softmax(attention_scores, dim=-1)
        attention_output = weights @ V
        return attention_output

    def merge_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """將各 head 的特徵接回同一條向量。

        (N, heads, S, head_dim) → (N, S, heads, head_dim) → (N, S, d_model)。
        """
        N, heads, S, head_dim = x.shape

        # (N, S, heads, head_dim)
        x = x.transpose(1, 2)
        # (N, S, heads * head_dim = d_model)
        reshaped_x = x.reshape(N, S, heads * head_dim)
        return reshaped_x


class TemporalEncoderLayer(nn.Module):
    """一層 temporal encoder：attention 與 feed-forward 各有 residual、norm。"""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.RoPE = RoPESelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.fnn_output_dropout = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """輸入與輸出均為 (number_of_stocks, sequence_length, d_model)。

        使用 post-norm 順序：先加 residual，再 LayerNorm。
        attention 處理 token 間關係；feed-forward 對每個 token 的特徵做轉換。
        """

        # 1. Attention 分支：建立 Q/K/V，在 Q/K 內做 RoPE，再聚合時間資訊。
        attention_output = self.RoPE(x)
        attention_output = self.attention_dropout(attention_output)

        # 把原輸入 x 加回 attention 輸出，這個加法就是 residual connection。
        attention_output = self.layernorm(
            x + attention_output
        )  # residual 加回後再做 LayerNorm

        # 2. Feed-forward 分支：每個 token 各自做特徵轉換，shape 不變。
        feedforward_output = self.feedforward(attention_output)
        feedforward_output = self.fnn_output_dropout(feedforward_output)

        # 第二次 residual 使用這一分支的輸入 attention_output。
        attention_output = self.layernorm2(
            attention_output + feedforward_output
        )  # residual 加回後再做 LayerNorm

        return attention_output


class TemporalTransformer(nn.Module):
    """把 code token 放在歷史序列前端，學習每檔股票的時間表示。"""

    def __init__(
        self,
        historical_data_input_shape: int = 158,
        codeword_input_shape: int = 128,
        linear_output_shape: int = 64,
        final_output_shape: int = 64,
        nhead: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_transformer_layers: int = 3,
    ):
        """建立獨立的多層 encoder，以及最後的特徵融合投影。

        兩種輸入先投影成 linear_output_shape，才能沿序列維度拼接。
        fusion 使用第一個 token 的輸出，以及原始 codeword。
        activation 參數目前保留於介面；encoder 的 feed-forward 固定使用 GELU。
        """
        super().__init__()
        self.num_transformer_layers = num_transformer_layers

        self.historical_data_projection = nn.Linear(
            historical_data_input_shape, linear_output_shape
        )
        self.codeword_projection = nn.Linear(
            codeword_input_shape, linear_output_shape
        )

        self.encoders = nn.ModuleList(
            [
                TemporalEncoderLayer(
                    d_model=linear_output_shape,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(self.num_transformer_layers)
            ]
        )

        self.transformed_layernorm = nn.LayerNorm(linear_output_shape)
        self.codeword_layernorm = nn.LayerNorm(codeword_input_shape)
        self.fusion_linear = nn.Linear(
            linear_output_shape + codeword_input_shape, final_output_shape
        )

    def forward(
        self,
        x: torch.Tensor,
        codeword: torch.Tensor,
    ) -> torch.Tensor:
        """輸入歷史 x: (N, T, historical_data_input_shape)，
        以及 codeword: (N, codeword_input_shape)。

        codeword 的投影版本作為第一個 token；原始版本保留給最後融合。
        回傳 (N, final_output_shape)，不是最終股票預測。
        """

        # 1. 歷史資料與 codeword 各自投影到相同的 token 維度。
        historical_tokens = self.historical_data_projection(x)
        # (N, T, linear_output_shape)

        codeword_token = self.codeword_projection(codeword)
        codeword_token = codeword_token.unsqueeze(1)
        # (N, 1, linear_output_shape)

        # 2. 沿時間維度 dim=1 接起來；第一個 token 是 codeword，不是第一檔股票。
        data = torch.cat(
            [codeword_token, historical_tokens],
            dim=1,
        )
        # (N, T + 1, linear_output_shape)

        # 3. 依序通過多層 encoder；每層有自己的參數，每層都做 RoPE attention。
        for encoder in self.encoders:
            data = encoder(data)

        # 4. 取每檔股票的第 0 個 token，它已經透過 attention 讀過歷史序列。
        first_token = data[:, 0, :]
        # (N, linear_output_shape)

        # 5. 分別正規化時間表示與原始 code，再沿特徵維度拼接。
        # 兩者維度可不同，各自的 LayerNorm 使用各自的維度。
        normalized_data = self.transformed_layernorm(first_token)
        normalized_codeword = self.codeword_layernorm(codeword)

        fused = torch.cat(
            [normalized_codeword, normalized_data],
            dim=-1,
        )
        # (N, codeword_input_shape + linear_output_shape)

        return self.fusion_linear(fused)


class MOE(nn.Module):
    """用 codeword 選專家，再按權重合併專家輸出的特徵向量。"""

    def __init__(
        self,
        codeword_dim: int = 128,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_experts: int = 8,
        top_k: int = 2,
        noisy_gating: bool = True,
        loss_coef: float = 0.01,
        values_dim: int = 128,
    ):
        """設定專家數、top-k gating 與輸入融合層。

        codeword 在進入本類前已經過 LayerNorm。
        mean_mlp / noise_mlp 各輸出 (N, n_experts)，訓練時可加入獨立噪音。
        W_h 從單位矩陣開始學習，訓練和推論都會使用。

        每檔股票只保留 top_k 個 gating 權重，合計為 1。
        本實作仍計算所有專家，未選中的專家輸出乘 0；沒有做稀疏派送加速。

        負載 loss = loss_coef * n_experts * sum(F_j * W_j)。
        F_j 是選中專家 j 的股票比例；W_j 是包含零權重的平均 gating 權重。
        這是目前採用的公式，沒有使用 CV² 或 expected-load 估計。
        """
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError(
                "top_k must be equal or greater than n_experts and need to be at least 1"
            )

        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.loss_coef = loss_coef
        self.n_experts = n_experts

        self.mean_mlp = nn.Sequential(
            nn.Linear(codeword_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_experts),
        )

        self.noise_mlp = nn.Sequential(
            nn.Linear(codeword_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_experts),
        )

        self.softplus = nn.Softplus()
        self.W_h = nn.Parameter(torch.eye(n_experts))  # Identity matrix
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(n_experts)
            ]
        )

        self.lower_dim = nn.Sequential(
            nn.Linear(values_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
        )

        self.fusion = nn.Linear(hidden_dim + codeword_dim, input_dim)

    def forward(
        self,
        values: torch.Tensor,
        codeword: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """輸入 values: (N, values_dim)、codeword: (N, codeword_dim)。

        回傳 combined: (N, output_dim)，以及 scalar load-balancing loss。
        combined 是特徵向量，之後才交給 FinalPrediction 產生每股一個預測。
        """

        # 1. 用 codeword 算出每檔股票的專家權重，以及負載輔助 loss。
        gating_weights = self.gating_network(codeword)  # (N, E)
        loss = self.load_balancing_loss(gating_weights)

        # 2. 把 temporal values 降維，再和 codeword 拼接成專家輸入。
        values = self.lower_dim(values)  # (N, hidden_dim)
        fused_val = torch.cat([codeword, values], dim=-1)
        fused_val = self.fusion(fused_val)  # (N, input_dim)

        # 3. 各專家輸出 (N, output_dim)。
        # [:, i:i+1] 保留 (N, 1)，才能把每檔股票的權重乘到它的全部 features。
        weighted_outputs = [
            self.experts[i](fused_val) * gating_weights[:, i : i + 1]
            for i in range(self.n_experts)
        ]

        # 4. list 裡每個元素代表一個專家；dim=-1 把「專家」放到新增的最後一維。
        # (N, output_dim) × E 個 Tensor → (N, output_dim, E)。
        combined = torch.stack(weighted_outputs, dim=-1)
        # (N, output_dim, E)

        # 沿最後的專家維度加總，留下每檔股票的一條特徵向量。
        combined = combined.sum(dim=-1)
        # (N, output_dim)

        return combined, loss

    def gating_network(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """回傳 (number_of_stocks, n_experts) 的 gating 權重。

        每一列對應一檔股票；只有 top_k 個位置有非零權重，該列合計為 1。
        self.training 由 nn.Module 提供；model.train() / model.eval() 會切換它。
        """
        mean = self.mean_mlp(x)

        # 只有訓練且啟用噪音才加隨機值；model.eval() 時不加。
        if self.training and self.noisy_gating:
            n = torch.randn_like(mean)

            # SoftPlus is for ensuring noise's std is greater than 0
            noise = F.softplus(self.noise_mlp(x)) * n
            mean += noise

        mean = mean @ self.W_h
        top_k_experts, top_k_indicies = torch.topk(mean, k=self.top_k, dim=-1)

        # 只對已選中的 top-k 分數做 softmax，每檔股票的選中權重合計為 1。
        top_k_weights = torch.softmax(
            top_k_experts, dim=-1, dtype=torch.float32
        )

        gates = torch.zeros_like(mean, dtype=torch.float32)

        # 把 top-k 權重填回原專家欄位；未選中的欄位維持 0。
        gates = gates.scatter(src=top_k_weights, dim=-1, index=top_k_indicies)

        return gates

    def load_balancing_loss(
        self,
        gateing_weights: torch.Tensor,
    ):
        """依股票維度 dim=0，統計各專家的選中比例與平均權重。

        gateing_weights: (number_of_stocks, n_experts)。
        freq / weights: (n_experts,)，最後乘積加總成 scalar Tensor。
        係數 loss_coef 在這裡已乘一次，訓練器不需重複乘。

        正權重判斷是離散操作；本式的梯度經由平均 gating weight 回傳。
        top_k > 1 時 freq 總和為 top_k，這裡保留原實作的尺度。
        """
        # 每個專家的選中比例與平均權重，皆以股票數為分母。

        # dim=0 沿股票方向加總，得到「每個專家」被多少股票選中，再除股票數。
        freq = (gateing_weights > 0).sum(dim=0) / gateing_weights.shape[0]
        weights = gateing_weights.mean(dim=0)

        return self.loss_coef * self.n_experts * (freq * weights).sum()


class FinalPrediction(nn.Module):
    """把 MoE 的輸出向量轉成每檔股票一個預測值。"""

    def __init__(
        self,
        input_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Linear(input_dim, input_dim // 2) → GELU → Dropout → Linear(..., 1)。"""
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim // 2)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear_prediction = nn.Linear(input_dim // 2, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """輸入 (N, input_dim)，回傳 (N,)；只移除最後的單一輸出維度。"""
        x = self.linear_prediction(self.dropout(self.gelu(self.linear(x))))
        return x.squeeze(-1)


class Stage_2(nn.Module):
    """固定 Stage 1 編碼器與 codebook，只訓練時間模型、MoE 與預測頭。"""

    def __init__(
        self,
        trained_stage_1: Stage_1,
        num_features: int = 158,
    ):
        """接收已恢復最佳權重的 Stage_1，沿用它的子模組物件。

        不是複製一組新參數；requires_grad_(False) 也會影響同一份 Stage_1。
        如果未來要重新訓練那份 Stage_1，需要另外處理參數的凍結狀態。
        """
        super().__init__()

        self.code_encoder = nn.Sequential(
            trained_stage_1.revin,
            trained_stage_1.spatial_encoder,
        )
        self.vector_quantizer = trained_stage_1.vector_quantizer

        # Fix the model, don't alter it
        self.code_encoder.requires_grad_(False)
        self.vector_quantizer.requires_grad_(False)

        # Stage 2, models which require training
        self.transformer = TemporalTransformer(
            historical_data_input_shape=num_features,
        )
        self.codeword_norm = nn.LayerNorm(128)
        self.moe = MOE(values_dim=64)
        self.final_prediction = FinalPrediction(input_dim=64)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Since model.train() will alter every parts of the model into training mode
        # Use .eval() to make it fixed to evaluation
        """輸入 x: (N, T, num_features)。

        固定的 encoder / VQ 先產生 (N, 128) codeword。
        可訓練的時間模型與 MoE 再產生 (N, 64) 特徵。
        回傳 prediction: (N,) 與 MoE auxiliary loss: ()。
        """
        self.code_encoder.eval()
        self.vector_quantizer.eval()

        # 1. 固定分支只提供 codeword；不建立梯度圖。
        # eval 關閉訓練模式，requires_grad=False 凍結參數，兩者用途不同。
        with torch.no_grad():
            encoded = self.code_encoder(x)
            # (N stock, 128)

            codeword = self.vector_quantizer.find_nearest_code(encoded)
            # (N stock, 128)

        # 2. 從這裡開始是 Stage 2 的可訓練部分，時間表示為 (N, 64)。
        temporal_values = self.transformer(x, codeword)

        combined, model_loss = self.moe(
            temporal_values,
            self.codeword_norm(codeword),
        )

        # 3. 將 MoE 合併後的特徵轉成每股一個 Target 預測。
        prediction = self.final_prediction(combined)

        return prediction, model_loss


# ======================================================================
# 4. 資料序列：每檔股票的歷史窗口，以及同一天的股票 batch
# ======================================================================


class StockSequenceDataset(Dataset):
    """按股票建立歷史窗口；需要用到某個樣本時，才切出它的資料。

    原始資料先依股票、日期排序，所以同一股票的歷史是連續的。
    endpoints 儲存各個合法窗口的最後一列位置，而不是複製所有窗口。
    """

    def __init__(
        self,
        df,
        feature_columns,
        lookback,
        target_horizon=2,
    ):
        """準備特徵矩陣、合法窗口，以及每個樣本的 metadata。

        df 至少要有 Date、SecuritiesCode、Target，以及 feature_columns。
        lookback 是窗口包含的觀測筆數；窗口最後一天也包含在輸入中。
        target_horizon 預設為 2，用同一股票往後兩筆觀測日估計標籤結束日。
        停牌缺日不會自動補齊，所以觀測筆數不一定等於日曆天數。
        """

        # 1. 檢查輸入：Target 不能混入特徵，同一股票、日期不可重複。
        if not isinstance(lookback, (int, np.integer)) or lookback < 1:
            raise ValueError("lookback must be a positive integer")
        if target_horizon < 1:
            raise ValueError("target_horizon must be positive")
        if not feature_columns or len(set(feature_columns)) != len(
            feature_columns
        ):
            raise ValueError("feature_columns must be nonempty and unique")
        if "Target" in feature_columns or "Date" in feature_columns:
            raise ValueError("Target and Date cannot be input features")
        self.lookback = int(lookback)
        frame = df.copy()
        frame["Date"] = pd.to_datetime(frame["Date"])
        if frame[["Date", "SecuritiesCode"]].isna().any().any():
            raise ValueError("Missing Date or SecuritiesCode")
        if frame.duplicated(["Date", "SecuritiesCode"]).any():
            raise ValueError("Duplicate Date/SecuritiesCode rows")

        # 2. 依股票分組、組內依時間排序，後面的切片才不會跨到其他股票。
        frame = frame.sort_values(["SecuritiesCode", "Date"]).reset_index(
            drop=True
        )

        # features: (原始資料列數, 特徵數)；targets: (原始資料列數,)。
        self.features = frame[feature_columns].to_numpy(
            dtype=np.float32, copy=True
        )
        self.targets = frame["Target"].to_numpy(dtype=np.float32, copy=True)

        # 3. 找出每一檔股票的有效窗口末端。
        endpoints = []
        for _, group in frame.groupby("SecuritiesCode", sort=False):
            rows = group.index.to_numpy()

            # bad 是每一列是否有 NaN / Inf 的布林陣列。
            # prefix 累計無效列數；窗口前後累計值相同，就代表窗口內沒有無效列。
            bad = ~np.isfinite(self.features[rows]).all(axis=1)
            prefix = np.r_[0, np.cumsum(bad)]

            # 最早可用的末端是 lookback - 1；例如 3 筆窗口的末端從索引 2 開始。
            ends = np.arange(self.lookback - 1, len(rows))
            valid = prefix[ends + 1] == prefix[ends + 1 - self.lookback]

            # 歷史 Target 不作為輸入，只要求窗口最後一天的 Target 有效。
            valid &= np.isfinite(self.targets[rows[ends]])
            endpoints.extend(rows[ends[valid]])
        self.endpoints = np.asarray(endpoints, dtype=np.int64)

        # 4. 標籤結束日只用於資料切分，不會送進模型當特徵。
        # shift(-2) 把同一股票往後兩筆的日期放到目前這列；尾端不足時為 NaT。
        label_end = frame.groupby("SecuritiesCode", sort=False)["Date"].shift(
            -target_horizon
        )
        self.metadata = frame.loc[
            self.endpoints, ["Date", "SecuritiesCode", "Target"]
        ].copy()
        self.metadata["LabelEndDate"] = label_end.loc[
            self.endpoints
        ].to_numpy()
        self.metadata = self.metadata.reset_index(drop=True)

        # SampleId 是 dataset 的索引；不等於原始 DataFrame 的列索引。
        self.metadata["SampleId"] = np.arange(len(self.metadata))

    def __len__(self):
        """回傳有效窗口數量；不是原始 DataFrame 的列數。"""
        return len(self.endpoints)

    def __getitem__(
        self,
        sample_id,
    ):
        """取出一個樣本，供 DataLoader 組成一批。

        回傳 x: (lookback, number_of_features)、y: ()、sample_id: int。
        多個樣本組成 batch 後，x 才變成 (number_of_stocks, lookback, features)。
        """
        end = int(self.endpoints[sample_id])

        # Python 切片不含右邊界，所以 end + 1 才能包含末端那一筆。
        x = torch.from_numpy(self.features[end + 1 - self.lookback : end + 1])
        y = torch.tensor(self.targets[end], dtype=torch.float32)
        return x, y, int(sample_id)


def make_daily_loader(
    dataset,
    sample_ids,
    *,
    shuffle_days=False,
    seed=42,
):
    """依預測日期分批，讓 SpatialEncoder 只比較同一天的股票。

    sample_ids 決定這個 loader 包含哪些樣本。
    每個 batch 是一份索引 list，例如 [0, 38, 76, 114]。
    shuffle_days=True 只打亂「整批日期」順序，不會混合不同日期。
    日期順序在建立 loader 時洗牌一次；每個 epoch 會重用這份順序。
    """
    ids = np.asarray(sample_ids, dtype=np.int64)
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("sample_ids must be nonempty and unique")
    if ids.min() < 0 or ids.max() >= len(dataset):
        raise ValueError("Sample index out of range")
    rows = dataset.metadata.iloc[ids]

    # 每個日期取出一份 SampleId 清單；DataLoader 依這些清單取樣。
    batches = [
        day["SampleId"].tolist() for _, day in rows.groupby("Date", sort=True)
    ]
    if shuffle_days:
        np.random.default_rng(seed).shuffle(batches)

    # 指定 batch_sampler 後，不再設定固定 batch_size；每批股票數可不同。
    return DataLoader(dataset, batch_sampler=batches, num_workers=0)


def split_daily_loaders(
    dataset,
    split,
    validation_fraction=0.2,
    *,
    end_inclusive=True,
):
    """切出內層訓練、內層驗證、外層回測，並排除標籤跨界的樣本。

    split = (train_start, train_end, test_start, test_end)。
    end_inclusive=True：包含迄日，保留直接呼叫時的原本介面。
    end_inclusive=False：不包含迄日，對應原專案 utils 的 [start, end)。
    從外層訓練區間尾端保留 validation_fraction 比例的可用日期做驗證。

    除了看樣本日期，還要看 Target 何時才會完整得知：
        fit 標籤結束日 < validation 起點。
        validation 標籤結束日 < test 起點。

    回傳 (fit_loader, valid_loader, test_loader, validation_start)。
    切分依據是窗口最後一天；較早的歷史仍可當作窗口輸入。
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    train_start, train_end, test_start, test_end = map(pd.Timestamp, split)
    if end_inclusive:
        ordered = train_start <= train_end < test_start <= test_end
    else:
        ordered = train_start < train_end <= test_start < test_end
    if not ordered:
        raise ValueError(
            "Split dates must be ordered and train/test must not overlap"
        )
    meta = dataset.metadata

    # 1. 先找外層訓練範圍，而且標籤在回測開始之前已能完整得知。
    inclusive = "both" if end_inclusive else "left"
    outer_train = meta["Date"].between(
        train_start, train_end, inclusive=inclusive
    )
    known_before_test = meta["LabelEndDate"] < test_start
    dates = sorted(meta.loc[outer_train & known_before_test, "Date"].unique())
    if len(dates) < 3:
        raise ValueError(
            "Too few eligible training dates for inner validation"
        )

    # 2. 按「日期數」保留最後一段，不是隨機抽股票。
    n_validation = max(1, int(np.ceil(len(dates) * validation_fraction)))
    validation_start = pd.Timestamp(dates[-n_validation])

    # 3. 移除標籤跨過驗證起點的訓練樣本，避免偷看驗證期間資訊。
    fit = outer_train & (meta["Date"] < validation_start)
    fit &= meta["LabelEndDate"] < validation_start
    valid = (
        outer_train & (meta["Date"] >= validation_start) & known_before_test
    )
    test = meta["Date"].between(test_start, test_end, inclusive=inclusive)

    # 4. 三組各自依日期分 batch，並保留相同 dataset 的樣本身分。
    loaders = [
        make_daily_loader(dataset, meta.index[mask].to_numpy())
        for mask in (fit, valid, test)
    ]
    return (*loaders, validation_start)


# ======================================================================
# 5. 訓練與驗證：兩階段共用，驗證集選最佳權重
# ======================================================================


def _batch(
    batch,
    device,
):
    """拆開一個 batch，並把輸入與標籤移到指定裝置。

    支援新格式 (x, y, sample_ids)，也保留舊格式 (x, y)。
    sample_ids 只用來查 metadata，沒有必要搬到 GPU。
    """
    if len(batch) not in (2, 3):
        raise ValueError("Expected (x, y) or (x, y, sample_id)")
    x, y = batch[:2]
    return x.to(device), y.to(device), batch[2] if len(batch) == 3 else None


def _check(
    prediction,
    target,
    prediction_loss,
    model_loss,
):
    """檢查 prediction / target 的 shape，以及兩個 loss 是否有效。

    prediction 和 target 都必須是 (number_of_stocks,)。
    loss 必須是有限的 scalar Tensor；NaN、Inf 或向量都會直接報錯。
    """
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError(
            f"Expected matching (stocks,) shapes: {prediction.shape}, {target.shape}"
        )
    for name, value in (
        ("prediction_loss", prediction_loss),
        ("model_loss", model_loss),
    ):
        if value.ndim != 0 or not torch.isfinite(value):
            raise ValueError(
                f"{name} must be a finite scalar; use loss reduction='mean'"
            )


def evaluation(
    model,
    data_loader,
    loss_fn,
    device="cpu",
):
    """用已訓練的模型計算 loss，並產生對齊日期、股票的預測表。

    參數：
        model       : Stage_1 或 Stage_2，回傳 (prediction, auxiliary_loss)。
        data_loader : 每次提供同一預測日期的股票。
        loss_fn     : 每批回傳平均 loss，例如 nn.MSELoss(reduction="mean")。
        device      : 模型運算裝置，例如 "cpu"、"cuda" 或 "mps"。

    回傳：
        prediction_loss : float，所有樣本的平均預測誤差。
        auxiliary_loss  : float，按樣本數加權的平均輔助 loss。
        predictions     : DataFrame，包含 Target、Prediction；有 sample_ids
                          時還包含 Date、SecuritiesCode 等 metadata。

    模型輔助 loss 已包含模型內設定的係數，這裡不重複乘。
    結束時會恢復呼叫前的 train / eval 模式。
    """

    # 1. 切換到評估模式：關閉 Dropout，以及 MoE 的訓練期隨機噪音。
    previous_mode = model.training
    model.to(device)
    model.eval()
    total_loss, total_aux, count = 0.0, 0.0, 0
    frames = []
    try:
        # 2. 評估不建立反向傳播計算圖；節省記憶體，也不更新模型。
        with torch.no_grad():
            for batch in data_loader:
                x, y, sample_ids = _batch(batch, device)
                prediction, auxiliary = model(x)

                # 先檢查 shape 再計算 loss，避免 (N,) 與 (N, 1) 被廣播成 (N, N)。
                if prediction.shape != y.shape or prediction.ndim != 1:
                    raise ValueError(
                        f"Prediction/target shape mismatch: {prediction.shape}, {y.shape}"
                    )
                loss = loss_fn(prediction, y)
                _check(prediction, y, loss, auxiliary)

                # 每批股票數可能不同。先把平均 loss 乘回股票數，最後再除總數。
                n = len(y)
                total_loss += loss.item() * n
                total_aux += auxiliary.item() * n
                count += n
                target = y.cpu().numpy()

                # 3. 用這批的 SampleId 查回身分；不能假設 loader 與原表格排序相同。
                if sample_ids is not None:
                    if not hasattr(data_loader.dataset, "metadata"):
                        raise ValueError(
                            "Sample IDs require dataset.metadata"
                        )
                    ids = (
                        torch.as_tensor(sample_ids)
                        .cpu()
                        .numpy()
                        .astype(np.int64)
                    )

                    # 一個 id 對應一列 metadata，因此順序和本批 prediction 一致。
                    frame = data_loader.dataset.metadata.iloc[ids].copy()
                    if not np.allclose(
                        frame["Target"], target, rtol=1e-5, atol=1e-7
                    ):
                        raise ValueError(
                            "Metadata targets do not match this batch"
                        )
                else:
                    frame = pd.DataFrame({"Target": target})
                frame["Prediction"] = prediction.cpu().numpy()
                frames.append(frame)
    finally:
        # 即使中途報錯，也恢復呼叫前的模式。
        model.train(previous_mode)
    if count == 0:
        raise ValueError("Empty evaluation loader")

    # 4. 合併各批預測；同一筆樣本不可在同一次 evaluation 重複出現。
    result = pd.concat(frames, ignore_index=True)
    if "SampleId" in result and result["SampleId"].duplicated().any():
        raise ValueError("Duplicate samples in evaluation")
    return total_loss / count, total_aux / count, result


def train(
    model,
    data_loader,
    loss_fn,
    optimizer,
    epochs=50,
    device="cpu",
    patience=10,
    min_change=1e-5,
    *,
    validation_loader,
    checkpoint_path=None,
    checkpoint_metadata=None,
):
    """訓練模型，用內層驗證集選擇最佳 epoch，最後恢復最佳權重。

    每個 epoch：
        1. 訓練集 forward → prediction loss + auxiliary loss。
        2. backward → gradient clipping → optimizer.step()。
        3. 只用驗證集 prediction loss 判斷是否改善與 early stopping。
        4. 保存最佳模型與 optimizer；累積每輪的 loss 紀錄。

    重要參數：
        data_loader       : 更新權重的 fit 資料，不包含驗證集或回測集。
        validation_loader : 選 epoch 的內層驗證資料；不做 backward。
        epochs            : 最多訓練幾輪。
        patience          : 連續幾輪沒有足夠改善，就停止訓練。
        min_change        : 重設 patience 計數器需要超過的改善幅度。
        checkpoint_path   : 最佳權重的 .pt 路徑；None 表示只保留在記憶體。
        checkpoint_metadata : 同時存入檔案的特徵、日期、lookback 等資訊。

    回傳 dictionary：best_epoch、best_validation_loss、history。
    請先 model.to(device)，再建立 optimizer，最後呼叫本函式。
    """

    # 1. 檢查設定，避免開始訓練之後才發現裝置或 loss 格式錯誤。
    if (
        not isinstance(epochs, (int, np.integer))
        or epochs < 1
        or patience < 1
    ):
        raise ValueError("epochs and patience must be positive integers")
    if min_change < 0:
        raise ValueError("min_change must be nonnegative")
    if getattr(loss_fn, "reduction", "mean") != "mean":
        raise ValueError("This trainer requires loss reduction='mean'")
    requested_device = torch.device(device)
    if any(
        p.device.type != requested_device.type for p in model.parameters()
    ):
        raise ValueError(
            "Move model to device before constructing the optimizer"
        )

    # best_loss：實際出現過的最低驗證 loss，用來保存權重。
    # patience_reference：上次足夠改善時的 loss，用來決定是否繼續等。
    best_loss, patience_reference = math.inf, math.inf
    stale, best_epoch, history = 0, 0, []
    best_state = best_optimizer = None

    # 2. 外層 loop：一個 epoch 會跑完所有訓練日期。
    for epoch in range(1, epochs + 1):
        model.train()
        total_pred = total_aux = 0.0
        count = 0

        # 內層 loop：一次處理同一日期的所有可用股票。
        for batch in data_loader:
            x, y, _ = _batch(batch, device)

            # 清除上一批梯度；set_to_none=True 表示暫時不配置梯度 Tensor。
            optimizer.zero_grad(set_to_none=True)
            prediction, auxiliary = model(x)
            if prediction.shape != y.shape or prediction.ndim != 1:
                raise ValueError(
                    f"Prediction/target shape mismatch: {prediction.shape}, {y.shape}"
                )
            prediction_loss = loss_fn(prediction, y)
            _check(prediction, y, prediction_loss, auxiliary)

            # 預測誤差訓練最終任務；輔助 loss 訓練 VQ 或 MoE。
            # auxiliary 已含模型內設定的係數，不在這裡再乘一次。
            loss = prediction_loss + auxiliary
            loss.backward()

            # 把所有可訓練參數的梯度總範數限制在 1.0，減少過大更新。
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            n = len(y)
            count += n
            total_pred += prediction_loss.item() * n
            total_aux += auxiliary.item() * n
        if count == 0:
            raise ValueError("Empty training loader")

        # 3. 每個 epoch 結束後看內層驗證集；outer test 不參與選 epoch。
        val_loss, val_aux, _ = evaluation(
            model, validation_loader, loss_fn, device
        )

        # 分別記錄 prediction / auxiliary，才看得出總 loss 是哪一項在變。
        record = dict(
            epoch=epoch,
            train_prediction_loss=total_pred / count,
            train_model_loss=total_aux / count,
            validation_prediction_loss=val_loss,
            validation_model_loss=val_aux,
        )
        history.append(record)
        print(
            f"Epoch {epoch}/{epochs}: train_prediction={total_pred/count:.6g}, "
            f"train_aux={total_aux/count:.6g}, val_prediction={val_loss:.6g}, "
            f"val_aux={val_aux:.6g}",
            flush=True,
        )

        # 4. 只要真的出現更低的驗證 loss，就保存這一輪的權重。
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch

            # detach 切斷計算圖，cpu 移出 GPU，clone 避免後續更新改到這份副本。
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_optimizer = deepcopy(optimizer.state_dict())
            if checkpoint_path is not None:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)

                # 先寫暫存檔，再換成正式 checkpoint，避免寫到一半破壞舊檔。
                temporary = path.with_suffix(path.suffix + ".tmp")
                torch.save(
                    dict(
                        model_state_dict=best_state,
                        optimizer_state_dict=best_optimizer,
                        best_epoch=best_epoch,
                        best_validation_loss=best_loss,
                        metadata=checkpoint_metadata or {},
                    ),
                    temporary,
                )
                temporary.replace(path)

        # 5. 足夠改善才重設等待次數；微小改善仍會保存，但不一定重設 patience。
        if patience_reference - val_loss > min_change:
            patience_reference, stale = val_loss, 0
        else:
            stale += 1
        if checkpoint_path is not None:
            pd.DataFrame(history).to_csv(
                Path(checkpoint_path).with_suffix(".history.csv"), index=False
            )
        if stale >= patience:
            print(
                f"Early stopping: validation did not improve enough for {patience} epochs",
                flush=True,
            )
            break

    # 6. 回傳之前恢復最佳 epoch，而不是保留停止當下那一輪。
    # optimizer 也恢復到同一輪，讓權重與更新狀態一致。
    model.load_state_dict(best_state)
    optimizer.load_state_dict(best_optimizer)
    model.eval()
    return dict(
        best_epoch=best_epoch, best_validation_loss=best_loss, history=history
    )


# ======================================================================
# 6. 績效：依每日預測排名選股，計算 Sharpe 與 Rank IC
# ======================================================================


def rank(
    df: pd.DataFrame,
    col_name: str,
) -> pd.DataFrame:
    """在各日期內由大到小排名，新增「欄位名_rank」欄位。

    例如 col_name="Prediction"，就新增 Prediction_rank。
    此函式會直接在傳入的 DataFrame 上新增欄位。
    """
    df[f"{col_name}_rank"] = df.groupby("Date")[col_name].rank(
        ascending=False
    )
    return df


def calculate_weights(
    df: pd.DataFrame,
    n_long: int = 200,
    n_short: int = 200,
) -> pd.DataFrame:
    """接收「單一天」的股票表，挑出預測最高與最低的股票。

    多頭總權重 = n_long / (n_long + n_short)。
    空頭總權重 = -n_short / (n_long + n_short)。
    兩邊權重絕對值總和為 1；各側沿用由 2 到 1 的線性排名權重。
    例如各挑 200 檔時，多頭總權重 +0.5、空頭 -0.5。

    回傳包含 Weight 欄位的新表格；沒有選中的股票權重為 0。
    """
    for name, count in (("n_long", n_long), ("n_short", n_short)):
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError(f"{name} must be a non-negative integer")
        if count < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    # Normalize NumPy integer inputs before adding or slicing.
    n_long, n_short = int(n_long), int(n_short)
    total_count = n_long + n_short
    if total_count == 0:
        raise ValueError("At least one of n_long or n_short must be positive")
    if len(df) < total_count:
        raise ValueError(
            f"Need at least {total_count} stocks per day "
            f"({n_long} long + {n_short} short); got {len(df)}"
        )

    df = df.copy()
    if "Prediction_rank" not in df:
        df = rank(df, "Prediction")

    # Keep the original ordering, including its handling of tied predictions.
    df = df.sort_values("Prediction_rank", kind="stable")
    weight_column = np.zeros(len(df))

    if n_long > 0:
        # 排名越前，原始權重越大；除以總和後再乘多頭預算。
        raw_long = np.linspace(2, 1, n_long)
        long_budget = n_long / total_count
        weight_column[:n_long] = long_budget * raw_long / raw_long.sum()

    if n_short > 0:
        raw_short = np.linspace(2, 1, n_short)
        short_budget = n_short / total_count
        short_weights = short_budget * raw_short / raw_short.sum()

        # 最後一列是預測最低的股票，反轉權重後它會拿到最大的空頭部位。
        weight_column[-n_short:] = -short_weights[::-1]

    df["Weight"] = weight_column
    return df


def daily_earning(
    df: pd.DataFrame,
) -> np.array:
    """計算每天的組合報酬：sum(Weight * Target)。

    回傳一維 numpy 陣列，依日期排序，每個元素代表一天。
    Target 是事後真實結果；選股與權重只由 Prediction 決定。
    """
    return np.array(
        [
            float((daily_df["Weight"] * daily_df["Target"]).sum())
            for _, daily_df in df.groupby("Date")
        ],
        dtype="float32",
    )


def calculate_sharpe(
    df: pd.DataFrame,
    base_number: int | float | np.float32 | np.float64 = 0.0,
) -> float:
    """用每日報酬計算未年化 Sharpe。

    公式：(每日報酬平均 - base_number) / 每日報酬樣本標準差。
    base_number 使用與每日報酬一致的單位；預設為 0。
    至少需要兩天；標準差為 0 時回傳 NaN，表示此比值未定義。
    """
    if df.empty:
        raise ValueError("df is empty")

    daily_earnings = daily_earning(df)

    if len(daily_earnings) < 2:
        raise ValueError("Required more than 2 days' earnings")

    std = daily_earnings.std(ddof=1)
    if std == 0:
        return float("nan")  # Undefined for constant daily returns.

    return float((daily_earnings.mean() - base_number) / std)


def calculate_rank_IC(
    df: pd.DataFrame,
) -> float:
    """每天計算 Prediction 與 Target 的 Spearman 排名相關，再取平均。

    這裡衡量當日排序能力，不把不同日期的股票混在一起算相關性。
    某天若沒有可定義的相關係數會得到 NaN，pandas.mean 會略過它。
    """
    if "Prediction" not in df:
        raise ValueError("Missing required columns in df")

    return (
        df.groupby("Date")[["Prediction", "Target"]]
        .apply(
            (
                lambda day: day["Prediction"].corr(
                    day["Target"], method="spearman"
                )
            )
        )
        .mean()
    )


def performance_from_predictions(
    eval_df: pd.DataFrame,
    *,
    n_long: int = 200,
    n_short: int = 200,
) -> dict[str, float]:
    """評估已帶有日期、股票代碼的預測表。

    必要欄位：Date、SecuritiesCode、Target、Prediction。
    每個「日期 + 股票」只能出現一次，每天至少要有 n_long + n_short 檔。

    回傳 dictionary：
        sharpe   : 未年化的日報酬 Sharpe。
        rank_IC  : 每日 Spearman 排名相關的平均。
        earnings : 每日組合 Target 的算術總和，不是複利淨值。
    """

    # 1. 先確認樣本身分、數值與唯一性，避免拿錯配的預測計算績效。
    required = ["Date", "SecuritiesCode", "Target", "Prediction"]
    if not set(required).issubset(eval_df.columns):
        raise ValueError(f"Required columns: {required}")
    eval_df = eval_df[required].copy()
    eval_df["Date"] = pd.to_datetime(eval_df["Date"])
    if eval_df[["Date", "SecuritiesCode"]].isna().any().any():
        raise ValueError("Missing sample identity")
    if eval_df.duplicated(["Date", "SecuritiesCode"]).any():
        raise ValueError("Duplicate Date/SecuritiesCode predictions")
    if not np.isfinite(eval_df[["Target", "Prediction"]].to_numpy()).all():
        raise ValueError("Target/Prediction contains NaN or infinity")
    eval_df = eval_df.sort_values(["Date", "SecuritiesCode"])

    if eval_df.empty:
        raise ValueError("Missing data in DataFrame")

    # 2. 每個日期獨立排名與選股，不能把整段期間一起排名。
    weighted_days = []

    for _, daily_df in eval_df.groupby("Date"):
        weighted_df = calculate_weights(
            daily_df, n_long=n_long, n_short=n_short
        )
        weighted_days.append(weighted_df)

    eval_df = pd.concat(weighted_days, ignore_index=True)

    # 3. 用真實 Target 評估選股結果；不在這裡訓練或調整模型。
    earnings = daily_earning(eval_df).sum()
    sharpe = calculate_sharpe(eval_df)
    rank_IC = calculate_rank_IC(eval_df)
    # Target portfolio exposure, with short positions carrying negative weights.
    long_count, short_count = int(n_long), int(n_short)
    total_count = long_count + short_count
    long_exposure = long_count / total_count
    short_exposure = -short_count / total_count
    # Calculate directly from counts so equal sides display exactly 0.00%.
    net_exposure = (long_count - short_count) / total_count
    print(
        "=============================== Evaluation =============================== "
    )
    print(f"Long: {long_count} stocks | Weight: {long_exposure:.2%}")
    print(f"Short: {short_count} stocks | Weight: {short_exposure:.2%}")
    print(f"Gross exposure: {long_exposure - short_exposure:.2%}")
    print(f"Net exposure: {net_exposure:.2%}")
    print(
        f"Sharpe: {sharpe} \nRank IC: {rank_IC}\nEarnings: {earnings * 100}%"
    )
    print(
        "========================================================================== "
    )

    return {"sharpe": sharpe, "rank_IC": rank_IC, "earnings": earnings}


# ======================================================================
# 7. 基本檢查：裝置、日期邊界與各窗口的有效股票數
# ======================================================================


def choose_device(
    requested="auto",
) -> torch.device:
    """auto 優先 CUDA，否則 CPU；Apple MPS 需明確指定。

    Mac 預設 CPU，避免未驗證的 MPS 算子讓訓練中途失敗。
    若要試 Apple GPU，可用 DEVICE = "mps"；本包未實測 MPS。
    """
    if str(requested) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device 只接受 auto、cpu、cuda 或 mps")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("目前環境沒有可用的 CUDA GPU，請使用 DEVICE = 'cpu'")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("目前環境沒有可用的 MPS，請使用 DEVICE = 'cpu'")
    return device


def normalize_splits(
    splits,
    *,
    end_inclusive,
) -> list[tuple]:
    """明確區分閉區間與半開區間，不靠加減一天猜測日期邊界。"""
    result = [
        tuple(pd.Timestamp(value) for value in split) for split in splits
    ]
    if not result:
        raise ValueError(
            "沒有可用的日期切分；請增加日期範圍或縮短 split_days"
        )

    previous_test_end = None
    previous_train_end = None
    for train_start, train_end, test_start, test_end in result:
        if end_inclusive:
            ordered = train_start <= train_end < test_start <= test_end
        else:
            ordered = train_start < train_end <= test_start < test_end
        if not ordered:
            raise ValueError("日期切分順序錯誤，或訓練與回測區間重疊")

        if previous_test_end is not None:
            overlap = (
                test_start <= previous_test_end
                if end_inclusive
                else test_start < previous_test_end
            )
            if overlap:
                raise ValueError("外層回測日期重複，會造成樣本被計算兩次")
        if previous_train_end is not None and train_end < previous_train_end:
            raise ValueError("後一輪訓練迄日不可早於前一輪")
        previous_test_end = test_end
        previous_train_end = train_end
    return result


def validate_daily_loaders(
    fit_loader,
    valid_loader,
    test_loader,
    *,
    n_long,
    n_short,
) -> dict:
    """確認三段資料可使用，且每天有足夠股票供指定的多空組合選股。"""
    for name, count in (("n_long", n_long), ("n_short", n_short)):
        if isinstance(count, bool) or not isinstance(
            count, (int, np.integer)
        ):
            raise ValueError(f"{name} 必須是非負整數")
        if count < 0:
            raise ValueError(f"{name} 不可為負數")
    needed = n_long + n_short
    if needed == 0:
        raise ValueError("n_long 和 n_short 不能同時為 0")

    # 一批就是一個日期，所以 len(loader) 等於可用日期數。
    if len(test_loader) < 2:
        raise ValueError("回測至少需要兩個有效日期才能計算 Sharpe")

    for batch_ids in test_loader.batch_sampler:
        if len(batch_ids) < needed:
            date = test_loader.dataset.metadata.iloc[batch_ids[0]]["Date"]
            raise ValueError(
                f"{date.date()} 只有 {len(batch_ids)} 檔有效股票，"
                f"但多空設定需要 {needed} 檔。"
                "請檢查缺值 / 特徵，或明確調整 N_LONG 和 N_SHORT。"
            )

    return {
        "fit_dates": len(fit_loader),
        "validation_dates": len(valid_loader),
        "test_dates": len(test_loader),
        "minimum_test_stocks": min(map(len, test_loader.batch_sampler)),
    }


def preflight_data(
    df,
    col_names,
    time_spans,
    splits,
    *,
    validation_fraction,
    target_horizon,
    n_long,
    n_short,
    end_inclusive,
) -> pd.DataFrame:
    """在任何訓練開始前，逐一檢查全部 lookback / split。

    只建立索引與日期批次，不執行 forward / backward。
    避免訓練很久後，才在某個回測窗口發現股票數量不足。
    """
    reports = []
    for lookback in time_spans:
        dataset = StockSequenceDataset(
            df, col_names, lookback, target_horizon
        )
        for split_index, split in enumerate(splits, start=1):
            fit, valid, test, validation_start = split_daily_loaders(
                dataset,
                split,
                validation_fraction,
                end_inclusive=end_inclusive,
            )
            counts = validate_daily_loaders(
                fit, valid, test, n_long=n_long, n_short=n_short
            )
            reports.append(
                {
                    "lookback": lookback,
                    "split": split_index,
                    "validation_start": validation_start.date(),
                    **counts,
                }
            )

    report = pd.DataFrame(reports)
    if report.empty:
        raise ValueError("沒有可執行的視窗；請檢查日期範圍與 split_days")
    print("\n開跑前檢查：", flush=True)
    print(report.to_string(index=False), flush=True)
    return report


if __name__ == "__main__":
    main()

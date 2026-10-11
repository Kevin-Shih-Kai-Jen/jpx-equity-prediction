# JPX CNN-LSTM：自行修正用的 Debug 清單

## 本次檢查範圍

已讀取六個 Python 檔，並以小型合成資料實際呼叫其中函式。未取得 `stock_prices.csv`，未執行真實 JPX 全量訓練，也未核定 AdjustmentFactor 生效日、真實缺失比例或回測表現。原始六個檔案沒有被改寫。

測試環境：Python 3.13.5、pandas 2.2.3、NumPy 2.3.5、PyTorch 2.10.0+cpu、scikit-learn 1.8.0。你的本機版本可能不同，錯誤訊息文字可能略有差異。

`test_jpx_debug.py` 是驗收工具，不是修正版。27 個測試方法中，目前 9 個通過、18 個未通過；部分失敗有共同根因，不能解讀成 18 個獨立 bug。unittest 顯示的 11 failures + 8 errors 含一個方法下的兩個 factor 子案例。

## 使用方式

把測試檔放在 `deep_learning.py`、`features.py` 等六個模組所在的同一個資料夾，在專案原本的 Python 環境執行。測試使用假資料，不會讀取你 Mac 上的 CSV 路徑。

建議逐輪執行，而非一開始盯著整份失敗紀錄：

```bash
python3 -m unittest -v test_jpx_debug.TensorTests
python3 -m unittest -v test_jpx_debug.DataTests
python3 -m unittest -v test_jpx_debug.ModelTests
python3 -m unittest -v test_jpx_debug.TrainingTests
python3 -m unittest -v test_jpx_debug.PipelineTests
```

保留已正確的窗口邏輯：

```bash
python3 -m unittest -v test_jpx_debug.SequenceTests
```

全部重跑：

```bash
python3 -m unittest -v test_jpx_debug
```

這些測試設定了明確契約，例如目前仍支援 time_sequence=1、預設一次 train 跑完整一個 epoch。若你有意改變設計，要同步修改對應契約，而不是把測試通過當成唯一目的。測試通過不代表已完成真實資料驗證或樣本外研究。

## 先固定資料契約

| 交接點 | 期望內容 |
|---|---|
| create_time_sequence 的 X | list，每個元素是 T 列、F 欄的 DataFrame |
| create_time_sequence 的 y | list，每個元素是窗口最後一列的單一 Target |
| 轉換後整份 X | float32 Tensor，(N, T, F) |
| make_loader 送出的 X_batch | (B, T, F)，B 是當批樣本數 |
| 目前 CNN-LSTM 的輸出 | (B, 1) |
| 送入 loss 的 target | 必須與 prediction 同 shape；本輪採 (B, 1) |

N 是所有窗口總數，B 是 DataLoader 每次取出的窗口數；它们不是同一個概念。y 不需要為了「匹配 X」變成 (B,T,1)，要匹配的是模型輸出。

## 第一輪：只改 make_it_tensor

位置：`deep_learning.py:68–80`。

目前 pandas Series/DataFrame 可以轉換；NumPy array 被送去呼叫不存在的 `.to_numpy()`；Python list 則直接被送進只接受 NumPy array 的 `torch.from_numpy()`。你真正的 X 是 list[DataFrame]，y 是數值 list，所以實際流程正好會走到失敗路徑。

先寫出每種輸入應走哪一條分支，再動程式。所有分支在進入 `torch.from_numpy()` 前，應匯流成 `np.ndarray` 且 dtype 為 float32，並保留軸的順序。可以查 `np.asarray`；等長窗口也可以用 `np.stack` 明確堆疊。不要用 flatten 把 time 維消掉。

| 測試輸入 | 輸出 shape |
|---|---|
| pd.Series([1,2]) | (2,) |
| pd.DataFrame([[1,2],[3,4]]) | (2,2) |
| np.array([[1,2],[3,4]]) | (2,2) |
| [1,2] | (2,) |
| 兩個各有 2 天、2 個特徵的 DataFrame | (2,2,2) |

這一輪不需要改 create_time_sequence、不需要加入缺失值處理，也不用更動 forward。型別註記應使用 `torch.Tensor`，不是建構函式 `torch.tensor`。

## 第二輪：資料數值與檢查

### 2.1 成交量被另一個欄位覆蓋

位置：`data_processor.py:171–173`。

反例：所有 AdjustmentFactor 都是 1，Close=100、Volume=1000。原程式回傳 Volume=100。提示：for 迴圈結束後，`col_name` 的值仍然是最後一個欄位名稱。檢查成交量公式右側究竟讀了哪一欄。驗收：沒有拆併股時，OHLCV 都應維持原值，輸入 df 本身也不應被修改。

這個反例不依賴 JPX 因子生效日期，因此可先獨立修正。

### 2.2 index 標籤被當成位置

位置：`features.py:12–22`、`data_processor.py:99–101`。

check_series 回傳的是 Series.items() 的 index 標籤，不是保證從 0 開始的 row position。若 index=[10,20,30]，把 20 用來索引長度 3 的 list 會越界；若 index=[1,0]，甚至可能不報错卻留下錯誤資料列。

驗收：不連續 index、重新排序 index 都應刪掉真正無效的那一列。可選擇全程使用位置，或全程使用有索引的遮罩，但不要混用。

### 2.3 宣告要檢查的欄位，不等於真的寫了規則

位置：`data_processor.py:33–59`、`features.py:12–20`。

JPX_STOCK_PRICES_REQUIREMENTS 目前只有 SecuritiesCode、Date、Close、Volume、Target。即使傳入的欄位名包含 Open、High、Low、ExpectedDividend，也不會自動生出這些欄位的檢查。

合成案例中，Open=NaN 和 ExpectedDividend=NaN 都能留到下一階段。應在你指定的上游清理階段處理，不必在窗口函式重做所有檢查。ExpectedDividend 的缺失處理應區分資料語意；不能把「沒有當列紀錄」直接解讀成公司從不配息。

StandardScaler 不會補掉 NaN：fit 時略過，transform 後仍保留。標準化不是缺失值處理。

### 2.4 因子的數值有效性

位置：`data_processor.py:154–158`。

目前因子只檢查型別、缺失與有限性，0 和負數仍可能通過。因子第一天為 0 的反例會使後續調整價出現 inf。驗收：明確拒絕無效因子，或在有依據的政策下修復；不可讓非正因子直接累乘。

此外，把任何錯誤因子都補 1 是資料政策，不是必然正確的事實，可能抹去真實拆併股事件。

### 2.5 兩個小型問題

`features.py:29–30` 的 missing 推導式是在檢查「從清單取出的 n 不在同一份清單」，因此無法偵測缺欄。應比較需求欄位與實際 df 欄位；現在只是後面的 pandas 存取才拋 KeyError，而非你的自訂 ValueError。

`data_loader.py:25` 印出的是 `.tolist` 方法物件，沒有執行它。檢查方法呼叫與方法本身的差別。

## 第三輪：模型與 shape

### 3.1 num_features 不等於 len(df)

位置：`JPX_deep_learning_models.py:24–25`。

增加資料列不應改變單日特徵數。合成案例有 8 列、6 個 features，目前卻把 8 當成 in_channels。驗收：增加資料列或改變 batch size，不應改變 CNN 所需輸入通道數。

### 3.2 time_sequence=1 與 MaxPool1d(2) 不相容

位置：`deep_learning.py:15–29、47–50`，呼叫設定在 `JPX_deep_learning_models.py:21`。

實測 (3,1,6) 即使已是正確三維输入，仍會出現 pooling 輸出長度為 0 的 RuntimeError。你要決定保留 T=1 並調整 pooling，或有意改變輸入窗口與模型契約，不要靠任意 reshape 掩蓋。

在現有 stride=1、padding=0 的 convolution，接預設 MaxPool1d(2) 時，需要卷積後時間長度至少為 2。kernel_size=1 時即原始 T 至少為 2。

設計觀察，不等同於報錯：T=2 經 pooling 後 LSTM 只看到 1 個時間步。kernel_size=1 時，T=3 的預設 pooling 還會忽略最後一個時間點。更換窗口時，應同時檢查每層的時間長度與最後一天資訊是否保留。

### 3.3 head 的 Dropout1d 語意

位置：`deep_learning.py:31、60–62`。

CNN 後的三維資料 (B,C,T) 適用 channel dropout，但取最後時間步後成為 (B,H)。Dropout1d 會把這種二維資料當成無 batch 的 (C,L)，因此在你的用法相當於把某些樣本整列丟掉。

固定隨機種子的一次合成試驗中，64 列有 9 列整列歸零、沒有部分歸零的列。這不是固定會有 9 列，而是用來展示其整列遮罩語意。

重用同一個 dropout 物件本身不是錯；問題是「層的種類」與「輸入軸的意義」。測試假定 head 的意圖是常見的特徵 dropout，而不是刻意做整個樣本的 dropout。

### 3.4 prediction 與 y 不同 shape 的靜默錯誤

位置：`deep_learning.py:63、85–89、108–110、135–136`。

目前模型輸出 (B,1)，scalar-list 的 y 自然轉換後是 (B,)。若不在 loader 或 loss 前統一，L1Loss 會廣播，而不是逐樣本一一對應。

最小反例：prediction=[[1],[2]]、y=[1,2]。正確逐筆 MAE 是 0；直接傳入不同 shape 得到 0.5，因為實際比較成了 2×2。

驗收：訓練與評估都應在 loss 前確保 shape 完全相同。不要只修 train 而漏掉 evaluation。若用 squeeze，注意無參數 squeeze 在 B=1 時也可能拿掉 batch 軸。

## 第四輪：訓練與評估迴圈

位置：`deep_learning.py:99–120`。

train 的 return 仍縮排在 for 內，造成只處理第一個 batch；total_loss 也使用覆寫而非累加。這是兩個問題：只修其中一個不夠。

驗收案例：5 個樣本、batch_size=2，應執行 3 次 forward。用不更新權重的零預測模型與 target=[1,1,3,3,5]，整體 MAE 應為 2.6。原程式僅執行一次 forward 並回傳 1.0。

evaluation 在 y shape 已相同的案例中，已正確遍歷 3 個 batch、得到 2.6，且正確關閉梯度及進入 eval 模式。這段可用作你自己的參照。

補充契約：train 的 epoch 參數目前没有使用。可以定義函式只負責一個 epoch，外層控制輪數；也可以明確支援多輪，不要留下一個沒有作用的參數。空的訓練 loader 目前回傳 None，應像 evaluation 一樣提供清楚的錯誤。

## 第五輪：研究正確性（不能由這份單元測試宣告通過）

### 分割與標準化

位置：`JPX_deep_learning_models.py:13–33`、`features.py:48–55`。

目前整份資料先 fit_transform，並使用同一個 loader 訓練與 evaluation。因此輸出是訓練資料上的評估，並非樣本外表現。先定義依時間的訓練／驗證區間；scaler 只 fit 訓練部分，驗證資料用同一組參數 transform。窗口歸屬應依預測當日／窗口結束日決定，同時確認訓練標籤在訓練截止時間已可取得。

你的 mean/std 有回傳但目前沒有被使用。另注意常數欄位：scaler.var_**0.5 會是 0，而 StandardScaler 實際採用的 scale_ 對零變異欄位為 1；未來若自行拿回傳 std 去除，可能重新產生除以零。保留已 fit 的 scaler 通常更容易保持一致。

### 刪列、拆股與缺日

目前先 get_valid_rows，再 correct_prices_with_factors。若被刪掉的資料列恰好帶有拆併股因子，後續累乘可能漏掉事件。這是依流程判讀的風險，需用原始資料與日期查核。

find_duplicates 的結果目前被丟棄，沒有真正處理同股票同日期的重複列。刪除缺失日後，T 列也不一定等於 T 個相鄰交易日。要明確決定這些資料政策，不能只用 list shape 判定正確。

### 尚未核定的金融口徑

此檢查沒有確認因子究竟在哪個交易日生效、調整基準及 Target 的真實欄位語意，也沒有用實際拆併股事件對照。成交量覆寫 bug 與這些口徑無關，可以先修；口徑本身仍需另行核定。

### 記憶體

每一個重疊窗口都建立 DataFrame，再全部轉換成陣列，可能讓大資料集記憶體用量很高。先用少量股票及日期通過整條管線；若全量才卡住，檢查窗口數、每個窗口大小與物件數，不要直接把它當成模型運算慢。按需索引窗口的 Dataset 屬後續工程改善，不是第一輪要重寫的範圍。

## 低優先順序註記

- `main.py` 的入口呼叫未發現問題。
- `load_data` 的檔案存在及空資料檢查可保留；expanduser 只展開 ~，不代表一定轉成絕對路徑。
- `check_JPX_stock_price_csv` 回傳的是 (dict, DataFrame)，不是註記的兩個 DataFrame。
- `create_time_sequence` 實際需要多欄 DataFrame，Series 的型別註記不符合實作；y 是單一數值 list，不是多層 list。
- `normalize_data` 的 `pd.DataFrame | pd.DataFrame` 重複。
- `col_names=[]` 現在未被修改，尚未造成共享狀態錯誤；可改成更清楚的參數設計，但不要把它排在數值 bug 前面。
- time_sequence 的正整數檢查應在函式邊界或呼叫設定處有明確責任，不必反覆加進每一層。
- FEAUTRES 拼字不標準，但目前定義與引用一致，不是 NameError 根因。
- forward 每批印 shape 可在除錯後改成受控輸出；沒有 GPU 路徑不是目前 CPU 流程的錯誤。

## 修正後如何證明自己真的理解

每輪只說明「輸入契約是什麼、原程式在哪裡違反、為什麼反例會失敗」，再用另一個不同大小或順序的例子驗證。不是背誦修正後的那一行。完成本輪，不代表其他輪也已正確；完整測試通過，也不代表研究結論成立。

## 查核依據

程式定位以上傳原始版本行號為準；修正後行號可能變動。函式行為另對照 PyTorch 2.10 的 from_numpy、Conv1d、MaxPool1d、Dropout1d、L1Loss 官方文件，以及 pandas iloc 和 scikit-learn StandardScaler／Common pitfalls 官方文件。本報告的數字反例來自本次合成資料執行，而非真實 JPX 樣本。

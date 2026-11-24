"""
把 stock_price_final.zip.001、.002 … 合併、驗證並解壓成 stock_price_final.csv

用法（在任何資料夾執行皆可，輸出在本檔所在的 dataset/ 資料夾）：
    python join_dataset.py
"""
import glob
import hashlib
import os
import sys
import zipfile

for _s in (sys.stdout, sys.stderr):  # Windows cp950 主控台印 emoji 不會當掉
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(HERE, "stock_price_final.zip")
CSV_NAME = "stock_price_final.csv"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_sums():
    sums = {}
    with open(os.path.join(HERE, "SHA256SUMS.txt"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                digest, name = line.split(maxsplit=1)
                sums[name.strip()] = digest
    return sums


def main():
    sums = load_sums()
    parts = sorted(glob.glob(os.path.join(HERE, "stock_price_final.zip.[0-9][0-9][0-9]")))
    expected = sorted(n for n in sums if n.startswith("stock_price_final.zip."))
    if [os.path.basename(p) for p in parts] != expected:
        sys.exit(f"分割檔不齊全：需要 {expected}，找到 {[os.path.basename(p) for p in parts]}")

    print(f"合併 {len(parts)} 個分割檔...")
    with open(ZIP_PATH, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                out.write(f.read())
    if sha256(ZIP_PATH) != sums["stock_price_final.zip"]:
        sys.exit("❌ 合併後的 zip 校驗碼不符，分割檔可能下載不完整（Git LFS / 網路中斷），請重新下載")

    print("解壓縮...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extract(CSV_NAME, HERE)
    csv_path = os.path.join(HERE, CSV_NAME)
    if sha256(csv_path) != sums[CSV_NAME]:
        sys.exit("❌ 解壓後的 CSV 校驗碼不符")

    os.remove(ZIP_PATH)
    print(f"✅ 完成：{csv_path}（{os.path.getsize(csv_path) / 1048576:.0f} MB，校驗碼正確）")


if __name__ == "__main__":
    main()

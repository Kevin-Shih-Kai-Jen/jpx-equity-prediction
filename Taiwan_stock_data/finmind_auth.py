"""
FinMind Token 共用工具：StocksSymbol.py、03、04 都從這裡讀 token。

讀取順序：
    1. 與本程式同一個資料夾的 finmind_token.txt（不論從哪個目錄執行都找得到）
    2. 環境變數 FINMIND_TOKEN

容錯：
    - Windows 記事本存的 UTF-8 BOM、PowerShell `echo ... > finmind_token.txt` 產生的 UTF-16 檔
    - 前後空白、換行、引號，或不小心貼上 "Bearer " / "token=" 前綴
"""
import os
import sys

import requests

TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finmind_token.txt")
USER_INFO_URL = "https://api.web.finmindtrade.com/v2/user_info"


def _read_text(path):
    raw = open(path, "rb").read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="ignore")


def _clean(token):
    token = token.replace("﻿", "").strip().strip('"').strip("'").strip()
    for prefix in ("Bearer ", "bearer ", "token=", "TOKEN="):
        if token.startswith(prefix):
            token = token[len(prefix):].strip()
    # 只取第一個非空白行，避免多貼了其他內容
    lines = [ln.strip() for ln in token.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def load_finmind_token():
    if os.path.exists(TOKEN_FILE):
        token = _clean(_read_text(TOKEN_FILE))
        if token:
            return token
    return _clean(os.getenv("FINMIND_TOKEN", ""))


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def check_finmind_token(token, verbose=True):
    """確認 token 可用並印出額度。token 無效時回傳 False；網路問題只警告、不擋住執行。"""
    if not token:
        print(f"❌ 找不到 FinMind Token：請把 token 貼進 {TOKEN_FILE}（只放 token 本身一行）")
        return False
    try:
        resp = requests.get(USER_INFO_URL, headers=auth_headers(token), timeout=10)
    except Exception as e:
        print(f"⚠️ 無法連線查詢額度（將繼續嘗試）：{e}")
        return True

    try:
        info = resp.json()
    except ValueError:
        info = {}

    if resp.status_code in (401, 403) or "api_request_limit" not in info:
        masked = f"{token[:6]}…{token[-4:]}" if len(token) > 12 else "(太短)"
        print(f"❌ FinMind Token 無效（HTTP {resp.status_code}，讀到的 token：{masked}，長度 {len(token)}）")
        print(f"   伺服器回應：{str(info.get('msg', resp.text))[:150]}")
        print("   請到 FinMind 網站的使用者資訊頁重新複製 token；若曾重設 token，舊的會立即失效。")
        return False

    if verbose:
        print(f"✅ FinMind Token 有效，目前已用 {info.get('user_count')} / {info.get('api_request_limit')} 次")
    return True


if __name__ == "__main__":
    # 單獨執行可快速測試 token：python finmind_auth.py
    ok = check_finmind_token(load_finmind_token())
    sys.exit(0 if ok else 1)

import os
import json
import math
import time
import requests
import re
import random
from datetime import datetime, date
from zoneinfo import ZoneInfo
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

import gspread
from google.oauth2.service_account import Credentials

import jpholiday


JST = ZoneInfo("Asia/Tokyo")

# ==========================================
# ★ 設定: B/C列のスクレイピング・入力切替
# True : スクレイピングを行い、B列(銘柄名)・C列(業種)からL列までを更新 (低速)
# False: スクレイピングを行わず、D列(判定結果)からL列までを更新 (高速)
# ==========================================
UPDATE_BC_WITH_SCRAPING = False

BREAKOUT_LOOKBACK = 20
BREAKOUT_MAX_EXTENSION = 0.05

RS_LOOKBACKS = {
    63: 0.40,
    126: 0.20,
    189: 0.20,
    252: 0.20,
}
RS_PERCENTILE_THRESHOLD = 70.0
MIN_RS_UNIVERSE_SIZE = 20

VOLUME_LOOKBACK = 50
BREAKOUT_VOLUME_RATIO = 1.5
VOLUME_DRY_LOOKBACK = 10
VOLUME_DRY_RATIO = 0.80

ATR_PERIOD = 10
ATR_BASELINE_LOOKBACK = 60
ATR_CONTRACTION_RATIO = 0.80


# ==========================================
# 1. スクレイピング & 設定ロジック (from buhin.py)
# ==========================================

# 東証33業種リスト
TSE_SECTORS = [
    "水産・農林業", "鉱業", "建設業", "食料品", "繊維製品", "パルプ・紙", "化学",
    "医薬品", "石油・石炭製品", "ゴム製品", "ガラス・土石製品", "鉄鋼", "非鉄金属",
    "金属製品", "機械", "電気機器", "輸送用機器", "精密機器", "その他製品",
    "電気・ガス業", "陸運業", "海運業", "空運業", "倉庫・運輸関連業", "情報・通信業",
    "卸売業", "小売業", "銀行業", "証券、商品先物取引業", "保険業",
    "その他金融業", "不動産業", "サービス業"
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
]

def create_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session

_HTTP_SESSION = create_session()

def get_value(df, keys, date_col):
    """財務データDataFrameから特定の日付・キーの値を取得する"""
    if df.empty or date_col is None or date_col not in df.columns:
        return 0
    for key in keys:
        if key in df.index:
            val = df.loc[key, date_col]
            return val if not pd.isna(val) else 0
    return 0

def get_japanese_name_and_sector(ticker_code):
    """Yahoo!ファイナンス(日本)から銘柄名と業種を取得"""
    code_only = ticker_code.replace(".T", "")
    url = f"https://finance.yahoo.co.jp/quote/{code_only}.T"
    headers = {"User-Agent": random.choice(USER_AGENTS)}

    try:
        time.sleep(random.uniform(2.0, 4.0))

        res = _HTTP_SESSION.get(url, headers=headers, timeout=10)
        res.encoding = res.apparent_encoding
        html = res.text

        name = None
        match = re.search(r'<title>(.*?)【', html)
        if match:
            name = match.group(1).strip()

        sector = "-"
        for candidate in TSE_SECTORS:
            if candidate in html:
                sector = candidate
                break

        if not name:
            name = str(ticker_code)

        return name, sector
    except Exception as e:
        print(f"Scraping warning: {e}")
        return str(ticker_code), "-"


# ----------------------------
# Config / Secrets
# ----------------------------
def load_app_config() -> dict:
    raw = os.environ.get("APP_CONFIG_JSON", "").strip()
    if not raw:
        raise RuntimeError("APP_CONFIG_JSON is empty. Set GitHub Secret 'APP_CONFIG_JSON'.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"APP_CONFIG_JSON is not valid JSON: {e}") from e


def get_today_jst() -> date:
    return datetime.now(JST).date()


# ----------------------------
# Market calendar filters
# ----------------------------
def is_skip_day(d: date) -> bool:
    if d.weekday() >= 5:
        return True
    if (d.month == 12 and d.day == 31) or (d.month == 1 and d.day in (1, 2, 3)):
        return True
    if jpholiday.is_holiday(d):
        return True
    return False


# ----------------------------
# Google Sheets
# ----------------------------
def open_worksheet(cfg: dict):
    sa_info = cfg["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)

    gc = gspread.authorize(creds)
    sh = gc.open_by_url(cfg["sheet_url"])

    ws_name = (cfg.get("worksheet_name") or "").strip()
    if not ws_name:
        raise RuntimeError("worksheet_name is required in APP_CONFIG_JSON (do not hardcode in main.py).")

    ws = sh.worksheet(ws_name)
    return ws


def read_tickers_from_sheet(ws) -> list:
    """
    A列(コード), B列(銘柄名), C列(業種) を取得する。
    戻り値は [ [code, name, sector], ... ] のリスト
    """
    raw_rows = ws.get("A2:C")

    if not raw_rows:
        return []

    data_list = []
    seen = set()

    for row in raw_rows:
        if not row:
            continue

        code = (row[0] or "").strip()
        name = (row[1] or "").strip() if len(row) > 1 else ""
        sector = (row[2] or "").strip() if len(row) > 2 else ""

        if not code:
            continue

        if "DELISTED" in name or "廃止" in name:
            continue

        if code not in seen:
            seen.add(code)
            data_list.append([code, name, sector])

    return data_list


def write_output_batch(ws, rows: list[list], start_row: int):
    """
    設定に応じて書き込み範囲を変更
    UPDATE_BC_WITH_SCRAPING is True  => B列〜L列 = 11列
    UPDATE_BC_WITH_SCRAPING is False => D列〜L列 = 9列
    """
    if not rows:
        return
    end_row = start_row + len(rows) - 1

    if UPDATE_BC_WITH_SCRAPING:
        range_name = f"B{start_row}:L{end_row}"
    else:
        range_name = f"D{start_row}:L{end_row}"

    ws.update(range_name=range_name, values=rows)


def write_output_headers(ws):
    ws.update(
        range_name="D1:L1",
        values=[[
            "判定",
            "終値",
            "ピボット価格",
            "50日線判定",
            "200日線判定",
            "RS順位(%)",
            "VCP",
            "出来高倍率",
            "ATR収縮",
        ]],
    )


# ----------------------------
# Finance helpers
# ----------------------------
def safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def slope_positive(series: pd.Series, lookback: int = 20) -> bool:
    s = series.dropna()
    if len(s) < lookback + 1:
        return False
    y = s.iloc[-lookback:].values
    x = np.arange(len(y))
    a = np.polyfit(x, y, 1)[0]
    return a > 0


def compute_cagr_from_series(values: list[float]) -> float | None:
    vals = [safe_float(v) for v in values]
    if any(v is None for v in vals):
        return None
    if len(vals) < 2:
        return None
    old = vals[0]
    new = vals[-1]
    years = len(vals) - 1
    if old is None or new is None or old <= 0 or new <= 0:
        return None
    return (new / old) ** (1 / years) - 1


def pick_row(financials: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    if financials is None or financials.empty:
        return None
    for k in candidates:
        if k in financials.index:
            return financials.loc[k]
    return None


def annual_points_last_4(series_recent_to_old: pd.Series) -> list[float] | None:
    if series_recent_to_old is None:
        return None
    s = series_recent_to_old.dropna()
    if len(s) < 2:
        return None
    s = s.iloc[:4]
    vals = list(reversed(s.values.tolist()))
    return vals


def format_bool_mark(b: bool) -> str:
    return "○" if b else "×"


def format_date(d: date | None) -> str:
    if d is None:
        return ""
    return d.isoformat()


def parse_earnings_date_from_calendar(cal) -> date | None:
    if cal is None:
        return None
    try:
        for key in ["Earnings Date", "EarningsDate", "earningsDate"]:
            if key in cal:
                v = cal[key]
                if isinstance(v, (list, tuple, np.ndarray, pd.Series, pd.Index)):
                    if len(v) == 0:
                        continue
                    v0 = v[0]
                else:
                    v0 = v

                ts = pd.to_datetime(v0, errors="coerce")

                if isinstance(ts, (pd.Index, pd.Series, np.ndarray)):
                    if len(ts) == 0:
                        return None
                    ts = ts[0]

                if pd.isna(ts):
                    return None
                return ts.date()
    except Exception:
        return None
    return None


# ----------------------------
# Core analysis (Parallelized)
# ----------------------------

def get_series_value_on_or_before(series: pd.Series, target_date) -> float | None:
    if series is None or series.empty:
        return None
    try:
        eligible = series.loc[:target_date].dropna()
        if eligible.empty:
            return None
        return safe_float(eligible.iloc[-1])
    except Exception:
        return None


def process_single_ticker(ticker_data_tuple, d, idx_close):
    """
    1銘柄分の基礎指標を計算する関数（並列処理用）
    ticker_data_tuple: (t_raw, pre_name, pre_sector) のタプル
    d: 株価DataFrame
    idx_close: 指数Closeデータ
    """
    t_raw, pre_name, pre_sector = ticker_data_tuple

    if UPDATE_BC_WITH_SCRAPING:
        time.sleep(random.uniform(3.0, 5.0))

    api_t = f"{t_raw}.T" if str(t_raw).isdigit() else t_raw

    def make_result_row(msg_list):
        if UPDATE_BC_WITH_SCRAPING:
            return [pre_name, pre_sector] + msg_list
        return msg_list

    close = pd.Series(dtype=float)
    high = pd.Series(dtype=float)
    low = pd.Series(dtype=float)
    volume = pd.Series(dtype=float)

    if d is not None and not d.empty:
        close = d["Close"].dropna()
        high = d["High"].dropna()
        low = d["Low"].dropna()
        volume = d["Volume"].dropna()

    if close.empty or len(close) < 1:
        return {"final_row": make_result_row(["取得失敗(株価なし)"] + [""] * 8)}

    if len(close) < 260:
        return {"final_row": make_result_row(["データ不足"] + [""] * 8)}

    last_close = float(close.iloc[-1])

    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    ma50_last = safe_float(ma50.iloc[-1])
    ma150_last = safe_float(ma150.iloc[-1])
    ma200_last = safe_float(ma200.iloc[-1])

    trend_ok = (
        (ma50_last is not None)
        and (ma150_last is not None)
        and (ma200_last is not None)
        and (last_close > ma50_last > ma150_last > ma200_last)
        and slope_positive(ma50, lookback=20)
        and slope_positive(ma200, lookback=20)
    )

    high_52w = safe_float(high.iloc[-252:].max()) if len(high) >= 252 else None
    low_52w = safe_float(low.iloc[-252:].min()) if len(low) >= 252 else None
    hl_ok = (
        high_52w is not None
        and low_52w is not None
        and (last_close > low_52w * 1.30)
        and (last_close >= high_52w * 0.95)
    )

    trend_pass = trend_ok and hl_ok

    rs_score = None
    rs_12m = None
    if not idx_close.empty and len(idx_close) > max(RS_LOOKBACKS):
        base_date = idx_close.index[-1]
        index_now = safe_float(idx_close.iloc[-1])
        ticker_now = get_series_value_on_or_before(close, base_date)
        rs_values = {}

        if index_now is not None and ticker_now is not None and index_now > 0 and ticker_now > 0:
            for lookback, weight in RS_LOOKBACKS.items():
                target_date = idx_close.index[-(lookback + 1)]
                index_past = safe_float(idx_close.iloc[-(lookback + 1)])
                ticker_past = get_series_value_on_or_before(close, target_date)

                if (
                    index_past is not None
                    and ticker_past is not None
                    and index_past > 0
                    and ticker_past > 0
                ):
                    rs_values[lookback] = (
                        (ticker_now / ticker_past) / (index_now / index_past)
                    ) - 1

            rs_12m = rs_values.get(252)
            if len(rs_values) == len(RS_LOOKBACKS):
                rs_score = sum(
                    rs_values[lookback] * weight
                    for lookback, weight in RS_LOOKBACKS.items()
                )

    pivot_price = None
    breakout_today = False
    if len(high) >= BREAKOUT_LOOKBACK + 1 and len(close) >= 2:
        pivot_price = safe_float(
            high.shift(1).rolling(BREAKOUT_LOOKBACK).max().iloc[-1]
        )
        previous_close = safe_float(close.iloc[-2])
        if pivot_price is not None and pivot_price > 0 and previous_close is not None:
            breakout_extension = (last_close / pivot_price) - 1
            breakout_today = (
                previous_close <= pivot_price
                and last_close > pivot_price
                and breakout_extension <= BREAKOUT_MAX_EXTENSION
            )

    volume_ratio = None
    volume_ok = False
    volume_dry_ok = False
    if volume.notna().sum() >= VOLUME_LOOKBACK + 1:
        avg_volume_50 = safe_float(
            volume.shift(1).rolling(VOLUME_LOOKBACK).mean().iloc[-1]
        )
        latest_volume = safe_float(volume.iloc[-1])
        pre_breakout_volume_10 = safe_float(
            volume.shift(1).rolling(VOLUME_DRY_LOOKBACK).mean().iloc[-1]
        )

        if avg_volume_50 is not None and avg_volume_50 > 0:
            if latest_volume is not None:
                volume_ratio = latest_volume / avg_volume_50
                volume_ok = volume_ratio >= BREAKOUT_VOLUME_RATIO

            if pre_breakout_volume_10 is not None:
                volume_dry_ok = (
                    pre_breakout_volume_10
                    <= avg_volume_50 * VOLUME_DRY_RATIO
                )

    atr_contraction_ok = False
    if not high.empty and not low.empty:
        true_range = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr10 = true_range.rolling(ATR_PERIOD).mean()
        normalized_atr10 = atr10 / close

        latest_normalized_atr = safe_float(normalized_atr10.iloc[-1])
        atr_baseline = safe_float(
            normalized_atr10.shift(1)
            .rolling(ATR_BASELINE_LOOKBACK)
            .mean()
            .iloc[-1]
        )

        if (
            latest_normalized_atr is not None
            and atr_baseline is not None
            and atr_baseline > 0
        ):
            atr_contraction_ok = (
                latest_normalized_atr
                <= atr_baseline * ATR_CONTRACTION_RATIO
            )

    vcp_ok = volume_dry_ok and atr_contraction_ok

    stock_name = pre_name
    industry = pre_sector
    try:
        if UPDATE_BC_WITH_SCRAPING:
            stock_name, industry = get_japanese_name_and_sector(api_t)
    except Exception as e:
        print(f"Error analyzing (masked): {e}")

    return {
        "final_row": None,
        "stock_name": stock_name,
        "industry": industry,
        "trend_pass": trend_pass,
        "breakout_today": breakout_today,
        "last_close": last_close,
        "pivot_price": pivot_price,
        "ma50_mark": "○" if (
            ma50_last is not None
            and last_close > ma50_last
            and slope_positive(ma50, 20)
        ) else "×",
        "ma200_text": (
            f"{ma200_last:.0f} (上向き)"
            if ma200_last is not None and slope_positive(ma200, 20)
            else (
                f"{ma200_last:.0f} (横/下)"
                if ma200_last is not None
                else ""
            )
        ),
        "rs_score": rs_score,
        "rs_12m": rs_12m,
        "rs_percentile": None,
        "volume_ratio": volume_ratio,
        "volume_ok": volume_ok,
        "vcp_ok": vcp_ok,
        "atr_contraction_ok": atr_contraction_ok,
    }


def finalize_results(results: list[dict]) -> list[list]:
    valid_indexes = [
        i
        for i, result in enumerate(results)
        if result.get("final_row") is None
        and result.get("rs_score") is not None
    ]

    if valid_indexes:
        rs_series = pd.Series(
            [results[i]["rs_score"] for i in valid_indexes],
            index=valid_indexes,
            dtype=float,
        )
        rs_percentiles = rs_series.rank(method="average", pct=True) * 100
        for i in valid_indexes:
            results[i]["rs_percentile"] = safe_float(rs_percentiles.loc[i])

    use_percentile_ranking = len(valid_indexes) >= MIN_RS_UNIVERSE_SIZE
    output_rows = []

    for result in results:
        if result.get("final_row") is not None:
            output_rows.append(result["final_row"])
            continue

        if use_percentile_ranking:
            rs_ok = (
                result["rs_percentile"] is not None
                and result["rs_percentile"] >= RS_PERCENTILE_THRESHOLD
            )
        else:
            rs_ok = (
                result["rs_12m"] is not None
                and result["rs_12m"] > 0
            )

        if (
            result["trend_pass"]
            and rs_ok
            and result["breakout_today"]
            and result["volume_ok"]
            and result["vcp_ok"]
        ):
            verdict = "合格"
        elif result["trend_pass"] and rs_ok:
            verdict = "監視"
        else:
            verdict = "除外"

        analysis_row = [
            verdict,
            round(result["last_close"], 2),
            round(result["pivot_price"], 2)
            if result["pivot_price"] is not None
            else "",
            result["ma50_mark"],
            result["ma200_text"],
            round(result["rs_percentile"], 1)
            if result["rs_percentile"] is not None
            else "",
            format_bool_mark(result["vcp_ok"]),
            round(result["volume_ratio"], 2)
            if result["volume_ratio"] is not None
            else "",
            format_bool_mark(result["atr_contraction_ok"]),
        ]

        if UPDATE_BC_WITH_SCRAPING:
            output_rows.append(
                [result["stock_name"], result["industry"]] + analysis_row
            )
        else:
            output_rows.append(analysis_row)

    return output_rows

def main():
    cfg = load_app_config()
    today = get_today_jst()

    if is_skip_day(today):
        print(f"[SKIP] {today.isoformat()} (weekend/holiday/year-end)")
        return

    ws = open_worksheet(cfg)
    tickers_data = read_tickers_from_sheet(ws)
    if not tickers_data:
        print("No tickers in sheet.")
        return

    index_ticker = cfg.get("index_ticker", "^TOPX")

    idx_close = pd.Series(dtype=float)
    try:
        df_idx = yf.download(index_ticker, period="2y", interval="1d", auto_adjust=False, progress=False)
        if not df_idx.empty:
            if isinstance(df_idx.columns, pd.MultiIndex):
                idx_close = df_idx["Close"][index_ticker].dropna() if index_ticker in df_idx.columns.get_level_values(0) else df_idx["Close"].iloc[:, 0].dropna()
            else:
                idx_close = df_idx["Close"].dropna()
    except Exception as e:
        print(f"Index download error: {e}")

    BATCH_SIZE = 50
    total_tickers = len(tickers_data)
    current_index = 0
    all_results = []

    print(f"Total Tickers: {total_tickers}")

    while current_index < total_tickers:
        end_index = min(current_index + BATCH_SIZE, total_tickers)
        batch_tickers_tuples = tickers_data[current_index:end_index]

        print(f"Processing batch: {current_index + 1} - {end_index} / {total_tickers}")

        def to_api_ticker(t):
            return f"{t}.T" if str(t).isdigit() else t

        batch_tickers_api = [to_api_ticker(t[0]) for t in batch_tickers_tuples]

        batch_data = {}
        try:
            df_p = yf.download(
                batch_tickers_api,
                period="2y",
                interval="1d",
                group_by='ticker',
                auto_adjust=False,
                threads=True,
                progress=False
            )

            if isinstance(df_p.columns, pd.MultiIndex):
                for t_api in batch_tickers_api:
                    if t_api in df_p.columns.get_level_values(0):
                        batch_data[t_api] = df_p[t_api].dropna(how="all")
            else:
                if len(batch_tickers_api) == 1:
                    batch_data[batch_tickers_api[0]] = df_p.dropna(how="all")
        except Exception as e:
            print(f"Batch download error: {e}")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for t_tuple in batch_tickers_tuples:
                t_raw = t_tuple[0]
                t_api = to_api_ticker(t_raw)
                d = batch_data.get(t_api)
                futures.append(executor.submit(process_single_ticker, t_tuple, d, idx_close))

            for future in futures:
                try:
                    all_results.append(future.result())
                except Exception as e:
                    print(f"Future result error: {e}")
                    if UPDATE_BC_WITH_SCRAPING:
                        all_results.append({
                            "final_row": ["Error", "Error", "Error"] + [""] * 8
                        })
                    else:
                        all_results.append({
                            "final_row": ["Error"] + [""] * 8
                        })

        current_index += BATCH_SIZE
        if current_index < total_tickers:
            time.sleep(15)

    output_rows = finalize_results(all_results)

    try:
        write_output_headers(ws)
    except Exception as e:
        print(f"Sheet header write error: {e}")

    current_index = 0
    while current_index < total_tickers:
        end_index = min(current_index + BATCH_SIZE, total_tickers)
        batch_rows = output_rows[current_index:end_index]

        # 1行目は固定ヘッダー用。データは2行目から出力する。
        start_write_row = current_index + 2
        try:
            write_output_batch(ws, batch_rows, start_write_row)
        except Exception as e:
            print(f"Sheet write error at batch index {current_index}: {e}")

        current_index += BATCH_SIZE

    print("[OK] All batches processed.")


if __name__ == "__main__":
    main()

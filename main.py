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
    # .T を除去してURL作成
    code_only = ticker_code.replace(".T", "")
    url = f"https://finance.yahoo.co.jp/quote/{code_only}.T"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        time.sleep(random.uniform(1.0, 2.0)) # Access block avoidance
        
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
        
        # fallback
        if not name:
            name = str(ticker_code)
            
        return name, sector
    except Exception as e:
        print(f"Scraping warning {ticker_code}: {e}")
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
    # Weekend
    if d.weekday() >= 5:
        return True
    # Year-end / New-year skip
    if (d.month == 12 and d.day == 31) or (d.month == 1 and d.day in (1, 2, 3)):
        return True
    # JP holiday
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


def read_tickers_from_sheet(ws) -> list[str]:
    # A2 downwards, allow blanks
    col = ws.col_values(1)  # A column
    if len(col) <= 1:
        return []
    raw = col[1:]  # from A2
    tickers = []
    for s in raw:
        s = (s or "").strip()
        if not s:
            continue
        # normalize: append .T if not already has suffix
        if "." not in s:
            s = f"{s}.T"
        tickers.append(s)
    # de-dup while preserving order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def write_output(ws, headers: list[str], rows: list[list]):
    # Write headers row 1 and data from row 2
    ws.update(range_name="A1", values=[headers])
    if rows:
        ws.update(range_name="A2", values=rows)
    else:
        # Clear old data area minimally (leave headers)
        ws.batch_clear(["A2:Z"])


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

def process_single_ticker(t, d, idx_close):
    """
    1銘柄分の分析を実行する関数（並列処理用）
    """
    # 取得失敗チェック
    if d is None or d.empty:
        return [t, "", "", "取得失敗"] + [""] * 19

    close = d["Close"].dropna()
    high = d["High"].dropna()

    # データ不足チェック
    if len(close) < 260 or (not idx_close.empty and len(idx_close) < 260):
        if len(close) < 260:
            return [t, "", "", "データ不足"] + [""] * 19
        else:
            return [t, "", "", "指数データ不足"] + [""] * 19

    last_close = float(close.iloc[-1])

    ma75 = close.rolling(75).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    trend_ok = (
        (safe_float(ma75.iloc[-1]) is not None)
        and (safe_float(ma150.iloc[-1]) is not None)
        and (safe_float(ma200.iloc[-1]) is not None)
        and (last_close > float(ma75.iloc[-1]) > float(ma150.iloc[-1]) > float(ma200.iloc[-1]))
        and slope_positive(ma75, lookback=20)
    )

    close_252 = close.iloc[-252:]
    low_52w = float(close_252.min())
    high_52w = float(close_252.max())
    hl_ok = (last_close > low_52w * 1.30) and (last_close > high_52w * 0.75)

    trend_pass = trend_ok and hl_ok

    t_now = float(close.iloc[-1])
    t_1y = safe_float(close.shift(252).iloc[-1])
    
    rs_ratio = None
    rs_ok = False

    if not idx_close.empty:
        i_now = float(idx_close.iloc[-1])
        i_1y = safe_float(idx_close.shift(252).iloc[-1])
        if t_1y and i_1y and t_1y > 0 and i_1y > 0:
            rs_ratio = (t_now / t_1y) / (i_now / i_1y)
            rs_ok = rs_ratio > 1.0

    std10 = close.rolling(10).std()
    vcp_hint = False
    if std10.notna().sum() >= 70:
        recent_std10 = safe_float(std10.iloc[-1])
        mean_std10_60 = safe_float(std10.iloc[-60:].mean())
        if recent_std10 is not None and mean_std10_60 is not None:
            vcp_hint = recent_std10 < mean_std10_60

    rets = close.pct_change()
    vol20 = rets.rolling(20).std()
    vol_high = False
    if vol20.notna().sum() >= 90:
        recent_vol20 = safe_float(vol20.iloc[-1])
        mean_vol20_60 = safe_float(vol20.iloc[-60:].mean())
        if recent_vol20 is not None and mean_vol20_60 is not None:
            vol_high = recent_vol20 > mean_vol20_60

    buy_timing_price = None
    hi20 = safe_float(high.iloc[-20:].max()) if len(high) >= 20 else None
    if hi20 is not None and last_close >= hi20 * 0.95:
        buy_timing_price = hi20

    # --- buhin.py logic integration starts here ---
    
    # Initialize placeholders
    stock_name = ""
    industry = "-"
    fair_value = None
    divergence = None
    op_cagr_val = None
    ord_cagr_val = None
    eps_cagr_val = None
    uprev = ""
    earnings_date = None
    alert = ""

    try:
        # 1. Scraping Name/Sector (buhin.py style)
        stock_name, industry = get_japanese_name_and_sector(t)

        tk = yf.Ticker(t)
        
        # 2. Calendar / Alert
        try:
            cal = tk.calendar
            earnings_date = parse_earnings_date_from_calendar(cal)
        except Exception:
            earnings_date = None

        if earnings_date is not None:
            days = (earnings_date - get_today_jst()).days
            if 0 <= days <= 30:
                alert = "⚠️1ヶ月以内"
        
        # 3. Financials (buhin.py robust logic)
        financials = tk.financials
        balance_sheet = tk.balance_sheet
        info = tk.info or {}
        
        # Basic info extraction (fallback for name)
        if stock_name == t or stock_name == "":
            stock_name = info.get('longName', t)

        # If no financials, skip calculation but keep row
        if not financials.empty and not balance_sheet.empty:
            latest_date_bs = balance_sheet.columns[0]
            latest_date_pl = financials.columns[0]
            
            # --- ROIC Logic ---
            op_income = get_value(financials, ['Operating Income', 'Operating Profit'], latest_date_pl)
            nopat = op_income * 0.7
            
            interest_bearing_debt = get_value(balance_sheet, ['Total Debt'], latest_date_bs)
            if interest_bearing_debt == 0:
                short_debt = get_value(balance_sheet, ['Current Debt', 'Short Long Term Debt'], latest_date_bs)
                long_debt = get_value(balance_sheet, ['Long Term Debt'], latest_date_bs)
                interest_bearing_debt = short_debt + long_debt
            
            equity = get_value(balance_sheet, ['Total Stockholder Equity', 'Total Equity', 'Stockholders Equity'], latest_date_bs)
            forex_adj = get_value(balance_sheet, ['Foreign Currency Translation Adjustments', 'Accumulated Other Comprehensive Income'], latest_date_bs)
            adjusted_equity = equity - forex_adj
            if adjusted_equity <= 0: adjusted_equity = equity
            
            invested_capital = interest_bearing_debt + adjusted_equity
            roic = (nopat / invested_capital * 100) if invested_capital > 0 else 0

            # --- CAGR Logic ---
            growth_dates = financials.columns[:4]
            final_cagr = 0
            
            # Helper for EPS history for display
            eps_row = pick_row(financials, ["Basic EPS", "Diluted EPS"])
            eps_vals_display = annual_points_last_4(eps_row) if eps_row is not None else None
            if eps_vals_display and len(eps_vals_display) >= 2:
                eps_cagr_val = compute_cagr_from_series(eps_vals_display)

            # Helper for OP/Ord history for display
            op_row_disp = pick_row(financials, ["Operating Income"])
            op_vals_disp = annual_points_last_4(op_row_disp) if op_row_disp is not None else None
            if op_vals_disp and len(op_vals_disp) >= 2:
                op_cagr_val = compute_cagr_from_series(op_vals_disp)

            ord_row = pick_row(financials, ["Pretax Income", "Income Before Tax"])
            ord_vals = annual_points_last_4(ord_row) if ord_row is not None else None
            if ord_vals and len(ord_vals) >= 2:
                ord_cagr_val = compute_cagr_from_series(ord_vals)

            # CAGR calculation for Valuation (buhin.py logic)
            if len(growth_dates) >= 2:
                latest_d = growth_dates[0]
                oldest_d = growth_dates[-1]
                years_span = len(growth_dates) - 1
                
                def get_eps_val(d):
                    val = get_value(financials, ['Basic EPS'], d)
                    if val != 0: return val
                    ni = get_value(financials, ['Net Income', 'Net Income Common Stockholders'], d)
                    shares = get_value(financials, ['Basic Average Shares'], d)
                    return ni / shares if (ni != 0 and shares > 0) else 0

                eps_latest = get_eps_val(latest_d)
                eps_oldest = get_eps_val(oldest_d)
                
                cagr_eps_internal = 0
                if eps_oldest > 0 and eps_latest > 0:
                    cagr_eps_internal = ((eps_latest / eps_oldest)**(1 / years_span) - 1) * 100
                
                op_latest = get_value(financials, ['Operating Income'], latest_d)
                op_oldest = get_value(financials, ['Operating Income'], oldest_d)
                cagr_op_internal = 0
                if op_oldest > 0 and op_latest > 0:
                    cagr_op_internal = ((op_latest / op_oldest)**(1 / years_span) - 1) * 100
                
                if cagr_eps_internal != 0 and cagr_op_internal != 0:
                    final_cagr = min(cagr_eps_internal, cagr_op_internal)
                elif cagr_eps_internal != 0:
                    final_cagr = cagr_eps_internal
                elif cagr_op_internal != 0:
                    final_cagr = cagr_op_internal

            # --- Target Price Logic ---
            # Base PER selection
            calc_target_per = 15
            if final_cagr > 20: calc_target_per = 25
            elif final_cagr > 10: calc_target_per = 20
            elif final_cagr < 0: calc_target_per = 10
            
            # ROIC adjustment
            if roic > 15: calc_target_per *= 1.15
            elif roic >= 10: calc_target_per *= 1.05
            elif roic < 5: calc_target_per *= 0.9
            
            # Cap
            if calc_target_per > 25: calc_target_per = 25
            
            per_info = info.get('trailingPE', 0)
            # Use current price from yf.download (last_close) for consistency
            current_eps = last_close / per_info if (per_info and per_info > 0) else 0
            
            fair_value = current_eps * calc_target_per
            if fair_value > 0 and last_close > 0:
                divergence = fair_value / last_close - 1
            else:
                divergence = None

        # Uprev check
        forward_eps = safe_float(info.get("forwardEps"))
        trailing_eps = safe_float(info.get("trailingEps"))
        if forward_eps is not None and trailing_eps is not None and trailing_eps > 0:
            if forward_eps > trailing_eps * 1.10:
                uprev = "あり"
            else:
                uprev = "なし"
        
    except Exception as e:
        print(f"Error analyzing {t}: {e}")
        pass

    if trend_pass and rs_ok:
        verdict = "合格"
    elif trend_pass or rs_ok:
        verdict = "監視"
    else:
        verdict = "除外"

    allocation = "Half" if (alert or vol_high) else "Full"
    target_sell = last_close * 1.14

    return [
        t.replace(".T", ""),
        stock_name,
        industry,
        verdict,
        round(last_close, 2),
        round(fair_value, 2) if fair_value is not None else "",
        divergence if divergence is not None else "",
        round(buy_timing_price, 2) if buy_timing_price is not None else "",
        round(target_sell, 2),
        "○" if (last_close > float(ma75.iloc[-1]) and slope_positive(ma75, 20)) else "×",
        f"{float(ma200.iloc[-1]):.0f} (上向き)" if slope_positive(ma200, 20) else f"{float(ma200.iloc[-1]):.0f} (横/下)",
        round(rs_ratio, 3) if rs_ratio is not None else "",
        format_bool_mark(vcp_hint),
        op_cagr_val if op_cagr_val is not None else "",
        ord_cagr_val if ord_cagr_val is not None else "",
        eps_cagr_val if eps_cagr_val is not None else "",
        uprev,
        format_date(earnings_date),
        alert,
        format_bool_mark(vol_high),
        allocation,
    ]

def analyze_universe(
    tickers: list[str],
    index_ticker: str,
    target_per: float,
) -> list[list]:
    if not tickers:
        return []

    all_symbols = tickers + [index_ticker]

    # threads=True に変更 (buhin.py参照)
    df = yf.download(
        tickers=all_symbols,
        period="2y",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True, 
        progress=False,
    )

    data = {}
    if isinstance(df.columns, pd.MultiIndex):
        for sym in all_symbols:
            if sym in df.columns.get_level_values(0):
                sub = df[sym].dropna(how="all")
                data[sym] = sub
    else:
        if len(all_symbols) == 1:
             data[all_symbols[0]] = df.dropna(how="all")
        else:
             pass

    if index_ticker not in data or data[index_ticker].empty:
        print(f"Warning: Index ticker {index_ticker} data missing. RS calculation will fail.")
        idx_close = pd.Series(dtype=float)
    else:
        idx_close = data[index_ticker]["Close"].dropna()

    rows = []
    
    # ThreadPoolExecutorによる並列処理に変更
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for t in tickers:
            print(f"Submitting {t}...")
            d = data.get(t)
            futures.append(executor.submit(process_single_ticker, t, d, idx_close))
        
        for future in futures:
            try:
                res = future.result()
                rows.append(res)
            except Exception as e:
                print(f"Processing Error: {e}")

    return rows


def main():
    cfg = load_app_config()
    today = get_today_jst()

    if is_skip_day(today):
        print(f"[SKIP] {today.isoformat()} (weekend/holiday/year-end)")
        return

    ws = open_worksheet(cfg)
    tickers = read_tickers_from_sheet(ws)
    if not tickers:
        print("No tickers in sheet.")
        return

    index_ticker = cfg.get("index_ticker", "^TOPX")
    target_per = float(cfg.get("target_per", 15))

    rows = analyze_universe(tickers, index_ticker=index_ticker, target_per=target_per)

    headers = [
        "銘柄コード", "銘柄名", "業種", "判定結果", "現在値", "適正株価", "乖離率", "買いタイミング", "目標売値(利確)",
        "トレンド(75MA)", "トレンド(200MA)",
        "RS比(52週)", "VCP示唆", "営業利益3年増", "経常利益3年増", "EPS3年増",
        "上方修正期待", "決算発表日", "決算アラート", "Vol高", "推奨資金配分"
    ]

    write_output(ws, headers, rows)
    print(f"[OK] Updated rows: {len(rows)}")


if __name__ == "__main__":
    main()

import os
import json
import math
from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

import gspread
from google.oauth2.service_account import Credentials

import jpholiday


JST = ZoneInfo("Asia/Tokyo")


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
    # FIX: Use named arguments to avoid DeprecationWarning
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


def format_pct(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x*100:+.1f}%"


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
                # v can be a list, an array, or a scalar.
                # If it's a sequence, take the first element.
                if isinstance(v, (list, tuple, np.ndarray, pd.Series, pd.Index)):
                    if len(v) == 0:
                        continue
                    v0 = v[0]
                else:
                    v0 = v
                
                ts = pd.to_datetime(v0, errors="coerce")
                
                # FIX: Handle case where ts is still an array/Index (e.g. from array input)
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
# Core analysis
# ----------------------------
def analyze_universe(
    tickers: list[str],
    index_ticker: str,
    target_per: float,
) -> list[list]:
    if not tickers:
        return []

    all_symbols = tickers + [index_ticker]

    # FIX: Attempt to reduce 'database is locked' errors by not using shared cache if possible,
    # though yfinance cache is automatic. 
    # threads=True is faster but caused the lock. We keep True for speed but handle errors gracefully.
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
        # If single ticker was downloaded (rare here as we add index), handle gracefully
        if len(all_symbols) == 1:
             data[all_symbols[0]] = df.dropna(how="all")
        else:
             # Fallback if structure is unexpected
             pass

    if index_ticker not in data or data[index_ticker].empty:
        # If index download failed (e.g. database locked), we cannot calculate RS
        print(f"Warning: Index ticker {index_ticker} data missing. RS calculation will fail.")
        idx_close = pd.Series(dtype=float)
    else:
        idx_close = data[index_ticker]["Close"].dropna()

    rows = []
    for t in tickers:
        d = data.get(t)
        # If ticker download failed (e.g. 8283.T lock), d is None or empty
        if d is None or d.empty:
            rows.append([t, "取得失敗"] + [""] * 17)
            continue

        close = d["Close"].dropna()
        high = d["High"].dropna()

        # Check data length
        if len(close) < 260 or (not idx_close.empty and len(idx_close) < 260):
            # Not enough data
            if len(close) < 260:
                 rows.append([t, "データ不足"] + [""] * 17)
            else:
                 # Index data missing case
                 rows.append([t, "指数データ不足"] + [""] * 17)
            continue

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

        buy_timing = ""
        hi20 = safe_float(high.iloc[-20:].max()) if len(high) >= 20 else None
        if hi20 is not None and last_close >= hi20 * 0.95:
            buy_timing = f"{hi20:.0f}超えで買い"

        op_cagr = None
        ord_cagr = None
        eps_cagr = None
        uprev = ""

        fair_value = None
        divergence = None

        earnings_date = None
        alert = ""

        try:
            tk = yf.Ticker(t)

            try:
                cal = tk.calendar
                earnings_date = parse_earnings_date_from_calendar(cal)
            except Exception:
                earnings_date = None

            if earnings_date is not None:
                days = (earnings_date - get_today_jst()).days
                if 0 <= days <= 30:
                    alert = "⚠️1ヶ月以内"

            fin = None
            try:
                fin = tk.financials
            except Exception:
                fin = None

            op_row = pick_row(fin, ["Operating Income"])
            op_vals = annual_points_last_4(op_row) if op_row is not None else None
            if op_vals and len(op_vals) >= 2:
                op_cagr = compute_cagr_from_series(op_vals)

            ord_row = pick_row(fin, ["Pretax Income", "Income Before Tax"])
            ord_vals = annual_points_last_4(ord_row) if ord_row is not None else None
            if ord_vals and len(ord_vals) >= 2:
                ord_cagr = compute_cagr_from_series(ord_vals)

            eps_row = pick_row(fin, ["Basic EPS", "Diluted EPS"])
            eps_vals = annual_points_last_4(eps_row) if eps_row is not None else None
            if eps_vals and len(eps_vals) >= 2:
                eps_cagr = compute_cagr_from_series(eps_vals)

            info = {}
            try:
                info = tk.info or {}
            except Exception:
                info = {}

            forward_eps = safe_float(info.get("forwardEps"))
            trailing_eps = safe_float(info.get("trailingEps"))

            if forward_eps is not None and trailing_eps is not None and trailing_eps > 0:
                if forward_eps > trailing_eps * 1.10:
                    uprev = "あり"
                else:
                    uprev = "なし"
            else:
                uprev = ""

            eps_used = forward_eps if (forward_eps is not None and forward_eps > 0) else trailing_eps
            if eps_used is not None and eps_used > 0 and target_per and target_per > 0:
                fair_value = eps_used * target_per
                divergence = fair_value / last_close - 1

        except Exception:
            pass

        if trend_pass and rs_ok:
            verdict = "合格"
        elif trend_pass or rs_ok:
            verdict = "監視"
        else:
            verdict = "除外"

        allocation = "Half" if (alert or vol_high) else "Full"

        target_sell = last_close * 1.14

        rows.append(
            [
                t.replace(".T", ""),
                verdict,
                round(last_close, 2),
                "○" if (last_close > float(ma75.iloc[-1]) and slope_positive(ma75, 20)) else "×",
                f"{float(ma200.iloc[-1]):.0f} (上向き)" if slope_positive(ma200, 20) else f"{float(ma200.iloc[-1]):.0f} (横/下)",
                round(rs_ratio, 3) if rs_ratio is not None else "",
                format_bool_mark(vcp_hint),
                format_pct(op_cagr),
                format_pct(ord_cagr),
                format_pct(eps_cagr),
                uprev,
                format_date(earnings_date),
                alert,
                format_bool_mark(vol_high),
                buy_timing,
                round(fair_value, 2) if fair_value is not None else "",
                format_pct(divergence) if divergence is not None else "",
                allocation,
                round(target_sell, 2),
            ]
        )

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
        "銘柄コード", "判定結果", "現在値(終値)", "トレンド(75MA)", "トレンド(200MA)",
        "RS比(52週)", "VCP示唆", "営業利益3年増", "経常利益3年増", "EPS3年増",
        "上方修正期待", "決算発表日", "決算アラート", "Vol高",
        "買いタイミング", "適正株価", "乖離率", "推奨資金配分", "目標売値(利確)"
    ]

    write_output(ws, headers, rows)
    print(f"[OK] Updated rows: {len(rows)}")


if __name__ == "__main__":
    main()

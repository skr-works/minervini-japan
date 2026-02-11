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
# True : スクレイピングを行い、B列(銘柄名)・C列(業種)からV列までを更新 (低速)
# False: スクレイピングを行わず、D列(判定結果)からV列までを更新 (高速)
# ==========================================
UPDATE_BC_WITH_SCRAPING = False


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

# 修正: User-Agentリスト (ランダム化用)
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
    # .T を除去してURL作成
    code_only = ticker_code.replace(".T", "")
    url = f"https://finance.yahoo.co.jp/quote/{code_only}.T"
    
    # 修正: User-Agentランダム化
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    
    try:
        # 修正: 待機時間を延長 (max_workers=4対策)
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
        
        # fallback
        if not name:
            name = str(ticker_code)
            
        return name, sector
    except Exception as e:
        # 修正: 銘柄コードをログに出さない
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


def read_tickers_from_sheet(ws) -> list:
    """
    修正: A列(コード), B列(銘柄名), C列(業種) を取得する。
    戻り値は [ [code, name, sector], ... ] のリスト
    """
    raw_rows = ws.get("A2:C")
    
    if not raw_rows:
        return []

    data_list = []
    seen = set()

    for row in raw_rows:
        # row[0]=Code, row[1]=Name, row[2]=Sector (存在する場合)
        if not row:
            continue
            
        code = (row[0] or "").strip()
        
        # 安全に名前と業種を取得
        name = (row[1] or "").strip() if len(row) > 1 else ""
        sector = (row[2] or "").strip() if len(row) > 2 else ""

        if not code:
            continue
        
        # 修正: 名称に "DELISTED" や "廃止" が含まれていたらスキップ
        if "DELISTED" in name or "廃止" in name:
            continue
            
        if code not in seen:
            seen.add(code)
            data_list.append([code, name, sector])

    return data_list


def write_output_batch(ws, rows: list[list], start_row: int):
    """
    修正: 設定に応じて書き込み範囲を変更
    UPDATE_BC_WITH_SCRAPING is True  => B列(2番目)〜V列(22番目) = 21列分
    UPDATE_BC_WITH_SCRAPING is False => D列(4番目)〜V列(22番目) = 19列分
    """
    if not rows:
        return
    end_row = start_row + len(rows) - 1
    
    if UPDATE_BC_WITH_SCRAPING:
        range_name = f"B{start_row}:V{end_row}"
    else:
        range_name = f"D{start_row}:V{end_row}"
        
    ws.update(range_name=range_name, values=rows)


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

def process_single_ticker(ticker_data_tuple, d, idx_close):
    """
    1銘柄分の分析を実行する関数（並列処理用）
    ticker_data_tuple: (t_raw, pre_name, pre_sector) のタプル
    d: 株価DataFrame
    idx_close: 指数Closeデータ
    
    修正: 設定(UPDATE_BC_WITH_SCRAPING)に応じて戻り値を変更
    True  -> [Name, Sector, 判定, ... (計21要素)]
    False -> [判定, ... (計19要素)]
    """
    t_raw, pre_name, pre_sector = ticker_data_tuple
    
    # === 修正: 待機時間を延長 (max_workers=4対策) ===
    time.sleep(random.uniform(3.0, 5.0))
    
    # API用のティッカーシンボルを内部で生成 (例: "7203" -> "7203.T")
    api_t = f"{t_raw}.T" if str(t_raw).isdigit() else t_raw

    # 結果作成ヘルパー: エラー時などの戻り値生成
    def make_result_row(msg_list, is_error=True):
        # msg_list: [Verdict, LastPrice...] 形式のリスト (19要素)
        # 設定がTrueの場合、先頭にName, Sectorを追加して返す
        if UPDATE_BC_WITH_SCRAPING:
            # エラー時等は元の名前・業種を維持して返す
            return [pre_name, pre_sector] + msg_list
        else:
            return msg_list

    # 取得失敗チェック (d is df from yf.download)
    if d is None or d.empty:
        pass 

    # 株価データの準備 (download結果を使用)
    close = pd.Series(dtype=float)
    high = pd.Series(dtype=float)
    if d is not None and not d.empty:
        close = d["Close"].dropna()
        high = d["High"].dropna()

    # データ不足チェック
    if close.empty or len(close) < 1:
         return make_result_row(["取得失敗(株価なし)"] + [""] * 18)

    if len(close) < 260:
         # データ不足(上場直後など)
         return make_result_row(["データ不足"] + [""] * 18)
         
    if not idx_close.empty and len(idx_close) < 260:
         # 指数不足でも個別分析は続ける
         pass

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
        # 1. Scraping Name/Sector (optimized settings)
        # 修正: 設定に応じてスクレイピング実行有無を分岐
        if UPDATE_BC_WITH_SCRAPING:
            stock_name, industry = get_japanese_name_and_sector(api_t)
        else:
            stock_name = pre_name
            industry = pre_sector

        tk = yf.Ticker(api_t)
        
        # === 追加修正: info取得にリトライロジックを追加 ===
        info = {}
        # Max retries = 3
        for i_retry in range(3):
            try:
                temp_info = tk.info or {}
                if temp_info:
                    info = temp_info
                    break
            except Exception as e:
                # 404の場合は即時撤退（リトライしても無駄なため）
                if "404" in str(e) or "Not Found" in str(e):
                    return make_result_row(["取得失敗(404)"] + [""] * 18)
                # それ以外は少し待ってリトライ
                time.sleep(1.0 + i_retry)
        
        # 2. Calendar / Alert
        # === 追加修正: Calendar取得にリトライロジックを追加 ===
        cal = None
        for i_retry in range(3):
            try:
                cal = tk.calendar
                if cal is not None:
                    break
            except Exception:
                time.sleep(1.0 + i_retry)

        earnings_date = parse_earnings_date_from_calendar(cal)

        if earnings_date is not None:
            days = (earnings_date - get_today_jst()).days
            if 0 <= days <= 30:
                alert = "⚠️1ヶ月以内"
        
        # 3. Financials (buhin.py robust logic)
        # === 追加修正: Financials/BalanceSheet取得にリトライロジックを追加 ===
        financials = pd.DataFrame()
        for i_retry in range(3):
            try:
                financials = tk.financials
                if not financials.empty:
                    break
            except Exception:
                time.sleep(1.0 + i_retry)

        balance_sheet = pd.DataFrame()
        for i_retry in range(3):
            try:
                balance_sheet = tk.balance_sheet
                if not balance_sheet.empty:
                    break
            except Exception:
                time.sleep(1.0 + i_retry)
        
        # Basic info extraction (fallback for name - 修正: 英語名フォールバック維持)
        if stock_name == api_t or stock_name == "":
            stock_name = info.get('longName', t_raw)

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
        # 修正: エラーメッセージから銘柄コードの可能性を排除
        print(f"Error analyzing (masked): {e}")
        pass

    if trend_pass and rs_ok:
        verdict = "合格"
    elif trend_pass or rs_ok:
        verdict = "監視"
    else:
        verdict = "除外"

    allocation = "Half" if (alert or vol_high) else "Full"
    target_sell = last_close * 1.14

    # 19要素の分析結果リスト
    analysis_row = [
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

    # 修正: 設定に応じて結合して返す
    if UPDATE_BC_WITH_SCRAPING:
        return [stock_name, industry] + analysis_row
    else:
        return analysis_row


def main():
    cfg = load_app_config()
    today = get_today_jst()

    if is_skip_day(today):
        print(f"[SKIP] {today.isoformat()} (weekend/holiday/year-end)")
        return

    ws = open_worksheet(cfg)
    # 修正: A-C列を読み込む変数に変更
    tickers_data = read_tickers_from_sheet(ws)
    if not tickers_data:
        print("No tickers in sheet.")
        return

    index_ticker = cfg.get("index_ticker", "^TOPX")
    
    # --- バッチ処理への構造変更 ---
    
    # ヘッダー書き込み (初回のみ)
    full_headers = [
        "銘柄コード", "銘柄名", "業種", "判定結果", "現在値", "適正株価", "乖離率", "買いタイミング", "目標売値(利確)",
        "トレンド(75MA)", "トレンド(200MA)",
        "RS比(52週)", "VCP示唆", "営業利益3年増", "経常利益3年増", "EPS3年増",
        "上方修正期待", "決算発表日", "決算アラート", "Vol高", "推奨資金配分"
    ]
    
    # 修正: 設定に応じて書き込むヘッダーを調整
    if UPDATE_BC_WITH_SCRAPING:
        # B列(銘柄名)から書き込む
        headers = full_headers[1:]
    else:
        # D列(判定結果)から書き込む
        headers = full_headers[3:]
        
    write_output_batch(ws, [headers], 1)

    # 全体のindexデータだけ先に取得しておく(効率化のため)
    idx_close = pd.Series(dtype=float)
    try:
        df_idx = yf.download(index_ticker, period="2y", interval="1d", auto_adjust=False, progress=False)
        if not df_idx.empty:
            if isinstance(df_idx.columns, pd.MultiIndex):
                idx_close = df_idx["Close"][index_ticker].dropna() if index_ticker in df_idx.columns.get_level_values(0) else df_idx["Close"].iloc[:,0].dropna()
            else:
                idx_close = df_idx["Close"].dropna()
    except Exception as e:
        print(f"Index download error: {e}")

    BATCH_SIZE = 50
    total_tickers = len(tickers_data)
    current_index = 0

    print(f"Total Tickers: {total_tickers}")

    while current_index < total_tickers:
        end_index = min(current_index + BATCH_SIZE, total_tickers)
        batch_tickers_tuples = tickers_data[current_index:end_index] 
        
        # 修正: バッチ処理のログから具体的な銘柄リストを削除
        print(f"Processing batch: {current_index + 1} - {end_index} / {total_tickers}")

        # 修正: API用リストを作成 (タプルの0番目=コード を使用)
        def to_api_ticker(t):
            return f"{t}.T" if str(t).isdigit() else t
        
        batch_tickers_api = [to_api_ticker(t[0]) for t in batch_tickers_tuples]

        # 1. バッチ分の株価一括取得 (yf.download維持)
        batch_data = {}
        try:
            # columns mismatch回避のため group_by='ticker'
            df_p = yf.download(
                batch_tickers_api, 
                period="2y", 
                interval="1d", 
                group_by='ticker', 
                auto_adjust=False, 
                threads=True, 
                progress=False
            )
            
            # DataFrame構造の正規化
            # API用ティッカーをキーとしてデータを格納
            if isinstance(df_p.columns, pd.MultiIndex):
                for t_api in batch_tickers_api:
                    if t_api in df_p.columns.get_level_values(0):
                        batch_data[t_api] = df_p[t_api].dropna(how="all")
            else:
                # 1銘柄だけの場合
                if len(batch_tickers_api) == 1:
                    batch_data[batch_tickers_api[0]] = df_p.dropna(how="all")
                else:
                    pass
        except Exception as e:
            # 修正: エラー内容のみ表示
            print(f"Batch download error: {e}")

        # 2. バッチ分の分析 (並列処理)
        batch_rows = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for t_tuple in batch_tickers_tuples:
                # 取得時はAPI用ティッカーを使う
                t_raw = t_tuple[0]
                t_api = to_api_ticker(t_raw)
                d = batch_data.get(t_api) # DataFrame or None
                # 関数には「タプル全体」を渡す
                futures.append(executor.submit(process_single_ticker, t_tuple, d, idx_close))
            
            for future in futures:
                try:
                    res = future.result()
                    batch_rows.append(res)
                except Exception as e:
                    print(f"Future result error: {e}")
                    # エラー行も埋める
                    err_padding = ["Error"] * 18
                    # 設定に合わせて列数を調整
                    if UPDATE_BC_WITH_SCRAPING:
                        batch_rows.append(["Error", "Error", "Error"] + err_padding)
                    else:
                        batch_rows.append(["Error"] + err_padding)

        # 3. バッチ書き込み
        # 開始行: ヘッダー(1) + 既処理数 + 1(1-based) => current_index + 2
        start_write_row = current_index + 2
        try:
            write_output_batch(ws, batch_rows, start_write_row)
            # API制限回避
            # 修正: バッチ間待機時間を延長
            time.sleep(15)
        except Exception as e:
            # 修正: バッチ番号のみ表示
            print(f"Sheet write error at batch index {current_index}: {e}")

        current_index += BATCH_SIZE
        # 修正: バッチ間待機時間を延長 (再掲: バックアップとしてループ末尾にも)
        time.sleep(15) 

    print("[OK] All batches processed.")


if __name__ == "__main__":
    main()

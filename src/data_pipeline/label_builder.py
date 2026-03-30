import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats
import pandas_market_calendars as mcal
from datetime import datetime, time
import pytz

NYSE = mcal.get_calendar("NYSE")
ET   = pytz.timezone("America/New_York")
MARKET_OPEN  = time(9, 30)
MARKET_CLOSE = time(16, 0)


def get_return_window_start(call_date: str, call_time_et: str = None) -> str:
    schedule = NYSE.schedule(
        start_date=call_date,
        end_date=pd.Timestamp(call_date) + pd.Timedelta(days=10)
    )
    valid_days = mcal.date_range(schedule, frequency="1D")
    valid_days = [str(d.date()) for d in valid_days]

    if call_time_et is None:
        return valid_days[1] if len(valid_days) > 1 else valid_days[0]

    call_dt = datetime.strptime(f"{call_date} {call_time_et}", "%Y-%m-%d %H:%M")
    t = call_dt.time()

    if t < MARKET_OPEN:
        return valid_days[0]
    elif t >= MARKET_CLOSE:
        return valid_days[1] if len(valid_days) > 1 else valid_days[0]
    else:
        return valid_days[0]


def compute_event_study_abnormal_return(
    ticker: str,
    call_date: str,
    estimation_window: int = 200,
    event_window_days: list = None
) -> dict:
    if event_window_days is None:
        event_window_days = [1, 3, 7]

    end_est   = pd.Timestamp(call_date) - pd.Timedelta(days=1)
    start_est = end_est - pd.Timedelta(days=estimation_window * 2)

    try:
        stock_hist = yf.download(ticker, start=start_est, end=end_est,
                                 auto_adjust=True, progress=False)["Close"]
        mkt_hist   = yf.download("SPY",   start=start_est, end=end_est,
                                 auto_adjust=True, progress=False)["Close"]

        stock_ret = stock_hist.pct_change().dropna()
        mkt_ret   = mkt_hist.pct_change().dropna()

        combined = pd.concat([stock_ret, mkt_ret], axis=1).dropna()
        combined.columns = ["stock", "market"]
        combined = combined.iloc[-estimation_window:]

        if len(combined) < 60:
            return _empty_labels(event_window_days)

        slope, intercept, r_value, _, _ = stats.linregress(
            combined["market"], combined["stock"]
        )
        alpha, beta = intercept, slope

        window_start = get_return_window_start(call_date)
        end_event    = pd.Timestamp(window_start) + pd.Timedelta(days=max(event_window_days) + 5)

        stock_event = yf.download(ticker, start=window_start, end=end_event,
                                  auto_adjust=True, progress=False)["Close"]
        mkt_event   = yf.download("SPY",  start=window_start, end=end_event,
                                  auto_adjust=True, progress=False)["Close"]

        stock_rets_event = stock_event.pct_change().dropna()
        mkt_rets_event   = mkt_event.pct_change().dropna()

        labels = {
            "estimation_alpha":  alpha,
            "estimation_beta":   beta,
            "estimation_r2":     r_value ** 2,
        }

        for w in event_window_days:
            if len(stock_rets_event) < w:
                labels[f"abnormal_ret_{w}d"]  = np.nan
                labels[f"abnormal_vol_{w}d"]  = np.nan
                continue

            ar = stock_rets_event.iloc[:w].values - (
                alpha + beta * mkt_rets_event.iloc[:w].values
            )
            labels[f"abnormal_ret_{w}d"]  = float(np.sum(ar))
            labels[f"abnormal_vol_{w}d"]  = float(np.std(ar))
            labels[f"raw_abs_ret_{w}d"]   = float(abs(stock_rets_event.iloc[:w].sum()))

        return labels

    except Exception as e:
        print(f"Label computation failed for {ticker} {call_date}: {e}")
        return _empty_labels(event_window_days)


def _empty_labels(windows: list) -> dict:
    labels = {}
    for w in windows:
        labels[f"abnormal_ret_{w}d"] = np.nan
        labels[f"abnormal_vol_{w}d"] = np.nan
        labels[f"raw_abs_ret_{w}d"]  = np.nan
    return labels


def compute_earnings_surprise(ticker: str, call_date: str) -> float | None:
    try:
        ticker_obj = yf.Ticker(ticker)
        earnings   = ticker_obj.earnings_history

        if earnings is None or earnings.empty:
            return None

        earnings.index = pd.to_datetime(earnings.index)
        call_ts = pd.Timestamp(call_date)

        diffs  = abs(earnings.index - call_ts)
        closest_idx = diffs.argmin()

        if diffs[closest_idx] > pd.Timedelta(days=10):
            return None

        row       = earnings.iloc[closest_idx]
        actual    = row.get("epsActual",   None)
        consensus = row.get("epsEstimate", None)

        if actual is None or consensus is None or abs(consensus) < 1e-6:
            return None

        return float((actual - consensus) / abs(consensus))

    except Exception:
        return None


def fetch_structured_features(ticker: str, call_date: str,
                               call_time_et: str = None) -> dict:
    try:
        info = yf.Ticker(ticker).info
        hist = yf.download("^VIX", start=call_date,
                           end=pd.Timestamp(call_date) + pd.Timedelta(days=5),
                           progress=False)
        vix = float(hist["Close"].iloc[0]) if len(hist) > 0 else np.nan

        end   = pd.Timestamp(call_date)
        start = end - pd.Timedelta(days=60)
        prices  = yf.download(ticker, start=start, end=end,
                               auto_adjust=True, progress=False)["Close"]
        hist_vol = float(prices.pct_change().dropna().std() * np.sqrt(252))

        surprise = compute_earnings_surprise(ticker, call_date)

        return {
            "earnings_surprise": surprise,
            "market_cap_log":    np.log(info.get("marketCap", 1) + 1),
            "sector":            info.get("sector", "Unknown"),
            "vix_at_call":       vix,
            "hist_vol_30d":      hist_vol,
            "call_time_et":      call_time_et,
        }

    except Exception:
        return {
            "earnings_surprise": None,
            "market_cap_log":    np.nan,
            "sector":            "Unknown",
            "vix_at_call":       np.nan,
            "hist_vol_30d":      np.nan,
            "call_time_et":      call_time_et,
        }


def build_labels_for_maec_calls(calls: list[dict],
                                  call_times: dict = None) -> pd.DataFrame:
    records = []

    for call in calls:
        ticker    = call["ticker"]
        call_date = call["call_date"]
        call_time = (call_times or {}).get(call["call_id"])

        labels     = compute_event_study_abnormal_return(ticker, call_date)
        structured = fetch_structured_features(ticker, call_date, call_time)

        records.append({
            "call_id":   call["call_id"],
            "ticker":    ticker,
            "call_date": call_date,
            **labels,
            **structured,
        })

    df = pd.DataFrame(records)
    from pathlib import Path
    Path("data/processed/labels").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/labels/labels.csv", index=False)
    return df

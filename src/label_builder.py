import os, csv, argparse, yaml
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.data_sources import MultiSourceFetcher


def _extract_close_series(df, ticker_hint=None):
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    if isinstance(df, pd.Series):
        return pd.to_numeric(df, errors="coerce").dropna()

    close = None

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]
        elif "Adj Close" in df.columns.get_level_values(0):
            close = df["Adj Close"]
    else:
        if "Close" in df.columns:
            close = df["Close"]
        elif "Adj Close" in df.columns:
            close = df["Adj Close"]

    if close is None:
        return pd.Series(dtype=float)

    if isinstance(close, pd.DataFrame):
        if ticker_hint is not None and ticker_hint in close.columns:
            close = close[ticker_hint]
        elif close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            return pd.Series(dtype=float)

    return pd.to_numeric(close, errors="coerce").dropna()


def compute_abnormal_vol(ticker, call_date, est_window=200, event_windows=[1, 3, 7], fetcher=None):
    end_est = pd.Timestamp(call_date) - pd.Timedelta(days=1)
    start_est = end_est - pd.Timedelta(days=est_window * 2)
    end_event = pd.Timestamp(call_date) + pd.Timedelta(days=max(event_windows) + 10)

    try:
        if fetcher is None:
            fetcher = MultiSourceFetcher()
        
        stock = fetcher.fetch(ticker, start_est, end_event)
        mkt = fetcher.fetch("SPY", start_est, end_event)
        
        if stock.empty or mkt.empty:
            return _empty(event_windows)

        stock_ret = stock.ffill().pct_change().dropna()
        mkt_ret = mkt.ffill().pct_change().dropna()

        combined = pd.concat([stock_ret.rename("stock"), mkt_ret.rename("market")], axis=1).dropna()
        if combined.empty:
            return _empty(event_windows)

        call_ts = pd.Timestamp(call_date)
        event_pos = combined.index.searchsorted(call_ts, side="left")
        est_data = combined.iloc[max(0, event_pos - est_window):event_pos]

        if len(est_data) < 60:
            return _empty(event_windows)

        if est_data["market"].nunique() < 2 or est_data["stock"].nunique() < 2:
            return _empty(event_windows)

        slope, intercept, r_val, _, _ = stats.linregress(est_data["market"], est_data["stock"])
        alpha, beta = float(intercept), float(slope)

        event_data = combined.iloc[event_pos:]

        labels = {"alpha": alpha, "beta": beta, "r2": float(r_val ** 2)}
        for w in event_windows:
            if len(event_data) < w:
                labels[f"abnormal_ret_{w}d"] = np.nan
                labels[f"abnormal_vol_{w}d"] = np.nan
                continue
            stock_w = event_data["stock"].iloc[:w].to_numpy()
            market_w = event_data["market"].iloc[:w].to_numpy()
            ar = stock_w - (alpha + beta * market_w)
            ar_sum = float(np.sum(ar))
            labels[f"abnormal_ret_{w}d"] = ar_sum
            labels[f"abnormal_vol_{w}d"] = float(np.abs(ar_sum))

        return labels
    except Exception:
        return _empty(event_windows)


def _empty(windows):
    labels = {}
    for w in windows:
        labels[f"abnormal_ret_{w}d"] = np.nan
        labels[f"abnormal_vol_{w}d"] = np.nan
    return labels


def process_one(args_tuple):
    call_id, ticker, call_date, est_window, event_windows, fetcher = args_tuple
    labels = compute_abnormal_vol(ticker, call_date, est_window, event_windows, fetcher)
    return {"call_id": call_id, "ticker": ticker, "call_date": call_date, **labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_calls", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--alpha_vantage_key", default="1OZTV4XVWLHYGLGK", help="Alpha Vantage API key")
    parser.add_argument("--polygon_key", default="6bCz9o2yBJx1VnkSa2PDQaVpCIT2eop0", help="Polygon.io API key")
    parser.add_argument("--fmp_key", default="f2juoA2I8hhjgjKLodPsdC2DqYS9UfDd", help="Financial Modeling Prep API key")
    parser.add_argument("--tiingo_key", default="294c99a08e420e338d6ab14379639b26e699da8d", help="Tiingo API key")
    parser.add_argument("--quandl_key", default="hmh75wKz-mfbcZYexuP5", help="Quandl/Nasdaq Data Link API key")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output = args.output or cfg["data"]["labels_path"]
    est_window = cfg["label"]["estimation_window"]
    event_windows = cfg["label"]["event_windows"]

    manifest = pd.read_csv(args.manifest)
    if args.max_calls:
        manifest = manifest.head(args.max_calls)
    
    fetcher = MultiSourceFetcher(
        alpha_vantage_key=args.alpha_vantage_key,
        polygon_key=args.polygon_key,
        fmp_key=args.fmp_key,
        tiingo_key=args.tiingo_key,
        quandl_key=args.quandl_key
    )

    tasks = []
    for _, row in manifest.iterrows():
        tasks.append((row["call_id"], row["ticker"], row["call_date"],
                       est_window, event_windows, fetcher))

    print(f"Fetching labels for {len(tasks)} calls with {args.workers} workers...")
    sources_enabled = ["yfinance", "Yahoo Direct", "Yahoo v8", "pandas_datareader", "Twelve Data", "EOD Historical", "World Trading Data"]
    if args.tiingo_key:
        sources_enabled.insert(3, "Tiingo")
    if args.fmp_key:
        sources_enabled.insert(4, "FMP")
    if args.alpha_vantage_key:
        sources_enabled.append("Alpha Vantage")
    if args.polygon_key:
        sources_enabled.append("Polygon.io")
    if args.quandl_key:
        sources_enabled.append("Quandl")
    print(f"  Active sources ({len(sources_enabled)}): {', '.join(sources_enabled[:6])}{'...' if len(sources_enabled) > 6 else ''}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(process_one, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Labels"):
            try:
                results.append(future.result())
            except Exception:
                pass

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)

    valid = df.dropna(subset=[cfg["label"]["primary_target"]])
    print(f"\n{'='*60}")
    print(f"Done. {len(valid)}/{len(df)} calls have valid labels ({len(valid)/len(df)*100:.1f}%)")
    print(f"Output: {output}")
    print(f"{'='*60}")
    print(f"Target stats:\n{valid[cfg['label']['primary_target']].describe()}")


if __name__ == "__main__":
    main()

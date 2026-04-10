import os, argparse, yaml
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from src.data_sources import MultiSourceFetcher

# Load environment variables from .env file
load_dotenv()


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
    labels = {"alpha": np.nan, "beta": np.nan, "r2": np.nan}
    for w in windows:
        labels[f"abnormal_ret_{w}d"] = np.nan
        labels[f"abnormal_vol_{w}d"] = np.nan
    return labels


def process_one(args_tuple):
    call_id, ticker, call_date, est_window, event_windows, fetcher = args_tuple
    labels = compute_abnormal_vol(ticker, call_date, est_window, event_windows, fetcher)
    return {"call_id": call_id, "ticker": ticker, "call_date": call_date, **labels}


def _run_stage(stage_name, stage_tasks, fetcher, workers):
    stage_results = []
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(
                process_one,
                (task["call_id"], task["ticker"], task["call_date"], task["est_window"], task["event_windows"], fetcher),
            ): task["call_id"]
            for task in stage_tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=stage_name):
            try:
                stage_results.append(future.result())
            except Exception:
                pass
    return stage_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_calls", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--alpha_vantage_key", default=None, help="Alpha Vantage API key")
    parser.add_argument("--polygon_key", default=None, help="Polygon.io API key")
    parser.add_argument("--fmp_key", default=None, help="Financial Modeling Prep API key")
    parser.add_argument("--tiingo_key", default=None, help="Tiingo API key")
    parser.add_argument("--quandl_key", default=None, help="Quandl/Nasdaq Data Link API key")
    args = parser.parse_args()

    # Load API keys from environment variables if not provided via CLI
    alpha_vantage_key = args.alpha_vantage_key or os.getenv("ALPHA_VANTAGE_KEY")
    polygon_key = args.polygon_key or os.getenv("POLYGON_KEY")
    fmp_key = args.fmp_key or os.getenv("FMP_KEY")
    tiingo_key = args.tiingo_key or os.getenv("TIINGO_KEY")
    quandl_key = args.quandl_key or os.getenv("QUANDL_KEY")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output = args.output or cfg["data"]["labels_path"]
    est_window = cfg["label"]["estimation_window"]
    event_windows = cfg["label"]["event_windows"]
    primary_target = cfg["label"]["primary_target"]

    manifest = pd.read_csv(args.manifest)
    if args.max_calls:
        manifest = manifest.head(args.max_calls)

    tasks = []
    for _, row in manifest.iterrows():
        tasks.append(
            {
                "call_id": row["call_id"],
                "ticker": row["ticker"],
                "call_date": row["call_date"],
                "est_window": est_window,
                "event_windows": event_windows,
            }
        )

    # Build staged fallback fetchers so APIs are used one-by-one:
    # free sources -> Tiingo -> FMP -> Alpha Vantage -> Polygon -> Quandl
    stages = [
        (
            "Stage 1/6 Free Sources",
            MultiSourceFetcher(
                alpha_vantage_key=None,
                polygon_key=None,
                fmp_key=None,
                tiingo_key=None,
                quandl_key=None,
            ),
        )
    ]

    if tiingo_key:
        stages.append(
            (
                "Stage 2/6 Tiingo",
                MultiSourceFetcher(
                    alpha_vantage_key=None,
                    polygon_key=None,
                    fmp_key=None,
                    tiingo_key=tiingo_key,
                    quandl_key=None,
                ),
            )
        )
    if fmp_key:
        stages.append(
            (
                "Stage 3/6 FMP",
                MultiSourceFetcher(
                    alpha_vantage_key=None,
                    polygon_key=None,
                    fmp_key=fmp_key,
                    tiingo_key=None,
                    quandl_key=None,
                ),
            )
        )
    if alpha_vantage_key:
        stages.append(
            (
                "Stage 4/6 Alpha Vantage",
                MultiSourceFetcher(
                    alpha_vantage_key=alpha_vantage_key,
                    polygon_key=None,
                    fmp_key=None,
                    tiingo_key=None,
                    quandl_key=None,
                ),
            )
        )
    if polygon_key:
        stages.append(
            (
                "Stage 5/6 Polygon",
                MultiSourceFetcher(
                    alpha_vantage_key=None,
                    polygon_key=polygon_key,
                    fmp_key=None,
                    tiingo_key=None,
                    quandl_key=None,
                ),
            )
        )
    if quandl_key:
        stages.append(
            (
                "Stage 6/6 Quandl",
                MultiSourceFetcher(
                    alpha_vantage_key=None,
                    polygon_key=None,
                    fmp_key=None,
                    tiingo_key=None,
                    quandl_key=quandl_key,
                ),
            )
        )

    print(f"Fetching labels for {len(tasks)} calls with staged fallback and {args.workers} workers/stage...")

    # Start with empty rows for every call_id so output always has all manifest rows.
    results_by_call = {
        task["call_id"]: {
            "call_id": task["call_id"],
            "ticker": task["ticker"],
            "call_date": task["call_date"],
            **_empty(event_windows),
        }
        for task in tasks
    }

    unresolved = {task["call_id"] for task in tasks}
    task_lookup = {task["call_id"]: task for task in tasks}

    for stage_name, stage_fetcher in stages:
        if not unresolved:
            break

        stage_tasks = [task_lookup[call_id] for call_id in unresolved]
        print(f"\n{stage_name}: processing {len(stage_tasks)} unresolved calls")
        stage_results = _run_stage(stage_name, stage_tasks, stage_fetcher, args.workers)

        for row in stage_results:
            results_by_call[row["call_id"]] = row

        unresolved = {
            call_id
            for call_id in unresolved
            if pd.isna(results_by_call[call_id].get(primary_target, np.nan))
        }
        print(f"{stage_name}: resolved {len(tasks) - len(unresolved)}/{len(tasks)} total")

    if unresolved:
        print(f"\nUnresolved after all stages: {len(unresolved)}")

    ordered_results = [results_by_call[task["call_id"]] for task in tasks]
    df = pd.DataFrame(ordered_results)

    if primary_target not in df.columns:
        df[primary_target] = np.nan

    print(f"\nRaw data: {len(df)} calls")
    original_count = len(df)

    # Clean the data
    print("\nCleaning labels...")
    
    # 1. Remove rows where beta=1.0 AND r2=1.0 (failed data fetch)
    suspicious_market_model = (df['beta'] == 1.0) & (df['r2'] == 1.0)
    print(f"  Removing {suspicious_market_model.sum()} calls with beta=1.0 & r2=1.0")
    df = df[~suspicious_market_model]

    # 2. Remove rows where abnormal_vol_3d is exactly 0.0 (suspicious)
    zero_vol = (df[primary_target] == 0.0)
    print(f"  Removing {zero_vol.sum()} calls with exactly 0.0 volatility")
    df = df[~zero_vol]

    # 3. Remove rows with missing target
    missing_target = df[primary_target].isna()
    print(f"  Removing {missing_target.sum()} calls with missing target")
    df = df[~missing_target]

    # 4. Remove extreme outliers (>99.5th percentile)
    if len(df) > 0:
        threshold = df[primary_target].quantile(0.995)
        outliers = df[primary_target] > threshold
        print(f"  Removing {outliers.sum()} extreme outliers (>{threshold:.4f})")
        df = df[~outliers]

    # 5. Remove rows with very low r2 (<0.01) - poor market model fit
    low_r2 = df['r2'] < 0.01
    print(f"  Removing {low_r2.sum()} calls with r2 < 0.01")
    df = df[~low_r2]

    print(f"\nCleaned: {len(df)} calls ({len(df)/original_count*100:.1f}% retained)")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\n{'='*60}")
    print(f"Done. Saved {len(df)} clean calls")
    print(f"Output: {output}")
    print(f"{'='*60}")
    if len(df):
        print(f"\nTarget stats:")
        print(f"  Mean: {df[primary_target].mean():.6f}")
        print(f"  Std: {df[primary_target].std():.6f}")
        print(f"  Min: {df[primary_target].min():.6f}")
        print(f"  25%: {df[primary_target].quantile(0.25):.6f}")
        print(f"  50%: {df[primary_target].quantile(0.50):.6f}")
        print(f"  75%: {df[primary_target].quantile(0.75):.6f}")
        print(f"  Max: {df[primary_target].max():.6f}")
        
        # Date distribution
        df['call_date'] = pd.to_datetime(df['call_date'])
        print(f"\nDate distribution:")
        print(f"  2017-04 to 2017-10 (train): {((df['call_date'] >= '2017-04-24') & (df['call_date'] <= '2017-10-31')).sum()}")
        print(f"  2017-11 to 2017-12 (val): {((df['call_date'] >= '2017-11-01') & (df['call_date'] <= '2017-12-31')).sum()}")
        print(f"  2018-01 to 2018-06 (test): {((df['call_date'] >= '2018-01-01') & (df['call_date'] <= '2018-06-21')).sum()}")
    else:
        print("Target stats: no valid rows")


if __name__ == "__main__":
    main()

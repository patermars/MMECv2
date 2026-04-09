import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from typing import Optional, Tuple
from datetime import datetime
import json

class MultiSourceFetcher:
    """Fetches stock data from multiple sources with fallback logic."""
    
    def __init__(self, alpha_vantage_key: Optional[str] = None, 
                 polygon_key: Optional[str] = None,
                 fmp_key: Optional[str] = None,
                 tiingo_key: Optional[str] = None,
                 quandl_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        self.polygon_key = polygon_key
        self.fmp_key = fmp_key
        self.tiingo_key = tiingo_key
        self.quandl_key = quandl_key
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
    
    def fetch(self, ticker: str, start_date: pd.Timestamp, 
              end_date: pd.Timestamp) -> pd.Series:
        """Try multiple sources in order until one succeeds."""
        
        # Source 1: yfinance (primary)
        result = self._fetch_yfinance(ticker, start_date, end_date)
        if not result.empty:
            return result
        
        # Source 2: Yahoo Finance direct API
        result = self._fetch_yahoo_direct(ticker, start_date, end_date)
        if not result.empty:
            return result
        
        # Source 3: Yahoo Finance v8 API
        result = self._fetch_yahoo_v8(ticker, start_date, end_date)
        if not result.empty:
            return result
        
        # Source 4: Tiingo (free tier: 500 tickers, unlimited history)
        if self.tiingo_key:
            result = self._fetch_tiingo(ticker, start_date, end_date)
            if not result.empty:
                return result
        
        # Source 5: Financial Modeling Prep (free tier: 250 calls/day)
        if self.fmp_key:
            result = self._fetch_fmp(ticker, start_date, end_date)
            if not result.empty:
                return result
        
        # Source 6: Alpha Vantage
        if self.alpha_vantage_key:
            result = self._fetch_alpha_vantage(ticker, start_date, end_date)
            if not result.empty:
                return result
        
        # Source 7: Polygon.io
        if self.polygon_key:
            result = self._fetch_polygon(ticker, start_date, end_date)
            if not result.empty:
                return result
        
        # Source 8: Quandl/Nasdaq Data Link
        if self.quandl_key:
            result = self._fetch_quandl(ticker, start_date, end_date)
            if not result.empty:
                return result
        
        # Source 9: pandas_datareader (multiple backends)
        result = self._fetch_pandas_datareader(ticker, start_date, end_date)
        if not result.empty:
            return result
        
        # Source 10: Twelve Data (free tier: 800 calls/day)
        result = self._fetch_twelve_data(ticker, start_date, end_date)
        if not result.empty:
            return result
        
        # Source 11: EOD Historical Data (free tier available)
        result = self._fetch_eod_historical(ticker, start_date, end_date)
        if not result.empty:
            return result
        
        # Source 12: World Trading Data
        result = self._fetch_world_trading_data(ticker, start_date, end_date)
        if not result.empty:
            return result
        
        return pd.Series(dtype=float)
    
    def _fetch_yfinance(self, ticker: str, start: pd.Timestamp, 
                        end: pd.Timestamp) -> pd.Series:
        """Fetch using yfinance library."""
        try:
            df = yf.download(ticker, start=start, end=end, 
                           auto_adjust=True, progress=False)
            return self._extract_close(df, ticker)
        except:
            return pd.Series(dtype=float)
    
    def _fetch_alpha_vantage(self, ticker: str, start: pd.Timestamp, 
                            end: pd.Timestamp) -> pd.Series:
        """Fetch from Alpha Vantage API (free tier: 5 calls/min, 500/day)."""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker,
                "outputsize": "full",
                "apikey": self.alpha_vantage_key
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                return pd.Series(dtype=float)
            
            ts = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(ts, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Use adjusted close
            close = pd.to_numeric(df['5. adjusted close'], errors='coerce')
            close = close[(close.index >= start) & (close.index <= end)]
            
            time.sleep(12)  # Rate limit: 5 calls/min
            return close.dropna()
        except:
            return pd.Series(dtype=float)
    
    def _fetch_polygon(self, ticker: str, start: pd.Timestamp, 
                       end: pd.Timestamp) -> pd.Series:
        """Fetch from Polygon.io API (free tier available)."""
        try:
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
            params = {"adjusted": "true", "apiKey": self.polygon_key}
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("status") != "OK" or "results" not in data:
                return pd.Series(dtype=float)
            
            results = data["results"]
            dates = [pd.Timestamp(r['t'], unit='ms') for r in results]
            closes = [r['c'] for r in results]
            
            series = pd.Series(closes, index=dates)
            return series.dropna()
        except:
            return pd.Series(dtype=float)
    
    def _fetch_yahoo_direct(self, ticker: str, start: pd.Timestamp, 
                           end: pd.Timestamp) -> pd.Series:
        """Fetch using Yahoo Finance direct API endpoint."""
        try:
            start_ts = int(start.timestamp())
            end_ts = int(end.timestamp())
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
            params = {
                "period1": start_ts,
                "period2": end_ts,
                "interval": "1d",
                "events": "history"
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return pd.Series(dtype=float)
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            
            close = pd.to_numeric(df['Adj Close'], errors='coerce')
            return close.dropna()
        except:
            return pd.Series(dtype=float)
    
    def _fetch_yahoo_v8(self, ticker: str, start: pd.Timestamp, 
                        end: pd.Timestamp) -> pd.Series:
        """Fetch using Yahoo Finance v8 chart API."""
        try:
            start_ts = int(start.timestamp())
            end_ts = int(end.timestamp())
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
            params = {
                "period1": start_ts,
                "period2": end_ts,
                "interval": "1d"
            }
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'chart' not in data or 'result' not in data['chart']:
                return pd.Series(dtype=float)
            
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            closes = result['indicators']['adjclose'][0]['adjclose']
            
            dates = [pd.Timestamp(ts, unit='s') for ts in timestamps]
            series = pd.Series(closes, index=dates)
            return series.dropna()
        except:
            return pd.Series(dtype=float)
    
    def _fetch_tiingo(self, ticker: str, start: pd.Timestamp, 
                      end: pd.Timestamp) -> pd.Series:
        """Fetch from Tiingo API (free tier: 500 tickers, unlimited history)."""
        try:
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            params = {
                "startDate": start_str,
                "endDate": end_str,
                "token": self.tiingo_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            if not isinstance(data, list) or len(data) == 0:
                return pd.Series(dtype=float)
            
            dates = [pd.Timestamp(d['date']) for d in data]
            closes = [d['adjClose'] for d in data]
            
            series = pd.Series(closes, index=dates)
            return series.dropna()
        except:
            return pd.Series(dtype=float)
    
    def _fetch_fmp(self, ticker: str, start: pd.Timestamp, 
                   end: pd.Timestamp) -> pd.Series:
        """Fetch from Financial Modeling Prep (free tier: 250 calls/day)."""
        try:
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
            params = {
                "from": start_str,
                "to": end_str,
                "apikey": self.fmp_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'historical' not in data:
                return pd.Series(dtype=float)
            
            historical = data['historical']
            dates = [pd.Timestamp(d['date']) for d in historical]
            closes = [d['adjClose'] for d in historical]
            
            series = pd.Series(closes, index=dates)
            series = series.sort_index()
            return series.dropna()
        except:
            return pd.Series(dtype=float)
    
    def _fetch_marketstack(self, ticker: str, start: pd.Timestamp, 
                          end: pd.Timestamp) -> pd.Series:
        """Fetch from Marketstack (free tier: 100 calls/month, no key needed for basic)."""
        try:
            # Try without API key first (limited)
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            url = f"https://api.marketstack.com/v1/eod"
            params = {
                "symbols": ticker,
                "date_from": start_str,
                "date_to": end_str,
                "limit": 1000
            }
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'data' not in data or len(data['data']) == 0:
                return pd.Series(dtype=float)
            
            dates = [pd.Timestamp(d['date']) for d in data['data']]
            closes = [d['adj_close'] if d['adj_close'] else d['close'] for d in data['data']]
            
            series = pd.Series(closes, index=dates)
            series = series.sort_index()
            return series.dropna()
        except:
            return pd.Series(dtype=float)
    
    def _fetch_quandl(self, ticker: str, start: pd.Timestamp, 
                      end: pd.Timestamp) -> pd.Series:
        """Fetch from Quandl/Nasdaq Data Link."""
        try:
            import quandl
            quandl.ApiConfig.api_key = self.quandl_key
            df = quandl.get(f"WIKI/{ticker}", start_date=start, end_date=end)
            if df.empty:
                return pd.Series(dtype=float)
            close = df['Adj. Close'] if 'Adj. Close' in df.columns else df['Close']
            return pd.to_numeric(close, errors='coerce').dropna()
        except:
            return pd.Series(dtype=float)
    
    def _fetch_pandas_datareader(self, ticker: str, start: pd.Timestamp, 
                                 end: pd.Timestamp) -> pd.Series:
        """Fetch using pandas_datareader with multiple backends."""
        backends = ['stooq', 'yahoo', 'iex']
        
        for backend in backends:
            try:
                import pandas_datareader.data as web
                df = web.DataReader(ticker, backend, start, end)
                if df.empty:
                    continue
                close = df['Close'] if 'Close' in df.columns else df['close']
                result = pd.to_numeric(close, errors='coerce').dropna()
                if not result.empty:
                    return result
            except:
                continue
        
        return pd.Series(dtype=float)
    
    def _fetch_twelve_data(self, ticker: str, start: pd.Timestamp, 
                          end: pd.Timestamp) -> pd.Series:
        """Fetch from Twelve Data (free tier: 800 calls/day, no key needed for basic)."""
        try:
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            url = f"https://api.twelvedata.com/time_series"
            params = {
                "symbol": ticker,
                "interval": "1day",
                "start_date": start_str,
                "end_date": end_str,
                "format": "JSON"
            }
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'values' not in data:
                return pd.Series(dtype=float)
            
            values = data['values']
            dates = [pd.Timestamp(v['datetime']) for v in values]
            closes = [float(v['close']) for v in values]
            
            series = pd.Series(closes, index=dates).sort_index()
            return series.dropna()
        except:
            return pd.Series(dtype=float)
    
    def _fetch_eod_historical(self, ticker: str, start: pd.Timestamp, 
                             end: pd.Timestamp) -> pd.Series:
        """Fetch from EOD Historical Data."""
        try:
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            url = f"https://eodhistoricaldata.com/api/eod/{ticker}.US"
            params = {
                "from": start_str,
                "to": end_str,
                "fmt": "json"
            }
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            if not isinstance(data, list) or len(data) == 0:
                return pd.Series(dtype=float)
            
            dates = [pd.Timestamp(d['date']) for d in data]
            closes = [d['adjusted_close'] for d in data]
            
            series = pd.Series(closes, index=dates).sort_index()
            return series.dropna()
        except:
            return pd.Series(dtype=float)
    
    def _fetch_world_trading_data(self, ticker: str, start: pd.Timestamp, 
                                  end: pd.Timestamp) -> pd.Series:
        """Fetch from World Trading Data."""
        try:
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            url = f"https://api.worldtradingdata.com/api/v1/history"
            params = {
                "symbol": ticker,
                "date_from": start_str,
                "date_to": end_str,
                "sort": "newest"
            }
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'history' not in data:
                return pd.Series(dtype=float)
            
            history = data['history']
            dates = [pd.Timestamp(date) for date in history.keys()]
            closes = [float(history[date]['close']) for date in history.keys()]
            
            series = pd.Series(closes, index=dates).sort_index()
            return series.dropna()
        except:
            return pd.Series(dtype=float)
    
    def _extract_close(self, df, ticker_hint: Optional[str] = None) -> pd.Series:
        """Extract close price from various DataFrame formats."""
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
            if ticker_hint and ticker_hint in close.columns:
                close = close[ticker_hint]
            elif close.shape[1] == 1:
                close = close.iloc[:, 0]
            else:
                return pd.Series(dtype=float)
        
        return pd.to_numeric(close, errors="coerce").dropna()

# API Keys Setup Guide

Quick guide to obtain free API keys for the 5 stock data sources.

## 1. Alpha Vantage
- **URL:** https://www.alphavantage.co/support/#api-key
- **Steps:** Click "Get Your Free API Key" → Fill form → Instant key
- **Free Tier:** 5 calls/min, 500 calls/day
- **Usage:** `--alpha_vantage_key YOUR_KEY`

## 2. Polygon.io
- **URL:** https://polygon.io/dashboard/signup
- **Steps:** Sign up → Verify email → Dashboard → API Keys
- **Free Tier:** 5 calls/min (limited historical data)
- **Usage:** `--polygon_key YOUR_KEY`

## 3. Financial Modeling Prep (FMP)
- **URL:** https://site.financialmodelingprep.com/developer/docs
- **Steps:** Sign up → Dashboard → Copy API key
- **Free Tier:** 250 calls/day
- **Usage:** `--fmp_key YOUR_KEY`

## 4. Tiingo
- **URL:** https://www.tiingo.com/account/api/token
- **Steps:** Sign up → Account → API → Create token
- **Free Tier:** 500 unique tickers, unlimited history
- **Usage:** `--tiingo_key YOUR_KEY`

## 5. Quandl (Nasdaq Data Link)
- **URL:** https://data.nasdaq.com/sign-up
- **Steps:** Sign up → Account Settings → API Key
- **Free Tier:** 50 calls/day (WIKI dataset discontinued, limited data)
- **Usage:** `--quandl_key YOUR_KEY`

## Usage Example

```bash
python -m src.label_builder \
  --alpha_vantage_key YOUR_AV_KEY \
  --polygon_key YOUR_POLYGON_KEY \
  --fmp_key YOUR_FMP_KEY \
  --tiingo_key YOUR_TIINGO_KEY \
  --quandl_key YOUR_QUANDL_KEY
```

## Notes
- All services offer free tiers suitable for research
- Keys are optional - system falls back to yfinance if APIs fail
- Tiingo and FMP recommended for best free tier coverage
- Store keys in environment variables for security

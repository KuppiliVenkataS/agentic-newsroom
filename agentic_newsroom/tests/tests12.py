import sys
sys.path.insert(0, '.')
from ingestion.yfinance_fetcher import fetch_live_prices
prices = fetch_live_prices()
for p in prices:
    print(p)
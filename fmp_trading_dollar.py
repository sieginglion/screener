#!/usr/bin/env python3
import csv
import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import load_dotenv

FMP_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"
THREADS = 8
LOOKBACK_DAYS = 14
LAST_N = 7


def read_symbols(path):
    with open(path) as f:
        return [s.replace(".", "-") for s in f.read().strip().split(";")]


def fetch_total(symbol, from_date, api_key):
    r = httpx.get(
        FMP_URL,
        params={"symbol": symbol, "from": from_date, "apikey": api_key},
        timeout=30,
    )
    r.raise_for_status()

    rows = sorted(r.json(), key=lambda x: x["date"], reverse=True)[:LAST_N]
    total = sum(float(r["vwap"]) * float(r["volume"]) for r in rows)

    return symbol, total


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python script.py symbols.csv")

    load_dotenv()
    api_key = os.environ["FMP_API_KEY"]

    symbols = read_symbols(sys.argv[1])
    from_date = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = pool.map(
            lambda s: fetch_total(s, from_date, api_key),
            symbols,
        )

    results = sorted(results, key=lambda x: x[1], reverse=True)

    writer = csv.writer(sys.stdout)
    writer.writerow(["symbol", "total_trading_dollar"])
    for symbol, total in results:
        writer.writerow([symbol, f"{total:.2f}"])


if __name__ == "__main__":
    main()

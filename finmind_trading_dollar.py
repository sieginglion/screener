#!/usr/bin/env python3
import csv
import datetime as dt
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from FinMind.data import DataLoader

THREADS = 1
LOOKBACK_DAYS = 14
LAST_N = 7


def read_symbols(path):
    with open(path) as f:
        raw = f.read().strip().split(";")

    symbols = []
    for item in raw:
        m = re.search(r"\d+", item)
        if m:
            symbols.append((item, m.group()))
    return symbols


def fetch_total(api, name, stock_id, start, end):
    df = api.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=start,
        end_date=end,
    )

    if df.empty:
        return name, 0.0

    total = df.sort_values("date", ascending=False).head(LAST_N)["Trading_money"].sum()

    return name, float(total)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: script.py symbols.txt")

    load_dotenv()
    api = DataLoader()
    api.login_by_token(os.environ["FINMIND_KEY"])

    symbols = read_symbols(sys.argv[1])

    today = dt.date.today()
    start = (today - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    end = today.isoformat()

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = list(
            pool.map(
                lambda s: fetch_total(api, s[0], s[1], start, end),
                symbols,
            )
        )

    results.sort(key=lambda x: x[1], reverse=True)

    writer = csv.writer(sys.stdout)
    writer.writerow(["symbol", "total_trading_dollar"])
    for name, total in results:
        writer.writerow([name, f"{total:.2f}"])


if __name__ == "__main__":
    main()

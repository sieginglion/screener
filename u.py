#!/usr/bin/env python3
import csv
import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import load_dotenv

FMP_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"
TV_URL = "https://scanner.tradingview.com/america/scan?label-product=screener-stock"

THREADS = 2
LOOKBACK_DAYS = 21
LAST_N = 10
TOP_N_SYMBOLS = 512

TV_HEADERS = {
    "accept": "application/json",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "content-type": "text/plain;charset=UTF-8",
    "origin": "https://www.tradingview.com",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://www.tradingview.com/",
    "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
}

TV_PAYLOAD = {
    "columns": ["ticker-view"],
    # "filter": [{"left": "is_primary", "operation": "equal", "right": True}],
    "ignore_unknown_fields": False,
    "options": {"lang": "en"},
    "range": [0, TOP_N_SYMBOLS],
    "sort": {"sortBy": "Value.Traded|1W", "sortOrder": "desc"},
    "symbols": {},
    "markets": ["america"],
    "filter2": {
        "operator": "and",
        "operands": [
            {
                "operation": {
                    "operator": "or",
                    "operands": [
                        {
                            "operation": {
                                "operator": "and",
                                "operands": [
                                    {
                                        "expression": {
                                            "left": "type",
                                            "operation": "equal",
                                            "right": "stock",
                                        }
                                    },
                                    {
                                        "expression": {
                                            "left": "typespecs",
                                            "operation": "has",
                                            "right": ["common"],
                                        }
                                    },
                                ],
                            }
                        },
                        {
                            "operation": {
                                "operator": "and",
                                "operands": [
                                    {
                                        "expression": {
                                            "left": "type",
                                            "operation": "equal",
                                            "right": "stock",
                                        }
                                    },
                                    {
                                        "expression": {
                                            "left": "typespecs",
                                            "operation": "has",
                                            "right": ["preferred"],
                                        }
                                    },
                                ],
                            }
                        },
                        {
                            "operation": {
                                "operator": "and",
                                "operands": [
                                    {
                                        "expression": {
                                            "left": "type",
                                            "operation": "equal",
                                            "right": "dr",
                                        }
                                    }
                                ],
                            }
                        },
                        {
                            "operation": {
                                "operator": "and",
                                "operands": [
                                    {
                                        "expression": {
                                            "left": "type",
                                            "operation": "equal",
                                            "right": "fund",
                                        }
                                    },
                                    {
                                        "expression": {
                                            "left": "typespecs",
                                            "operation": "has_none_of",
                                            "right": ["etf"],
                                        }
                                    },
                                ],
                            }
                        },
                    ],
                }
            },
            {
                "expression": {
                    "left": "typespecs",
                    "operation": "has_none_of",
                    "right": ["pre-ipo"],
                }
            },
        ],
    },
}


def fetch_symbols_from_tv():
    with httpx.Client() as client:
        response = client.post(TV_URL, headers=TV_HEADERS, json=TV_PAYLOAD, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [
            (item["d"][0]["name"].replace(".", "-"), item["d"][0]["description"])
            for item in data["data"]
        ]


def fetch_trading_dollar(symbol, description, from_date, api_key):
    r = httpx.get(
        FMP_URL,
        params={"symbol": symbol, "from": from_date, "apikey": api_key},
        timeout=30,
    )
    r.raise_for_status()

    rows = sorted(r.json(), key=lambda x: x["date"], reverse=True)[:LAST_N]
    total = sum(float(row["vwap"]) * float(row["volume"]) for row in rows)

    return description, total


def main():
    load_dotenv()
    api_key = os.environ["FMP_API_KEY"]

    print("Fetching symbols from TradingView...", file=sys.stderr)
    stocks = fetch_symbols_from_tv()
    print(f"Found {len(stocks)} symbols", file=sys.stderr)

    from_date = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()

    print("Fetching trading dollar data from FMP...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = list(
            pool.map(
                lambda s: fetch_trading_dollar(s[0], s[1], from_date, api_key), stocks
            )
        )

    results = sorted(results, key=lambda x: x[1], reverse=True)

    writer = csv.writer(sys.stdout, delimiter=";")
    writer.writerow(["description", "total_trading_dollar"])
    for description, total in results:
        writer.writerow([description, f"{total:.2f}"])


if __name__ == "__main__":
    main()

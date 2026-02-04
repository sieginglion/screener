#!/usr/bin/env python3
import csv
import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import load_dotenv
from FinMind.data import DataLoader

TV_URL = "https://scanner.tradingview.com/taiwan/scan?label-product=screener-stock"

THREADS = 8
LOOKBACK_DAYS = 21
LAST_N = 10
TOP_N_SYMBOLS = 512

TV_HEADERS = {
    "accept": "application/json",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "content-type": "text/plain;charset=UTF-8",
    "origin": "https://tw.tradingview.com",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://tw.tradingview.com/",
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
    "options": {"lang": "zh_TW"},
    "range": [0, TOP_N_SYMBOLS],
    "sort": {"sortBy": "Value.Traded|1W", "sortOrder": "desc"},
    "symbols": {},
    "markets": ["taiwan"],
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
            (item["d"][0]["name"], item["d"][0]["description"])
            for item in data["data"]
        ]


def fetch_trading_dollar(api, stock_id, description, start, end):
    df = api.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=start,
        end_date=end,
    )
    total = df.sort_values("date", ascending=False).head(LAST_N)["Trading_money"].sum()
    return description, float(total)


def main():
    load_dotenv()
    api = DataLoader()
    api.login_by_token(os.environ["FINMIND_KEY"])

    print("Fetching symbols from TradingView...", file=sys.stderr)
    stocks = fetch_symbols_from_tv()
    print(f"Found {len(stocks)} symbols", file=sys.stderr)

    today = dt.date.today()
    start = (today - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    end = today.isoformat()

    print("Fetching trading dollar data from FinMind...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = list(
            pool.map(
                lambda s: fetch_trading_dollar(api, s[0], s[1], start, end), stocks
            )
        )

    results = sorted(results, key=lambda x: x[1], reverse=True)

    writer = csv.writer(sys.stdout, delimiter=";")
    writer.writerow(["description", "total_trading_dollar"])
    for description, total in results:
        writer.writerow([description, f"{total:.2f}"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import datetime as dt
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Sequence, Tuple

import httpx
from dotenv import load_dotenv

FMP_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"
TV_US_URL = "https://scanner.tradingview.com/america/scan?label-product=screener-stock"
TV_TW_URL = "https://scanner.tradingview.com/taiwan/scan?label-product=screener-stock"

PYTHON_BIN = sys.executable
CHUNK_SIZE = 64
CHUNK_NUMS = 2
THREADS = 2
LOOKBACK_DAYS = 28
LAST_N = 5
MARGIN = 4
MARKET = "u"  # Set to "u" (america) or "t" (taiwan)

TV_US_HEADERS = {
    "accept": "application/json",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "content-type": "text/plain;charset=UTF-8",
    "origin": "https://www.tradingview.com",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://www.tradingview.com/",
    "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
}

TV_US_PAYLOAD = {
    "columns": ["ticker-view"],
    # "filter": [{"left": "is_primary", "operation": "equal", "right": True}],
    "ignore_unknown_fields": False,
    "options": {"lang": "en"},
    "range": [0, CHUNK_SIZE * CHUNK_NUMS * MARGIN],
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

TV_TW_HEADERS = {
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

TV_TW_PAYLOAD = {
    "columns": ["ticker-view"],
    "ignore_unknown_fields": False,
    "options": {"lang": "zh_TW"},
    "range": [0, CHUNK_SIZE * CHUNK_NUMS * MARGIN],
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


def fetch_symbols_from_tv(
    tv_url: str,
    tv_headers: dict[str, str],
    tv_payload: dict[str, Any],
    top_n_symbols: int,
) -> List[Tuple[str, str]]:
    payload = dict(tv_payload)
    payload["range"] = [0, top_n_symbols]
    with httpx.Client() as client:
        response = client.post(tv_url, headers=tv_headers, json=payload)
        data = response.json()

        def normalize(name: str) -> str:
            return name.replace(".", "-")

        return [
            (normalize(item["d"][0]["name"]), item["d"][0]["description"])
            for item in data["data"]
        ]


def fetch_trading_dollar_us(
    symbol: str, description: str, from_date: str, api_key: str
) -> Tuple[str, str, float]:
    response = httpx.get(
        FMP_URL,
        params={"symbol": symbol, "from": from_date, "apikey": api_key},
    )
    rows = sorted(response.json(), key=lambda x: x["date"], reverse=True)[:LAST_N]
    total = sum(float(row["vwap"]) * float(row["volume"]) for row in rows)
    return symbol, description, total


def load_top_company_names_us(
    api_key: str | None, top_n_symbols: int, top_n_results: int
) -> List[str]:
    sys.stderr.write("Fetching symbols from TradingView...\n")
    stocks = fetch_symbols_from_tv(
        tv_url=TV_US_URL,
        tv_headers=TV_US_HEADERS,
        tv_payload=TV_US_PAYLOAD,
        top_n_symbols=top_n_symbols,
    )
    sys.stderr.write(f"Found {len(stocks)} symbols\n")

    from_date = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    sys.stderr.write("Fetching trading dollar data from FMP...\n")
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = list(
            pool.map(
                lambda s: fetch_trading_dollar_us(s[0], s[1], from_date, api_key),
                stocks,
            )
        )

    top = sorted(results, key=lambda x: x[2], reverse=True)[:top_n_results]
    return [
        f"{symbol} {description.replace(';', ',')}" for symbol, description, _ in top
    ]


def fetch_trading_dollar_tw(
    api: Any, stock_id: str, description: str, start: str, end: str
) -> Tuple[str, str, float]:
    df = api.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=start,
        end_date=end,
    )
    total = df.sort_values("date", ascending=False).head(LAST_N)["Trading_money"].sum()
    return stock_id, description, float(total)


def load_top_company_names_tw(top_n_symbols: int, top_n_results: int) -> List[str]:
    finmind_key = os.environ.get("FINMIND_KEY")

    from FinMind.data import DataLoader

    api = DataLoader()
    api.login_by_token(finmind_key)

    sys.stderr.write("Fetching symbols from TradingView...\n")
    stocks = fetch_symbols_from_tv(
        tv_url=TV_TW_URL,
        tv_headers=TV_TW_HEADERS,
        tv_payload=TV_TW_PAYLOAD,
        top_n_symbols=top_n_symbols,
    )
    sys.stderr.write(f"Found {len(stocks)} symbols\n")

    today = dt.date.today()
    start = (today - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    end = today.isoformat()

    sys.stderr.write("Fetching trading dollar data from FinMind...\n")
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = list(
            pool.map(
                lambda s: fetch_trading_dollar_tw(api, s[0], s[1], start, end),
                stocks,
            )
        )

    top = sorted(results, key=lambda x: x[2], reverse=True)[:top_n_results]
    return [
        f"{symbol} {description.replace(';', ',')}" for symbol, description, _ in top
    ]


def chunked(items: Sequence[str], size: int) -> List[List[str]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def run_analyze_part(ticker_company_pairs: Sequence[str]) -> str:
    """Run analyze_tickers.py for one chunk via stdin and return stdout."""
    try:
        proc = subprocess.run(
            [PYTHON_BIN, "analyze_tickers.py"],
            input=";".join(ticker_company_pairs),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise

    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.stdout


def main() -> int:
    load_dotenv()
    market = "taiwan" if MARKET == "t" else "america"

    top_n_results = CHUNK_SIZE * CHUNK_NUMS
    top_n_symbols = top_n_results * MARGIN

    if market == "taiwan":
        ticker_company_pairs = load_top_company_names_tw(top_n_symbols, top_n_results)
    else:
        api_key = os.environ.get("FMP_API_KEY")
        ticker_company_pairs = load_top_company_names_us(
            api_key, top_n_symbols, top_n_results
        )

    parts = chunked(ticker_company_pairs, CHUNK_SIZE)

    max_workers = len(parts)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_analyze_part, part) for part in parts]
        outputs = [fut.result() for fut in futures]

    for i, output in enumerate(outputs):
        if i > 0:
            sys.stdout.write("\n\n")
        sys.stdout.write(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

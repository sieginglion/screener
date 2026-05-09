#!/usr/bin/env python3
"""Clear selected entries from the shared screener HTTP cache."""

import argparse

from run_u_analyze import cache


def key_matches_ticker(key, ticker: str) -> bool:
    if not isinstance(key, tuple) or len(key) != 4:
        return False

    _, _, args, kwargs = key

    param_sets = []
    if len(args) > 1:
        param_sets.append(args[1])

    for name, value in kwargs:
        if name == "params":
            param_sets.append(value)

    return any(
        ("symbol", ticker) in params or ("stock_id", ticker) in params
        for params in param_sets
    )


def key_matches_endpoint(key, endpoint: str) -> bool:
    if not isinstance(key, tuple) or len(key) != 4:
        return False

    _, _, args, _ = key
    return bool(args and isinstance(args[0], str) and args[0].endswith(endpoint))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clear either a ticker-specific cache or the Portman scoring cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ticker", nargs="?", help="Ticker or stock_id to clear.")
    parser.add_argument(
        "--portman",
        action="store_true",
        dest="clear_scoring_caches",
        help="Clear every cached response for Portman /growths and /scores.",
    )
    parser.epilog = """Examples:
  clear_ticker_cache.py AAPL
  clear_ticker_cache.py --portman"""
    args = parser.parse_args()

    selected_modes = sum(
        bool(mode) for mode in (args.clear_scoring_caches, args.ticker)
    )
    if selected_modes != 1:
        parser.error("provide exactly one of: ticker or --portman")

    if args.clear_scoring_caches:
        endpoints = ("/growths", "/scores")
        keys = [
            key
            for key in cache.iterkeys()
            if any(key_matches_endpoint(key, endpoint) for endpoint in endpoints)
        ]
        label = "/growths and /scores cache entries"
    elif args.ticker:
        ticker = args.ticker.upper()
        keys = [key for key in cache.iterkeys() if key_matches_ticker(key, ticker)]
        label = f"cache entries for {ticker}"

    for key in keys:
        del cache[key]

    print(f"removed {len(keys)} {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

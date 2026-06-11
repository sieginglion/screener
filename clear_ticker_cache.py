#!/usr/bin/env python3
"""Clear selected entries from the shared screener disk cache."""

import argparse

from run_u_analyze import FMP_STABLE_URL, cache


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


def key_matches_function(key, qualname: str) -> bool:
    if not isinstance(key, tuple) or len(key) != 4:
        return False

    _, key_qualname, _, _ = key
    return key_qualname == qualname


def key_matches_url(key, url: str) -> bool:
    if not isinstance(key, tuple) or len(key) != 4:
        return False

    _, _, args, _ = key
    return bool(args and args[0] == url)


def key_matches_liquidity(key) -> bool:
    return key_matches_function(key, "load_top_company_names_us") or (
        key_matches_function(key, "cached_httpx_get")
        and key_matches_url(key, FMP_STABLE_URL)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clear selected entries from the shared screener disk cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        help="Clear cached entries whose params include this symbol or stock_id.",
    )
    parser.add_argument(
        "--portman",
        action="store_true",
        dest="clear_scoring_caches",
        help="Clear every cached response for Portman /growths and /scores.",
    )
    parser.add_argument(
        "--liquidity",
        "--load-top-company-names-us",
        "--load_top_company_names_us",
        action="store_true",
        dest="clear_liquidity_cache",
        help="Clear cached FMP stable endpoint responses used for the US liquidity ranking.",
    )
    parser.epilog = """Examples:
  clear_ticker_cache.py AAPL
  clear_ticker_cache.py --portman
  clear_ticker_cache.py --liquidity"""
    args = parser.parse_args()

    selected_modes = sum(
        bool(mode)
        for mode in (
            args.clear_scoring_caches,
            args.clear_liquidity_cache,
            args.ticker,
        )
    )
    if selected_modes != 1:
        parser.error(
            "provide exactly one of: ticker, --portman, or --liquidity"
        )

    if args.clear_scoring_caches:
        endpoints = ("/growths", "/scores")
        keys = [
            key
            for key in cache.iterkeys()
            if any(key_matches_endpoint(key, endpoint) for endpoint in endpoints)
        ]
        label = "/growths and /scores cache entries"
    elif args.clear_liquidity_cache:
        keys = [key for key in cache.iterkeys() if key_matches_liquidity(key)]
        label = "FMP stable cache entries used for the US liquidity ranking"
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

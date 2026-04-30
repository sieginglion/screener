#!/usr/bin/env python3
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

    return any(("symbol", ticker) in params or ("stock_id", ticker) in params for params in param_sets)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear cache entries for one ticker.")
    parser.add_argument("ticker")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    keys = [key for key in cache.iterkeys() if key_matches_ticker(key, ticker)]

    for key in keys:
        del cache[key]

    print(f"removed {len(keys)} cache entries for {ticker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

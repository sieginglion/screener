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


def key_matches_endpoint(key, endpoint: str) -> bool:
    if not isinstance(key, tuple) or len(key) != 4:
        return False

    _, _, args, _ = key
    return bool(args and isinstance(args[0], str) and args[0].endswith(endpoint))


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear selected cache entries.")
    parser.add_argument("ticker", nargs="?", help="Ticker or stock_id to clear.")
    parser.add_argument(
        "--scores",
        action="store_true",
        help="Clear every cached /scores response.",
    )
    args = parser.parse_args()

    selected_modes = sum(bool(mode) for mode in (args.scores, args.ticker))
    if selected_modes != 1:
        parser.error("provide exactly one of: ticker or --scores")

    if args.scores:
        keys = [key for key in cache.iterkeys() if key_matches_endpoint(key, "/scores")]
        label = "/scores cache entries"
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

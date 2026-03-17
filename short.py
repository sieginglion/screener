#!/usr/bin/env python3
import json
import os
import sys

import httpx
from dotenv import load_dotenv

from config import MARGIN, MARKET, TOP_N_RESULTS, Q
from run_u_analyze import load_top_company_names, today_for_market

# TODO: rename MARGIN


def parse_score_tuple(body: str) -> tuple[float, float | None]:
    first, second = json.loads(body)
    return float(first), None if second is None else float(second)


def effective_score(score_tuple: tuple[float, float | None]) -> float:
    first, second = score_tuple
    if second is None:
        return first
    return (first + second) / 2.0


def fetch_symbol_score(
    client: httpx.Client, symbol: str, description: str, end_date: str
) -> tuple[str, str, float, float, float | None] | None:
    # TODO: don't handle error, assume body always correct, new session every request
    try:
        response = client.get(
            "http://localhost:8080/scores",
            headers={"Connection": "close"},
            params={
                "market": MARKET,
                "symbol": symbol,
                "end_date": end_date,
                "q": Q,
            },
        )
    except httpx.HTTPError as exc:
        sys.stderr.write(f"Skipping {symbol}: /scores request failed ({exc})\n")
        return None

    if response.status_code != 200:
        sys.stderr.write(
            f"Skipping {symbol}: /scores returned {response.status_code}\n"
        )
        return None

    body = response.text.strip()
    if not body:
        sys.stderr.write(f"Skipping {symbol}: /scores returned an empty body\n")
        return None

    try:
        first, second = parse_score_tuple(body)
    except Exception as exc:
        sys.stderr.write(f"Skipping {symbol}: invalid /scores payload ({exc})\n")
        return None

    return symbol, description, effective_score((first, second)), first, second


def iter_symbol_scores_sequentially(
    client: httpx.Client, stocks: list[tuple[str, str]], end_date: str
) -> list[tuple[str, str, float, float, float | None]]:
    results: list[tuple[str, str, float, float, float | None]] = []
    for index, (symbol, description) in enumerate(stocks):
        item = fetch_symbol_score(client, symbol, description, end_date)
        if item is not None:
            results.append(item)
    return results


def main() -> int:
    load_dotenv()
    today = today_for_market().isoformat()
    api_key = os.environ.get("FMP_API_KEY")

    rows = load_top_company_names(
        api_key=api_key,
        top_n_symbols=TOP_N_RESULTS * MARGIN,
        top_n_results=TOP_N_RESULTS,
    )
    stocks = [tuple(row.split(" ", 1)) for row in rows]

    sys.stderr.write(f"Fetching scores for {today}...\n")
    with httpx.Client(
        limits=httpx.Limits(max_keepalive_connections=0),
    ) as client:
        results = iter_symbol_scores_sequentially(client, stocks, today)

    results.sort(key=lambda item: item[2], reverse=True)

    for symbol, description, score, first, second in results:
        second_text = "" if second is None else f"{second:.6f}"
        print(f"{symbol};{description};{score:.6f};{first:.6f};{second_text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

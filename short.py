#!/usr/bin/env python3
import json
import os
import sys
import time

import httpx
from dotenv import load_dotenv

from config import CANDIDATE_POOL_MULTIPLIER, MARKET, RESULT_LIMIT, SCORES_SLEEP, Q
from run_u_analyze import load_top_company_names, today_for_market


def parse_score_tuple(body: str) -> tuple[float, float | None]:
    first, second = json.loads(body)
    return float(first), None if second is None else float(second)


def effective_score(score_tuple: tuple[float, float | None]) -> float:
    first, second = score_tuple
    if second is None:
        return first
    return (first + second) / 2.0
    # return max(first, second)


def fetch_symbol_score(
    symbol: str, description: str, end_date: str
) -> tuple[str, str, float, float, float | None] | None:
    with httpx.Client() as client:
        response = client.get(
            "http://localhost:8080/scores",
            params={
                "market": MARKET,
                "symbol": symbol,
                "end_date": end_date,
                "q": Q,
            },
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            if response.status_code == 500:
                sys.stderr.write(f"Skipping {symbol}: score service returned 500\n")
                return None
            raise

    first, second = parse_score_tuple(response.text)
    return symbol, description, effective_score((first, second)), first, second


def iter_symbol_scores_sequentially(
    stocks: list[tuple[str, str]], end_date: str
) -> list[tuple[str, str, float, float, float | None]]:
    results: list[tuple[str, str, float, float, float | None]] = []
    for index, (symbol, description) in enumerate(stocks):
        result = fetch_symbol_score(symbol, description, end_date)
        if result is not None:
            results.append(result)
        if index < len(stocks) - 1:
            time.sleep(SCORES_SLEEP)
    return results


def main() -> int:
    load_dotenv()
    today = today_for_market().isoformat()
    api_key = os.environ.get("FMP_API_KEY")

    rows = load_top_company_names(
        api_key=api_key,
        top_n_symbols=RESULT_LIMIT * CANDIDATE_POOL_MULTIPLIER,
        top_n_results=RESULT_LIMIT,
    )
    stocks = [tuple(row.split(" ", 1)) for row in rows]

    sys.stderr.write(f"Fetching scores for {today}...\n")
    results = iter_symbol_scores_sequentially(stocks, today)

    results.sort(key=lambda item: item[2], reverse=True)

    for symbol, description, score, first, second in results:
        second_text = "" if second is None else f"{second:.6f}"
        print(f"{symbol} {description};{score:.6f};{first:.6f};{second_text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

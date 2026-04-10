#!/usr/bin/env python3
import json
import math
import os
import sys
import time

import httpx
from dotenv import load_dotenv

from config import CANDIDATE_POOL_MULTIPLIER, MARKET, PEAK_CUTOFF_RATIO, RESULT_LIMIT, Q
from run_u_analyze import load_top_company_names


def parse_score_tuple(body: str) -> tuple[float, float | None]:
    first, second = json.loads(body)
    return float(first), None if second is None else float(second)


def effective_score(score_tuple: tuple[float, float | None]) -> float:
    first, second = score_tuple
    if second is None:
        return first
    return (first + second) / 2.0


def cutoff_count(size: int) -> int:
    if size <= 0:
        return 0
    return max(1, math.ceil(size * PEAK_CUTOFF_RATIO))


def fetch_growth(client: httpx.Client, symbol: str) -> float:
    response = client.get(
        "http://localhost:8080/growth",
        params={
            "market": MARKET,
            "symbol": symbol,
        },
    )
    response.raise_for_status()
    return float(response.text.strip())


def fetch_score(client: httpx.Client, symbol: str) -> tuple[float, float, float | None]:
    response = client.get(
        "http://localhost:8080/scores",
        params={
            "market": MARKET,
            "symbol": symbol,
            "q": Q,
        },
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        if response.status_code == 500:
            raise ValueError("score service returned 500") from None
        raise

    first, second = parse_score_tuple(response.text)
    return effective_score((first, second)), first, second


def main() -> int:
    load_dotenv()

    api_key = os.environ.get("FMP_API_KEY")

    rows = load_top_company_names(
        api_key=api_key,
        top_n_symbols=RESULT_LIMIT * CANDIDATE_POOL_MULTIPLIER,
        top_n_results=RESULT_LIMIT,
    )
    stocks = [tuple(row.split(" ", 1)) for row in rows]

    if not stocks:
        return 0

    sys.stderr.write(f"Fetching growth for {len(stocks)} stocks...\n")
    growth_results: list[tuple[str, str, float]] = []
    with httpx.Client(timeout=30.0) as client:
        for index, (symbol, description) in enumerate(stocks, start=1):
            try:
                growth = fetch_growth(client, symbol)
            except Exception as exc:
                sys.stderr.write(f"Skipping {symbol}: {exc}\n")
            else:
                growth_results.append((symbol, description, growth))

    growth_cutoff = cutoff_count(len(stocks))
    growth_results.sort(key=lambda item: item[2], reverse=True)
    top_growth = growth_results[:growth_cutoff]

    if not top_growth:
        return 0

    sys.stderr.write(f"Scoring top {len(top_growth)} stocks by growth...\n")
    final_results: list[tuple[str, str, float, float, float, float | None]] = []
    with httpx.Client(timeout=30.0) as client:
        for index, (symbol, description, growth) in enumerate(top_growth, start=1):
            try:
                score, first, second = fetch_score(client, symbol)
            except Exception as exc:
                sys.stderr.write(f"Skipping {symbol}: {exc}\n")
            else:
                final_results.append(
                    (symbol, description, growth, score, first, second)
                )

            if index < len(top_growth):
                time.sleep(0.5)

    score_cutoff = cutoff_count(len(top_growth))
    final_results.sort(key=lambda item: item[3])

    for symbol, description, growth, score, first, second in final_results[
        :score_cutoff
    ]:
        second_text = "" if second is None else f"{second:.6f}"
        print(
            f"{symbol} {description};{growth:.6f};{score:.6f};{first:.6f};{second_text}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

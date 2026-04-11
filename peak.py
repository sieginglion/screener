#!/usr/bin/env python3
import json
import math
import os
import sys
import time

import httpx
from dotenv import load_dotenv

from config import CANDIDATE_POOL_MULTIPLIER, MARKET, PEAK_CUTOFF_RATIO, RESULT_LIMIT, Q
from run_u_analyze import cached, load_top_company_names


def cutoff_count(size: int) -> int:
    if size <= 0:
        return 0
    return max(1, math.ceil(size * PEAK_CUTOFF_RATIO))


@cached(43200)
def fetch_growth(symbol: str) -> float:
    res = httpx.get(
        "http://localhost:8080/growth",
        params={"market": MARKET, "symbol": symbol},
    )
    res.raise_for_status()
    return float(res.text)


@cached(43200)
def fetch_score(symbol: str) -> tuple[float, float | None]:
    time.sleep(0.5)
    res = httpx.get(
        "http://localhost:8080/scores",
        params={"market": MARKET, "symbol": symbol, "q": Q},
    )
    res.raise_for_status()
    return res.json()


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
    for index, (symbol, description) in enumerate(stocks, start=1):
        try:
            growth = fetch_growth(symbol)
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
    for index, (symbol, description, growth) in enumerate(top_growth, start=1):
        try:
            first, second = fetch_score(symbol)
            if second is None:
                raise ValueError
            score = (first + second) / 2
        except Exception as exc:
            sys.stderr.write(f"Skipping {symbol}: {exc}\n")
        else:
            final_results.append((symbol, description, growth, score, first, second))

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

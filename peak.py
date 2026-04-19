#!/usr/bin/env python3
import math
import os
import sys

from dotenv import load_dotenv

from config import (
    CANDIDATE_POOL_MULTIPLIER,
    DIRECTION,
    MARKET,
    PEAK_CUTOFF_RATIO,
    RESULT_LIMIT,
    Q,
)
from run_u_analyze import cached_httpx_get, load_top_company_names


def cutoff_count(size: int) -> int:
    if size <= 0:
        return 0
    return max(1, math.ceil(size * PEAK_CUTOFF_RATIO))


def sort_directions() -> tuple[bool, bool]:
    if DIRECTION == "high_growth_low_valuation":
        return True, False
    if DIRECTION == "low_growth_high_valuation":
        return False, True
    raise ValueError(f"Unsupported DIRECTION: {DIRECTION}")


def growth_label() -> str:
    if DIRECTION == "high_growth_low_valuation":
        return "growth"
    if DIRECTION == "low_growth_high_valuation":
        return "low growth"
    raise ValueError(f"Unsupported DIRECTION: {DIRECTION}")


def combine_pair(first: float, second: float | None, missing_message: str) -> float:
    if second is None:
        if DIRECTION == "low_growth_high_valuation":
            return first
        raise ValueError(missing_message)
    return (first + second) / 2


def fetch_growth(symbol: str) -> tuple[float, float | None]:
    res = cached_httpx_get(
        "http://localhost:8080/growths",
        params=[
            ("market", MARKET),
            ("symbol", symbol),
        ],
    )
    return res.json()


def fetch_score(symbol: str) -> tuple[float, float | None]:
    res = cached_httpx_get(
        "http://localhost:8080/scores",
        params=[
            ("market", MARKET),
            ("symbol", symbol),
            ("q", Q),
        ],
    )
    return res.json()


def main() -> int:
    load_dotenv()
    growth_desc, score_desc = sort_directions()
    growth_mode_label = growth_label()

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
    for symbol, description in stocks:
        try:
            revenue_per_share_growth, eps_growth = fetch_growth(symbol)
            growth = combine_pair(
                revenue_per_share_growth,
                eps_growth,
                "growth response missing eps growth",
            )
        except Exception as exc:
            sys.stderr.write(
                f"Skipping {symbol} during growth fetch: "
                f"{type(exc).__name__}: {exc}\n"
            )
        else:
            growth_results.append((symbol, description, growth))

    growth_cutoff = cutoff_count(len(stocks))
    growth_results.sort(key=lambda item: item[2], reverse=growth_desc)
    top_growth = growth_results[:growth_cutoff]

    if not top_growth:
        return 0

    sys.stderr.write(
        f"Scoring top {len(top_growth)} stocks by {growth_mode_label}...\n"
    )
    final_results: list[tuple[str, str, float]] = []
    for symbol, description, _ in top_growth:
        try:
            first, second = fetch_score(symbol)
            score = combine_pair(first, second, "score response missing second value")
        except Exception as exc:
            sys.stderr.write(
                f"Skipping {symbol} during score fetch: "
                f"{type(exc).__name__}: {exc}\n"
            )
        else:
            final_results.append((symbol, description, score))

    score_cutoff = cutoff_count(len(top_growth))
    final_results.sort(key=lambda item: item[2], reverse=score_desc)

    for symbol, description, _ in final_results[:score_cutoff]:
        print(f"{symbol} {description}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import csv
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

from config import (
    CANDIDATE_POOL_MULTIPLIER,
    MARKET,
    PEAK_CUTOFF_RATIO,
    RESULT_LIMIT,
    Q,
)
from dotenv import load_dotenv
from run_u_analyze import cached_httpx_get, load_top_company_names

SCORING_BASE_URL = "http://localhost:8080"
ENABLE_BTC = False
BTC_GROWTH_MULTIPLIER = 0.75
GROWTH_CONCURRENCY = 2
VALUATION_CONCURRENCY = 1


def scalar_score(value):
    return float(value)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values)


@dataclass(frozen=True)
class Candidate:
    symbol: str
    description: str


@dataclass(frozen=True)
class GrowthCandidate:
    candidate: Candidate
    growth_score: float


@dataclass(frozen=True)
class ValuationCandidate:
    candidate: Candidate
    growth_score: float
    valuation_score: float

    @property
    def power(self) -> float:
        return self.growth_score * (0.5 - self.valuation_score)


def parse_candidate(row: str) -> Candidate:
    symbol, description = row.split(maxsplit=1)
    return Candidate(symbol=symbol, description=description)


def candidate_market(symbol: str) -> str:
    if symbol == "BTC":
        return "c"
    return MARKET


def adjusted_growth_score(symbol: str, score: float) -> float:
    if symbol == "BTC":
        return score * BTC_GROWTH_MULTIPLIER
    return score


def fetch_score(path: str, params: dict[str, str | int], selector) -> float:
    response = cached_httpx_get(
        f"{SCORING_BASE_URL}/{path}",
        params=list(params.items()),
    )
    values = response.json()
    return selector(values)


def score_growth(candidate: Candidate) -> GrowthCandidate | None:
    if candidate.symbol == "BTC" and not ENABLE_BTC:
        sys.stderr.write(f"Skipping {candidate.symbol}: disabled symbol\n")
        return None

    try:
        score = fetch_score(
            "growths",
            {
                "market": candidate_market(candidate.symbol),
                "symbol": candidate.symbol,
            },
            scalar_score,
        )
    except Exception as exc:
        sys.stderr.write(f"Skipping {candidate.symbol} growth score: {exc}\n")
        return None

    return GrowthCandidate(
        candidate=candidate,
        growth_score=adjusted_growth_score(candidate.symbol, score),
    )


def score_valuation(candidate: GrowthCandidate) -> ValuationCandidate | None:
    try:
        score = fetch_score(
            "scores",
            {
                "market": candidate_market(candidate.candidate.symbol),
                "symbol": candidate.candidate.symbol,
                "q": Q,
            },
            mean,
        )
    except Exception as exc:
        sys.stderr.write(
            f"Skipping {candidate.candidate.symbol} valuation score: {exc}\n"
        )
        return None

    return ValuationCandidate(
        candidate=candidate.candidate,
        growth_score=candidate.growth_score,
        valuation_score=score,
    )


def keep_count(attempt_count: int) -> int:
    return math.ceil(attempt_count * PEAK_CUTOFF_RATIO)


def keep_top_growth(candidates: Iterable[Candidate]) -> list[GrowthCandidate]:
    attempted = list(candidates)
    with ThreadPoolExecutor(max_workers=GROWTH_CONCURRENCY) as executor:
        scored = [
            scored_candidate
            for scored_candidate in executor.map(score_growth, attempted)
            if scored_candidate is not None
        ]
    scored.sort(key=lambda candidate: candidate.growth_score, reverse=True)
    return scored[: keep_count(len(attempted))]


def score_all_valuations(
    candidates: Iterable[GrowthCandidate],
) -> list[ValuationCandidate]:
    attempted = list(candidates)
    with ThreadPoolExecutor(max_workers=VALUATION_CONCURRENCY) as executor:
        scored = [
            scored_candidate
            for scored_candidate in executor.map(score_valuation, attempted)
            if scored_candidate is not None
        ]
    scored.sort(key=lambda candidate: candidate.valuation_score)
    return scored


def load_candidates() -> list[Candidate]:
    pool_size = math.ceil(RESULT_LIMIT * CANDIDATE_POOL_MULTIPLIER)
    # Fetch a wider TradingView pool, then let run_u_analyze keep the most
    # liquid RESULT_LIMIT candidates after individual trading-volume checks.
    rows = load_top_company_names(
        api_key=os.environ.get("FMP_API_KEY"),
        top_n_symbols=pool_size,
        top_n_results=RESULT_LIMIT,
    )
    candidates = [parse_candidate(row) for row in rows]

    if MARKET == "u" and ENABLE_BTC:
        candidates.append(Candidate(symbol="BTC", description="Bitcoin"))

    return candidates


def write_results(results: Iterable[ValuationCandidate]) -> None:
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["symbol", "description", "power"])
    for result in results:
        writer.writerow(
            [
                result.candidate.symbol,
                result.candidate.description,
                f"{result.power:.6g}",
            ]
        )


def main() -> int:
    load_dotenv()

    growth_survivors = keep_top_growth(load_candidates())
    final_survivors = score_all_valuations(growth_survivors)
    final_survivors.sort(key=lambda candidate: candidate.power, reverse=True)

    write_results(final_survivors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

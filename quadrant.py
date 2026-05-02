#!/usr/bin/env python3
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable

from dotenv import load_dotenv

from config import (
    CANDIDATE_POOL_MULTIPLIER,
    MARKET,
    PEAK_CUTOFF_RATIO,
    Q,
    RESULT_LIMIT,
)
from run_u_analyze import cached_httpx_get, load_top_company_names

SCORING_BASE_URL = "http://localhost:8080"


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
        return self.growth_score * (0.625 - self.valuation_score)


def parse_candidate(row: str) -> Candidate:
    symbol, description = row.split(maxsplit=1)
    return Candidate(symbol=symbol, description=description)


def candidate_market(symbol: str) -> str:
    if symbol == "BTC":
        return "c"
    return MARKET


def fetch_first_score(path: str, params: dict[str, str | int]) -> float:
    response = cached_httpx_get(
        f"{SCORING_BASE_URL}/{path}",
        params=list(params.items()),
    )
    values = response.json()
    return float(values[0])


def score_growth(candidate: Candidate) -> GrowthCandidate | None:
    try:
        score = fetch_first_score(
            "growths",
            {
                "market": candidate_market(candidate.symbol),
                "symbol": candidate.symbol,
            },
        )
    except Exception as exc:
        sys.stderr.write(f"Skipping {candidate.symbol} growth score: {exc}\n")
        return None

    return GrowthCandidate(candidate=candidate, growth_score=score)


def score_valuation(candidate: GrowthCandidate) -> ValuationCandidate | None:
    try:
        score = fetch_first_score(
            "scores",
            {
                "market": candidate_market(candidate.candidate.symbol),
                "symbol": candidate.candidate.symbol,
                "q": Q,
            },
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
    scored = [
        scored_candidate
        for candidate in attempted
        if (scored_candidate := score_growth(candidate)) is not None
    ]
    scored.sort(key=lambda candidate: candidate.growth_score, reverse=True)
    return scored[: keep_count(len(attempted))]


def keep_lowest_valuation(
    candidates: Iterable[GrowthCandidate],
) -> list[ValuationCandidate]:
    attempted = list(candidates)
    scored = [
        scored_candidate
        for candidate in attempted
        if (scored_candidate := score_valuation(candidate)) is not None
    ]
    scored.sort(key=lambda candidate: candidate.valuation_score)
    return scored[: keep_count(len(attempted))]


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

    if MARKET == "u":
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
    final_survivors = keep_lowest_valuation(growth_survivors)
    final_survivors.sort(key=lambda candidate: candidate.power, reverse=True)

    write_results(final_survivors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

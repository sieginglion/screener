#!/usr/bin/env python3
import csv
import datetime as dt
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import httpx
from config import (
    BTC_GROWTH_MULTIPLIER,
    CANDIDATE_POOL_MULTIPLIER,
    FINMIND_THREADS,
    GROWTH_CONCURRENCY,
    LAST_N,
    LOOKBACK_DAYS,
    MARKET,
    RESULT_LIMIT,
    TV_SORT_WINDOW,
    Q,
)
from diskcache import Cache
from dotenv import load_dotenv

SCORING_BASE_URL = "http://localhost:8080"
FMP_STABLE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"
FMP_LEGACY_URL = "https://financialmodelingprep.com/api/v3/historical-price-full"
TV_US_URL = "https://scanner.tradingview.com/america/scan?label-product=screener-stock"
TV_TW_URL = "https://scanner.tradingview.com/taiwan/scan?label-product=screener-stock"
TV_JP_URL = "https://scanner.tradingview.com/japan/scan?label-product=screener-stock"
cache = Cache(Path().resolve() / ".cache")
finmind_api = None

MARKET_TIMEZONE = {
    "u": "America/New_York",
    "t": "Asia/Taipei",
    "j": "Asia/Tokyo",
}


def candidate_pool_size(result_limit: int, multiplier: float) -> int:
    return math.ceil(result_limit * multiplier)


def validate_candidate_pool_sizes(top_n_symbols: int, top_n_results: int) -> None:
    if top_n_symbols <= top_n_results:
        raise ValueError(
            "top_n_symbols must be greater than top_n_results to allow liquidity reranking "
            f"(got top_n_symbols={top_n_symbols}, top_n_results={top_n_results})"
        )


def cached(ttl):
    def decorator(f):
        def sync_wrapper(*args, **kwargs):
            k = (f.__module__, f.__qualname__, args, tuple(sorted(kwargs.items())))
            if k in cache:
                return cache[k]
            v = f(*args, **kwargs)
            cache.set(k, v, ttl)
            return v

        return sync_wrapper

    return decorator


@cached(43200)
def cached_httpx_get(url: str, params: list[tuple[str, str | int]]) -> httpx.Response:
    res = httpx.get(url, params=dict(params), timeout=None)
    res.raise_for_status()
    return res


def today_for_market() -> dt.date:
    timezone = MARKET_TIMEZONE.get(MARKET, MARKET_TIMEZONE["u"])
    return dt.datetime.now(ZoneInfo(timezone)).date()


TV_US_HEADERS = {
    "accept": "application/json",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "content-type": "text/plain;charset=UTF-8",
    "origin": "https://www.tradingview.com",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://www.tradingview.com/",
    "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
}

TV_TW_HEADERS = {
    "accept": "application/json",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "content-type": "text/plain;charset=UTF-8",
    "origin": "https://tw.tradingview.com",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://tw.tradingview.com/",
    "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
}

TV_JP_HEADERS = {
    "accept": "application/json",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "content-type": "text/plain;charset=UTF-8",
    "origin": "https://jp.tradingview.com",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://jp.tradingview.com/",
    "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
}


@dataclass(frozen=True)
class TradingViewConfig:
    url: str
    headers: dict[str, str]
    market_name: str
    language: str


TV_CONFIG_BY_MARKET = {
    "u": TradingViewConfig(TV_US_URL, TV_US_HEADERS, "america", "en"),
    "t": TradingViewConfig(TV_TW_URL, TV_TW_HEADERS, "taiwan", "zh_TW"),
    "j": TradingViewConfig(TV_JP_URL, TV_JP_HEADERS, "japan", "ja"),
}


def type_filter(stock_type: str, *typespecs: str) -> dict:
    operands = [
        {
            "expression": {
                "left": "type",
                "operation": "equal",
                "right": stock_type,
            }
        }
    ]
    for typespec in typespecs:
        operands.append(
            {
                "expression": {
                    "left": "typespecs",
                    "operation": "has",
                    "right": [typespec],
                }
            }
        )

    return {"operation": {"operator": "and", "operands": operands}}


def tradingview_payload(market_name: str, language: str, limit: int) -> dict:
    return {
        "columns": ["ticker-view"],
        "filter": [{"left": "is_primary", "operation": "equal", "right": True}],
        "ignore_unknown_fields": False,
        "options": {"lang": language},
        "range": [0, limit],
        "sort": {"sortBy": f"Value.Traded|{TV_SORT_WINDOW}", "sortOrder": "desc"},
        "markets": [market_name],
        "filter2": {
            "operator": "and",
            "operands": [
                {
                    "operation": {
                        "operator": "or",
                        "operands": [
                            type_filter("stock", "common"),
                            type_filter("stock", "preferred"),
                            type_filter("dr"),
                            {
                                "operation": {
                                    "operator": "and",
                                    "operands": [
                                        {
                                            "expression": {
                                                "left": "type",
                                                "operation": "equal",
                                                "right": "fund",
                                            }
                                        },
                                        {
                                            "expression": {
                                                "left": "typespecs",
                                                "operation": "has_none_of",
                                                "right": ["etf", "mutual"],
                                            }
                                        },
                                    ],
                                }
                            },
                        ],
                    }
                },
                {
                    "expression": {
                        "left": "typespecs",
                        "operation": "has_none_of",
                        "right": ["pre-ipo"],
                    }
                },
            ],
        },
    }


def scalar_score(value: Any) -> float:
    return float(value)


def mean(values: Iterable[float | None]) -> float:
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
        return self.growth_score * self.valuation_score


def candidate_market(symbol: str) -> str:
    if symbol == "BTC":
        return "c"
    return MARKET


def adjusted_growth_score(symbol: str, score: float) -> float:
    if symbol == "BTC":
        return score * BTC_GROWTH_MULTIPLIER
    return score


def fetch_score(
    path: str,
    params: dict[str, str | int],
    selector: Callable[[Any], float],
) -> float:
    response = cached_httpx_get(
        f"{SCORING_BASE_URL}/{path}",
        params=list(params.items()),
    )
    values = response.json()
    return selector(values)


def current_tv_config() -> TradingViewConfig:
    return TV_CONFIG_BY_MARKET.get(MARKET, TV_CONFIG_BY_MARKET["u"])


def score_growth(candidate: Candidate) -> GrowthCandidate | None:
    try:
        score = fetch_score(
            "growth",
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


def parse_candidate(row: str) -> Candidate:
    symbol, description = row.split(maxsplit=1)
    return Candidate(symbol=symbol, description=description)


def fetch_symbols_from_tv(
    tv_config: TradingViewConfig,
    top_n_symbols: int,
) -> list[tuple[str, str]]:
    payload = tradingview_payload(
        tv_config.market_name,
        tv_config.language,
        top_n_symbols,
    )
    with httpx.Client() as client:
        response = client.post(tv_config.url, headers=tv_config.headers, json=payload)
        response.raise_for_status()
        data = response.json()

        def normalize(name: str) -> str:
            return name.replace(".", "-")

        return [
            (normalize(item["d"][0]["name"]), item["d"][0]["description"])
            for item in data["data"]
        ]


def fetch_trading_dollar_fmp(
    symbol: str, description: str, from_date: str, api_key: str
) -> tuple[str, str, float]:
    fmp_symbol = f"{symbol}.T" if MARKET == "j" else symbol
    res = cached_httpx_get(
        f"{FMP_LEGACY_URL}/{fmp_symbol}" if MARKET == "j" else FMP_STABLE_URL,
        params=[
            ("apikey", api_key),
            ("from", from_date),
            *([] if MARKET == "j" else [("symbol", fmp_symbol)]),
        ],
    )
    data = res.json()
    rows = data.get("historical", []) if MARKET == "j" else data

    if len(rows) < LAST_N:
        sys.stderr.write(
            f"Skipping {symbol} trading dollar data: only {len(rows)} FMP rows, need {LAST_N}\n"
        )
        return symbol, description, 0

    rows = sorted(rows, key=lambda x: x["date"], reverse=True)[:LAST_N]
    return symbol, description, sum(row["vwap"] * row["volume"] for row in rows)


def top_company_names(
    results: Iterable[tuple[str, str, float]],
    limit: int,
) -> list[str]:
    top = sorted(results, key=lambda x: x[2], reverse=True)[:limit]
    return [
        f"{symbol} {description.replace(';', ',')}" for symbol, description, _ in top
    ]


def load_top_company_names_fmp(
    api_key: str | None,
    top_n_symbols: int,
    top_n_results: int,
    tv_config: TradingViewConfig,
) -> list[str]:
    validate_candidate_pool_sizes(top_n_symbols, top_n_results)

    sys.stderr.write("Fetching symbols from TradingView...\n")
    stocks = fetch_symbols_from_tv(
        tv_config=tv_config,
        top_n_symbols=top_n_symbols,
    )
    sys.stderr.write(f"Found {len(stocks)} symbols\n")

    from_date = (today_for_market() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    sys.stderr.write("Fetching trading dollar data from FMP...\n")
    results = [
        fetch_trading_dollar_fmp(symbol, description, from_date, api_key)
        for symbol, description in stocks
    ]

    return top_company_names(results, top_n_results)


@cached(43200)
def fetch_taiwan_stock_daily(stock_id: str, start: str, end: str):
    df = finmind_api.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=start,
        end_date=end,
    )
    return df


def fetch_trading_dollar_tw(
    stock_id: str, description: str, start: str, end: str
) -> tuple[str, str, float]:
    df = fetch_taiwan_stock_daily(stock_id, start, end)
    total = df.sort_values("date", ascending=False).head(LAST_N)["Trading_money"].sum()
    return stock_id, description, float(total)


def load_top_company_names_tw(
    top_n_symbols: int, top_n_results: int, tv_config: TradingViewConfig
) -> list[str]:
    global finmind_api

    validate_candidate_pool_sizes(top_n_symbols, top_n_results)

    finmind_key = os.environ.get("FINMIND_KEY")

    from FinMind.data import DataLoader

    finmind_api = DataLoader()
    finmind_api.login_by_token(finmind_key)

    sys.stderr.write("Fetching symbols from TradingView...\n")
    stocks = fetch_symbols_from_tv(
        tv_config=tv_config,
        top_n_symbols=top_n_symbols,
    )
    sys.stderr.write(f"Found {len(stocks)} symbols\n")

    today = today_for_market()
    start = (today - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    end = today.isoformat()

    sys.stderr.write("Fetching trading dollar data from FinMind...\n")
    with ThreadPoolExecutor(max_workers=FINMIND_THREADS) as pool:
        results = list(
            pool.map(
                lambda s: fetch_trading_dollar_tw(s[0], s[1], start, end),
                stocks,
            )
        )

    return top_company_names(results, top_n_results)


def load_top_company_names(
    api_key: str | None, top_n_symbols: int, top_n_results: int
) -> list[str]:
    tv_config = current_tv_config()
    if MARKET == "t":
        return load_top_company_names_tw(top_n_symbols, top_n_results, tv_config)
    return load_top_company_names_fmp(
        api_key,
        top_n_symbols,
        top_n_results,
        tv_config,
    )


def score_growth_candidates(candidates: Iterable[Candidate]) -> list[GrowthCandidate]:
    attempted = list(candidates)
    with ThreadPoolExecutor(max_workers=GROWTH_CONCURRENCY) as executor:
        scored = [
            scored_candidate
            for scored_candidate in executor.map(score_growth, attempted)
            if scored_candidate is not None
        ]
    scored.sort(key=lambda candidate: candidate.growth_score, reverse=True)
    return scored


def score_all_valuations(
    candidates: Iterable[GrowthCandidate],
) -> list[ValuationCandidate]:
    scored = [
        scored_candidate
        for scored_candidate in (score_valuation(candidate) for candidate in candidates)
        if scored_candidate is not None
    ]
    return scored


def load_candidates() -> list[Candidate]:
    pool_size = candidate_pool_size(RESULT_LIMIT, CANDIDATE_POOL_MULTIPLIER)
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
    for result in results:
        writer.writerow(
            [
                result.candidate.symbol,
                result.candidate.description,
            ]
        )


def main() -> int:
    load_dotenv()

    growth_candidates = score_growth_candidates(load_candidates())
    scored_candidates = score_all_valuations(growth_candidates)
    ranked_candidates = sorted(
        (candidate for candidate in scored_candidates if candidate.power >= 0),
        key=lambda candidate: candidate.power,
        reverse=True,
    )

    write_results(ranked_candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

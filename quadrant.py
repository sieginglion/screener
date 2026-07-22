#!/usr/bin/env python3
import csv
import datetime as dt
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import httpx
from diskcache import Cache
from dotenv import load_dotenv

from config import (
    BTC_GROWTH_MULTIPLIER,
    CANDIDATE_POOL_MULTIPLIER,
    FINMIND_THREADS,
    GROWTH_CONCURRENCY,
    GROWTH_ENABLED,
    LAST_N,
    MARKET,
    RESULT_LIMIT,
    TV_SORT_WINDOW,
    Q,
)

SCORING_BASE_URL = "http://localhost:8080"
FMP_STABLE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"
FMP_LEGACY_URL = "https://financialmodelingprep.com/api/v3/historical-price-full"
LOOKBACK_DAYS = 56
NEW_YORK_TIMEZONE = ZoneInfo("America/New_York")
TV_US_URL = "https://scanner.tradingview.com/america/scan?label-product=screener-stock"
TV_TW_URL = "https://scanner.tradingview.com/taiwan/scan?label-product=screener-stock"
TV_JP_URL = "https://scanner.tradingview.com/japan/scan?label-product=screener-stock"
cache = Cache(Path().resolve() / ".cache")
finmind_api: Any = None


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
def cached_get_json(url: str, params: list[tuple[str, str | int]]) -> Any:
    response = httpx.get(url, params=dict(params), timeout=None)
    response.raise_for_status()
    return response.json()


def new_york_regular_session_in_progress(now: dt.datetime | None = None) -> bool:
    now = (now or dt.datetime.now(NEW_YORK_TIMEZONE)).astimezone(NEW_YORK_TIMEZONE)
    return dt.time(9, 30) <= now.time() < dt.time(16)


TV_COMMON_HEADERS = {
    "accept": "application/json",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "content-type": "text/plain;charset=UTF-8",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
}

TV_US_HEADERS = {
    **TV_COMMON_HEADERS,
    "origin": "https://www.tradingview.com",
    "referer": "https://www.tradingview.com/",
    "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
}

TV_TW_HEADERS = {
    **TV_COMMON_HEADERS,
    "origin": "https://tw.tradingview.com",
    "referer": "https://tw.tradingview.com/",
    "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
}

TV_JP_HEADERS = {
    **TV_COMMON_HEADERS,
    "origin": "https://jp.tradingview.com",
    "referer": "https://jp.tradingview.com/",
    "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
}


@dataclass(frozen=True)
class MarketConfig:
    code: str
    timezone: str
    tradingview_url: str
    tradingview_headers: dict[str, str]
    tradingview_market: str
    language: str
    liquidity_source: str
    fmp_legacy: bool = False
    fmp_symbol_suffix: str = ""
    exclude_intraday_fmp_row: bool = False
    include_bitcoin: bool = False


MARKET_CONFIG_BY_CODE = {
    "u": MarketConfig(
        code="u",
        timezone="America/New_York",
        tradingview_url=TV_US_URL,
        tradingview_headers=TV_US_HEADERS,
        tradingview_market="america",
        language="en",
        liquidity_source="fmp",
        exclude_intraday_fmp_row=True,
        include_bitcoin=True,
    ),
    "t": MarketConfig(
        code="t",
        timezone="Asia/Taipei",
        tradingview_url=TV_TW_URL,
        tradingview_headers=TV_TW_HEADERS,
        tradingview_market="taiwan",
        language="zh_TW",
        liquidity_source="finmind",
    ),
    "j": MarketConfig(
        code="j",
        timezone="Asia/Tokyo",
        tradingview_url=TV_JP_URL,
        tradingview_headers=TV_JP_HEADERS,
        tradingview_market="japan",
        language="ja",
        liquidity_source="fmp",
        fmp_legacy=True,
        fmp_symbol_suffix=".T",
    ),
}


def current_market_config() -> MarketConfig:
    return MARKET_CONFIG_BY_CODE.get(MARKET, MARKET_CONFIG_BY_CODE["u"])


def today_for_market(market: MarketConfig) -> dt.date:
    return dt.datetime.now(ZoneInfo(market.timezone)).date()


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


def tradingview_payload(market: MarketConfig, limit: int) -> dict:
    return {
        "columns": ["ticker-view"],
        "filter": [{"left": "is_primary", "operation": "equal", "right": True}],
        "ignore_unknown_fields": False,
        "options": {"lang": market.language},
        "range": [0, limit],
        "sort": {"sortBy": f"Value.Traded|{TV_SORT_WINDOW}", "sortOrder": "desc"},
        "markets": [market.tradingview_market],
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


def mean(values: Iterable[float | None]) -> float:
    values = [value for value in values if value is not None]
    return sum(values) / len(values)


@dataclass(frozen=True)
class Candidate:
    symbol: str
    description: str
    market: str = "u"
    growth_score: float | None = None
    valuation_score: float | None = None

    @property
    def power(self) -> float | None:
        if self.growth_score is None or self.valuation_score is None:
            return None
        return self.growth_score * self.valuation_score


@dataclass(frozen=True)
class LiquidityResult:
    symbol: str
    description: str
    trading_dollars: float | None


def adjusted_growth_score(symbol: str, score: float) -> float:
    if symbol == "BTC":
        return score * BTC_GROWTH_MULTIPLIER
    return score


def fetch_score(
    path: str,
    params: Mapping[str, str | int],
) -> Any:
    response = httpx.get(
        f"{SCORING_BASE_URL}/{path}",
        params=params,
        timeout=None,
    )
    response.raise_for_status()
    return response.json()


def report_scoring_failure(
    candidate: Candidate, score_name: str, exc: Exception
) -> None:
    sys.stderr.write(f"Skipping {candidate.symbol} {score_name}: {exc}\n")


def score_growth(candidate: Candidate) -> Candidate | None:
    try:
        score = float(
            fetch_score(
                "growth",
                {"market": candidate.market, "symbol": candidate.symbol},
            )
        )
    except Exception as exc:
        report_scoring_failure(candidate, "growth score", exc)
        return None

    return replace(
        candidate,
        growth_score=adjusted_growth_score(candidate.symbol, score),
    )


def score_valuation(candidate: Candidate) -> Candidate | None:
    try:
        score = mean(
            fetch_score(
                "scores",
                {"market": candidate.market, "symbol": candidate.symbol, "q": Q},
            )
        )
    except Exception as exc:
        report_scoring_failure(candidate, "valuation score", exc)
        return None

    return replace(candidate, valuation_score=score)


def fetch_symbols_from_tv(
    market: MarketConfig,
    top_n_symbols: int,
) -> list[tuple[str, str]]:
    payload = tradingview_payload(market, top_n_symbols)
    with httpx.Client() as client:
        response = client.post(
            market.tradingview_url,
            headers=market.tradingview_headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        def normalize(name: str) -> str:
            return name.replace(".", "-")

        return [
            (normalize(item["d"][0]["name"]), item["d"][0]["description"])
            for item in data["data"]
        ]


def recent_trading_dollars(
    rows: Iterable[tuple[str, float]],
    symbol: str,
) -> float | None:
    recent_rows = sorted(rows, key=lambda row: row[0], reverse=True)[:LAST_N]
    if len(recent_rows) < LAST_N:
        sys.stderr.write(
            f"Skipping {symbol} trading dollar data: only {len(recent_rows)} rows, need {LAST_N}\n"
        )
        return None
    return sum(notional for _, notional in recent_rows)


def fetch_trading_dollar_fmp(
    market: MarketConfig,
    symbol: str,
    description: str,
    from_date: str,
    api_key: str | None,
) -> LiquidityResult:
    fmp_symbol = f"{symbol}{market.fmp_symbol_suffix}"
    data = cached_get_json(
        f"{FMP_LEGACY_URL}/{fmp_symbol}" if market.fmp_legacy else FMP_STABLE_URL,
        params=[
            ("apikey", api_key),
            ("from", from_date),
            *([] if market.fmp_legacy else [("symbol", fmp_symbol)]),
        ],
    )
    rows = data.get("historical", []) if market.fmp_legacy else data

    if market.exclude_intraday_fmp_row and new_york_regular_session_in_progress():
        today = today_for_market(market).isoformat()
        rows = [row for row in rows if row["date"] != today]

    daily_notionals = [(row["date"], row["vwap"] * row["volume"]) for row in rows]
    return LiquidityResult(
        symbol=symbol,
        description=description,
        trading_dollars=recent_trading_dollars(daily_notionals, symbol),
    )


def top_candidates(
    results: Iterable[LiquidityResult],
    limit: int,
    market: str,
) -> list[Candidate]:
    top = sorted(
        (result for result in results if result.trading_dollars is not None),
        key=lambda result: result.trading_dollars or 0,
        reverse=True,
    )[:limit]
    return [
        Candidate(
            symbol=result.symbol,
            description=result.description,
            market=market,
        )
        for result in top
    ]


def fetch_trading_dollars_fmp(
    market: MarketConfig,
    stocks: Iterable[tuple[str, str]],
    api_key: str | None,
) -> list[LiquidityResult]:
    from_date = (
        today_for_market(market) - dt.timedelta(days=LOOKBACK_DAYS)
    ).isoformat()

    sys.stderr.write("Fetching trading dollar data from FMP...\n")
    return [
        fetch_trading_dollar_fmp(market, symbol, description, from_date, api_key)
        for symbol, description in stocks
    ]


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
) -> LiquidityResult:
    df = fetch_taiwan_stock_daily(stock_id, start, end)
    daily_notionals = [
        (str(row.date), float(row.Trading_money))
        for row in df[["date", "Trading_money"]].itertuples(index=False)
    ]
    return LiquidityResult(
        symbol=stock_id,
        description=description,
        trading_dollars=recent_trading_dollars(daily_notionals, stock_id),
    )


def fetch_trading_dollars_tw(
    market: MarketConfig,
    stocks: Iterable[tuple[str, str]],
) -> list[LiquidityResult]:
    global finmind_api

    finmind_key = os.environ.get("FINMIND_KEY")
    if not finmind_key:
        raise ValueError("FINMIND_KEY is required for Taiwanese liquidity data.")

    today = today_for_market(market)
    start = (today - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    end = today.isoformat()

    from FinMind.data import DataLoader

    finmind_api = DataLoader()
    finmind_api.login_by_token(finmind_key)

    sys.stderr.write("Fetching trading dollar data from FinMind...\n")
    with ThreadPoolExecutor(max_workers=FINMIND_THREADS) as pool:
        return list(
            pool.map(
                lambda s: fetch_trading_dollar_tw(s[0], s[1], start, end),
                stocks,
            )
        )


def load_top_candidates(
    market: MarketConfig,
    api_key: str | None,
    top_n_symbols: int,
    top_n_results: int,
) -> list[Candidate]:
    validate_candidate_pool_sizes(top_n_symbols, top_n_results)

    sys.stderr.write("Fetching symbols from TradingView...\n")
    stocks = fetch_symbols_from_tv(
        market=market,
        top_n_symbols=top_n_symbols,
    )
    sys.stderr.write(f"Found {len(stocks)} symbols\n")

    if market.liquidity_source == "finmind":
        liquidity = fetch_trading_dollars_tw(market, stocks)
    else:
        liquidity = fetch_trading_dollars_fmp(market, stocks, api_key)

    return top_candidates(liquidity, top_n_results, market.code)


def score_growth_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    attempted = list(candidates)
    if not GROWTH_ENABLED:
        return [replace(candidate, growth_score=1) for candidate in attempted]

    with ThreadPoolExecutor(max_workers=GROWTH_CONCURRENCY) as executor:
        scored = [
            scored_candidate
            for scored_candidate in executor.map(score_growth, attempted)
            if scored_candidate is not None
        ]
    return scored


def score_all_valuations(
    candidates: Iterable[Candidate],
) -> list[Candidate]:
    scored = [
        scored_candidate
        for scored_candidate in (score_valuation(candidate) for candidate in candidates)
        if scored_candidate is not None
    ]
    return scored


def load_candidates() -> list[Candidate]:
    market = current_market_config()
    pool_size = candidate_pool_size(RESULT_LIMIT, CANDIDATE_POOL_MULTIPLIER)
    candidates = load_top_candidates(
        market=market,
        api_key=os.environ.get("FMP_API_KEY"),
        top_n_symbols=pool_size,
        top_n_results=RESULT_LIMIT,
    )

    if market.include_bitcoin:
        candidates.append(Candidate(symbol="BTC", description="Bitcoin", market="c"))

    return candidates


def write_results(results: Iterable[Candidate]) -> None:
    writer = csv.writer(sys.stdout, lineterminator="\n")
    for result in results:
        writer.writerow(
            [
                result.symbol,
                result.description,
            ]
        )


def rank_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    return sorted(
        (
            candidate
            for candidate in candidates
            if candidate.power is not None and candidate.power > 0
        ),
        key=lambda candidate: candidate.power or 0,
        reverse=True,
    )


def main() -> int:
    load_dotenv()

    growth_candidates = score_growth_candidates(load_candidates())
    scored_candidates = score_all_valuations(growth_candidates)
    ranked_candidates = rank_candidates(scored_candidates)

    write_results(ranked_candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

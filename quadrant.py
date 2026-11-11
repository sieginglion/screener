#!/usr/bin/env python3
import csv
import datetime as dt
import math
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from diskcache import Cache
from dotenv import load_dotenv

import config
from config import (
    BTC_GROWTH_MULTIPLIER,
    FINMIND_THREADS,
    GROWTH_CONCURRENCY,
    LAST_N,
    MARKET,
    RESULT_LIMIT,
    Q,
)

SCORING_BASE_URL = "http://localhost:8080"
FMP_STABLE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"
FMP_LEGACY_URL = "https://financialmodelingprep.com/api/v3/historical-price-full"
LOOKBACK_DAYS = 61
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


def get_json(url: str, params: list[tuple[str, str | int | None]]) -> Any:
    response = httpx.get(url, params=dict(params), timeout=None)
    response.raise_for_status()
    return response.json()


@cached(43200)
def cached_get_json(url: str, params: list[tuple[str, str | int | None]]) -> Any:
    return get_json(url, params)


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


class ScreeningWindowError(ValueError):
    pass


@dataclass(frozen=True)
class ScreeningWindow:
    now: dt.datetime
    lookback_start: dt.datetime
    tv_sort_window: str
    tv_sort_start: dt.datetime
    trading_days_withing_last_n: int
    trading_days_withing_tv_sort_window: int
    candidate_pool_multiplier: float
    request_start: str
    request_end: str


def validate_last_n(last_n: object) -> None:
    if type(last_n) is not int or last_n not in (7, 14, 28):
        raise ScreeningWindowError("LAST_N must be an integer equal to 7, 14, or 28.")


def trading_close_time(market: MarketConfig, date: dt.date) -> dt.datetime:
    close = dt.time(16) if market.code == "u" else dt.time(13, 30)
    return dt.datetime.combine(date, close, tzinfo=ZoneInfo(market.timezone))


def select_screening_window(
    market: MarketConfig,
    now: dt.datetime,
    lookback_start: dt.datetime,
    trading_dates: Iterable[dt.date],
) -> ScreeningWindow:
    closes = [trading_close_time(market, date) for date in set(trading_dates)]
    closes = [close for close in closes if lookback_start <= close <= now]
    trading_days_withing_last_n = len(closes)
    if not trading_days_withing_last_n:
        raise ScreeningWindowError("No completed trading days within LAST_N.")

    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # LAST_N is at least 7, so the current week always fits. Keep it on ties.
    tv_sort_window = "1W"
    tv_sort_start = midnight - dt.timedelta(days=now.weekday())
    trading_days_withing_tv_sort_window = sum(
        close >= tv_sort_start for close in closes
    )
    month_start = midnight.replace(day=1)
    reason = "1M starts before the lookback boundary"
    if month_start >= lookback_start:
        month_days = sum(close >= month_start for close in closes)
        reason = (
            f"completed trading days: 1W={trading_days_withing_tv_sort_window}, "
            f"1M={month_days}; ties prefer 1W"
        )
        if month_days > trading_days_withing_tv_sort_window:
            tv_sort_window, tv_sort_start = "1M", month_start
            trading_days_withing_tv_sort_window = month_days
    if not trading_days_withing_tv_sort_window:
        raise ScreeningWindowError(
            "No completed trading days within the eligible TradingView windows."
        )
    multiplier = trading_days_withing_last_n / trading_days_withing_tv_sort_window * 1.1
    sys.stderr.write(
        f"Selected TV_SORT_WINDOW={tv_sort_window}: {reason}.\n"
        f"  {market.timezone}: lookback {lookback_start.isoformat()} to {now.isoformat()}; "
        f"selected window starts {tv_sort_start.isoformat()}.\n"
        f"  Completed trading days: lookback={trading_days_withing_last_n}, "
        f"selected window={trading_days_withing_tv_sort_window}; "
        f"candidate pool multiplier={multiplier:.6g}.\n"
    )
    return ScreeningWindow(
        now=now,
        lookback_start=lookback_start,
        tv_sort_window=tv_sort_window,
        tv_sort_start=tv_sort_start,
        trading_days_withing_last_n=trading_days_withing_last_n,
        trading_days_withing_tv_sort_window=trading_days_withing_tv_sort_window,
        candidate_pool_multiplier=multiplier,
        request_start=lookback_start.date().isoformat(),
        request_end=max(closes).date().isoformat(),
    )


def load_screening_window(
    market: MarketConfig, last_n: int, api_key: str | None
) -> ScreeningWindow:
    validate_last_n(last_n)
    if not api_key:
        raise ScreeningWindowError(
            "FMP_API_KEY is required for US/Taiwan reference history."
        )
    now = dt.datetime.now(ZoneInfo(market.timezone))
    # Keep the same local clock time, even across daylight saving changes.
    lookback_start = now - dt.timedelta(days=last_n)
    symbol = "NVDA" if market.code == "u" else "2330.TW"
    url = FMP_STABLE_URL if market.code == "u" else f"{FMP_LEGACY_URL}/{symbol}"
    params = [
        ("apikey", api_key),
        ("from", lookback_start.date().isoformat()),
        ("to", now.date().isoformat()),
        *([("symbol", symbol)] if market.code == "u" else []),
    ]
    try:
        # Reference dates must be fresh, even when candidate history is cached.
        data = get_json(url, params)
        rows = data if market.code == "u" else data["historical"]
        dates = [dt.date.fromisoformat(row["date"]) for row in rows]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise ScreeningWindowError(
            f"Could not load FMP reference history for {symbol}."
        ) from exc
    return select_screening_window(market, now, lookback_start, dates)


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


def tradingview_payload(market: MarketConfig, limit: int, sort_window: str) -> dict:
    return {
        "columns": ["ticker-view"],
        "filter": [{"left": "is_primary", "operation": "equal", "right": True}],
        "ignore_unknown_fields": False,
        "options": {"lang": market.language},
        "range": [0, limit],
        "sort": {"sortBy": f"Value.Traded|{sort_window}", "sortOrder": "desc"},
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


@dataclass(frozen=True)
class Candidate:
    symbol: str
    description: str
    market: str = "u"
    trading_dollars: float | None = None
    revenue_growth: float | None = None
    eps_growth: float | None = None
    ps_score: float | None = None
    pe_score: float | None = None
    growth_rank: float | None = None
    valuation_rank: float | None = None


def adjusted_growth_score(symbol: str, score: float) -> float:
    if symbol == "BTC":
        return score * BTC_GROWTH_MULTIPLIER
    return score


def fetch_score(
    path: str,
    params: Mapping[str, str | int | float],
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


def score_candidate(
    candidate: Candidate,
    score_name: str,
    score_fields: tuple[str, str],
    get_score: Callable[[], Any],
) -> Candidate | None:
    try:
        scores = get_score()
        values = dict(zip(score_fields, scores, strict=True))
        if values[score_fields[0]] is None:
            return None
        return replace(candidate, **values)
    except Exception as exc:
        report_scoring_failure(candidate, score_name, exc)
        return None


def score_growth(candidate: Candidate) -> Candidate | None:
    scored = score_candidate(
        candidate,
        "growth score",
        ("revenue_growth", "eps_growth"),
        lambda: fetch_score(
            "growths",
            {"market": candidate.market, "symbol": candidate.symbol},
        ),
    )
    if scored is None:
        return None
    return replace(
        scored,
        **{
            field: adjusted_growth_score(candidate.symbol, value)
            for field in ("revenue_growth", "eps_growth")
            if (value := getattr(scored, field)) is not None
        },
    )


def score_valuation(candidate: Candidate) -> Candidate | None:
    return score_candidate(
        candidate,
        "valuation score",
        ("ps_score", "pe_score"),
        lambda: fetch_score(
            "scores",
            {
                "market": candidate.market,
                "symbol": candidate.symbol,
                "q": Q,
            },
        ),
    )


def fetch_symbols_from_tv(
    market: MarketConfig,
    top_n_symbols: int,
    sort_window: str,
) -> list[tuple[str, str]]:
    payload = tradingview_payload(market, top_n_symbols, sort_window)
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
    market: MarketConfig,
    window: ScreeningWindow | None = None,
) -> float | None:
    if window is not None:
        recent_rows = [
            row
            for row in rows
            if window.lookback_start
            <= trading_close_time(market, dt.date.fromisoformat(row[0]))
            <= window.now
        ]
    else:
        if market.code != "j":
            raise ScreeningWindowError(
                "US/Taiwan liquidity requires a screening window."
            )
        today = today_for_market(market)
        recent_rows = [
            row
            for row in rows
            if 0 <= (today - dt.date.fromisoformat(row[0])).days < LAST_N
        ]
    if not recent_rows:
        sys.stderr.write(
            f"Skipping {symbol} trading dollar data: no rows in {LAST_N} days\n"
        )
        return None
    return sum(notional for _, notional in recent_rows)


def fetch_trading_dollar_fmp(
    market: MarketConfig,
    symbol: str,
    description: str,
    api_key: str | None,
    window: ScreeningWindow | None = None,
) -> Candidate:
    from_date = (
        window.request_start
        if window is not None
        else (
            today_for_market(market) - dt.timedelta(days=LOOKBACK_DAYS - 1)
        ).isoformat()
    )
    fmp_symbol = f"{symbol}{market.fmp_symbol_suffix}"
    data = cached_get_json(
        f"{FMP_LEGACY_URL}/{fmp_symbol}" if market.fmp_legacy else FMP_STABLE_URL,
        params=[
            ("apikey", api_key),
            ("from", from_date),
            *([("to", window.request_end)] if window is not None else []),
            *([] if market.fmp_legacy else [("symbol", fmp_symbol)]),
        ],
    )
    rows = data.get("historical", []) if market.fmp_legacy else data
    daily_notionals = [(row["date"], row["vwap"] * row["volume"]) for row in rows]
    return Candidate(
        symbol=symbol,
        description=description,
        market=market.code,
        trading_dollars=recent_trading_dollars(daily_notionals, symbol, market, window),
    )


def top_candidates(
    candidates: Iterable[Candidate],
    limit: int,
) -> list[Candidate]:
    return sorted(
        (
            candidate
            for candidate in candidates
            if candidate.trading_dollars is not None
        ),
        key=lambda candidate: candidate.trading_dollars or 0,
        reverse=True,
    )[:limit]


def fetch_trading_dollars_fmp(
    market: MarketConfig,
    stocks: Iterable[tuple[str, str]],
    api_key: str | None,
    window: ScreeningWindow | None = None,
) -> list[Candidate]:
    sys.stderr.write("Fetching trading dollar data from FMP...\n")
    return [
        fetch_trading_dollar_fmp(market, symbol, description, api_key, window)
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
    market: MarketConfig,
    stock_id: str,
    description: str,
    window: ScreeningWindow,
) -> Candidate:
    df = fetch_taiwan_stock_daily(stock_id, window.request_start, window.request_end)
    columns = ["date", "Trading_money"]
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        sys.stderr.write(
            f"Skipping {stock_id} trading dollar data: "
            f"FinMind response missing columns: {', '.join(missing_columns)}\n"
        )
        return Candidate(symbol=stock_id, description=description, market=market.code)

    daily_notionals = [
        (str(row.date), float(row.Trading_money))
        for row in df[columns].itertuples(index=False)
    ]
    return Candidate(
        symbol=stock_id,
        description=description,
        market=market.code,
        trading_dollars=recent_trading_dollars(
            daily_notionals, stock_id, market, window
        ),
    )


def fetch_trading_dollars_tw(
    market: MarketConfig,
    stocks: Iterable[tuple[str, str]],
    window: ScreeningWindow,
) -> list[Candidate]:
    global finmind_api

    finmind_key = os.environ.get("FINMIND_KEY")
    if not finmind_key:
        raise ValueError("FINMIND_KEY is required for Taiwanese liquidity data.")

    from FinMind.data import DataLoader

    finmind_api = DataLoader()
    finmind_api.login_by_token(finmind_key)

    sys.stderr.write("Fetching trading dollar data from FinMind...\n")
    with ThreadPoolExecutor(max_workers=FINMIND_THREADS) as pool:
        return list(
            pool.map(
                lambda s: fetch_trading_dollar_tw(market, s[0], s[1], window),
                stocks,
            )
        )


def load_top_candidates(
    market: MarketConfig,
    api_key: str | None,
    top_n_symbols: int,
    top_n_results: int,
    window: ScreeningWindow | None = None,
) -> list[Candidate]:
    validate_candidate_pool_sizes(top_n_symbols, top_n_results)
    if market.code != "j" and window is None:
        raise ScreeningWindowError("US/Taiwan screening requires a screening window.")

    sys.stderr.write("Fetching symbols from TradingView...\n")
    stocks = fetch_symbols_from_tv(
        market=market,
        top_n_symbols=top_n_symbols,
        sort_window=window.tv_sort_window
        if window is not None
        else getattr(config, "TV_SORT_WINDOW"),
    )
    sys.stderr.write(f"Found {len(stocks)} symbols\n")

    if market.liquidity_source == "finmind":
        assert window is not None
        liquidity = fetch_trading_dollars_tw(market, stocks, window)
    else:
        liquidity = fetch_trading_dollars_fmp(market, stocks, api_key, window)

    return top_candidates(liquidity, top_n_results)


def score_growth_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    with ThreadPoolExecutor(max_workers=GROWTH_CONCURRENCY) as executor:
        scored = [
            scored_candidate
            for scored_candidate in executor.map(score_growth, candidates)
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
    api_key = os.environ.get("FMP_API_KEY")
    window = (
        load_screening_window(market, LAST_N, api_key) if market.code != "j" else None
    )
    multiplier = (
        window.candidate_pool_multiplier
        if window is not None
        else getattr(config, "CANDIDATE_POOL_MULTIPLIER")
    )
    pool_size = candidate_pool_size(RESULT_LIMIT, multiplier)
    candidates = load_top_candidates(
        market=market,
        api_key=api_key,
        top_n_symbols=pool_size,
        top_n_results=RESULT_LIMIT,
        window=window,
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


def metric_ranks(
    candidates: Iterable[Candidate], score_field: str, *, reverse: bool = True
) -> dict[Candidate, float]:
    ordered = sorted(
        (
            candidate
            for candidate in candidates
            if getattr(candidate, score_field) is not None
        ),
        key=lambda candidate: getattr(candidate, score_field),
        reverse=reverse,
    )
    ranks = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        score = getattr(ordered[start], score_field)
        while end < len(ordered) and getattr(ordered[end], score_field) == score:
            end += 1

        rank = (start + 1 + end) / 2
        for candidate in ordered[start:end]:
            ranks[candidate] = rank
        start = end
    return ranks


def combine_metric_ranks(
    candidates: Iterable[Candidate], fields: tuple[str, str], rank_field: str
) -> list[Candidate]:
    candidates = [
        candidate
        for candidate in candidates
        if getattr(candidate, fields[0]) is not None
    ]
    ranks = [metric_ranks(candidates, field) for field in fields]
    combined = []
    for candidate in candidates:
        available = [rank[candidate] for rank in ranks if candidate in rank]
        if available:
            combined.append(
                replace(candidate, **{rank_field: sum(available) / len(available)})
            )
    return combined


def select_growth_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    ranked = combine_metric_ranks(
        candidates, ("revenue_growth", "eps_growth"), "growth_rank"
    )
    ranked.sort(key=lambda candidate: (candidate.growth_rank, candidate.symbol))
    return ranked[: (len(ranked) + 1) // 2]


def rank_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    eligible = combine_metric_ranks(
        (candidate for candidate in candidates if candidate.growth_rank is not None),
        ("ps_score", "pe_score"),
        "valuation_rank",
    )
    eligible.sort(key=lambda candidate: (candidate.valuation_rank, candidate.symbol))
    eligible = eligible[: (len(eligible) + 1) // 2]
    growth_ranks = metric_ranks(eligible, "growth_rank", reverse=False)
    valuation_ranks = metric_ranks(eligible, "valuation_rank", reverse=False)
    return sorted(
        eligible,
        key=lambda candidate: (
            growth_ranks[candidate] + valuation_ranks[candidate],
            candidate.symbol,
        ),
    )


def main() -> int:
    load_dotenv()

    try:
        candidates = load_candidates()
    except ScreeningWindowError as exc:
        sys.stderr.write(f"Screening stopped: {exc}\n")
        return 1
    growth_candidates = score_growth_candidates(candidates)
    selected_candidates = select_growth_candidates(growth_candidates)
    scored_candidates = score_all_valuations(selected_candidates)
    ranked_candidates = rank_candidates(scored_candidates)

    write_results(ranked_candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

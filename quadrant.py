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
TV_US_URL = "https://scanner.tradingview.com/america/scan?label-product=screener-stock"
TV_TW_URL = "https://scanner.tradingview.com/taiwan/scan?label-product=screener-stock"
cache = Cache(Path().resolve() / ".cache")


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


def get_json(url: str, params: Mapping[str, str | int | None]) -> Any:
    response = httpx.get(url, params=params, timeout=None)
    response.raise_for_status()
    return response.json()


@cached(43200)
def cached_get_json(url: str, params: list[tuple[str, str | int | None]]) -> Any:
    return get_json(url, dict(params))


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


@dataclass(frozen=True)
class MarketConfig:
    code: str
    timezone: str
    tradingview_url: str
    tradingview_headers: dict[str, str]
    tradingview_market: str
    language: str
    liquidity_source: str
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
}


def current_market_config() -> MarketConfig:
    return MARKET_CONFIG_BY_CODE.get(MARKET, MARKET_CONFIG_BY_CODE["u"])


class ScreeningWindowError(ValueError):
    pass


@dataclass(frozen=True)
class ScreeningWindow:
    now: dt.datetime
    lookback_start: dt.datetime
    tv_sort_window: str
    tv_sort_start: dt.datetime
    lookback_sessions: int
    window_sessions: int
    request_end: str

    @property
    def candidate_pool_multiplier(self) -> float:
        return self.lookback_sessions / self.window_sessions * 1.1

    @property
    def request_start(self) -> str:
        return self.lookback_start.date().isoformat()


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
    if not closes:
        raise ScreeningWindowError("No completed trading days within LAST_N.")

    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    starts = {
        "1W": midnight - dt.timedelta(days=now.weekday()),
        "1M": midnight.replace(day=1),
    }
    counts = {
        name: sum(close >= start for close in closes)
        for name, start in starts.items()
        if start >= lookback_start
    }
    # LAST_N >= 7 guarantees 1W is eligible; insertion order makes it win ties.
    tv_sort_window = max(counts, key=counts.__getitem__)
    window_sessions = counts[tv_sort_window]
    if not window_sessions:
        raise ScreeningWindowError(
            "No completed trading days within the eligible TradingView windows."
        )
    return ScreeningWindow(
        now=now,
        lookback_start=lookback_start,
        tv_sort_window=tv_sort_window,
        tv_sort_start=starts[tv_sort_window],
        lookback_sessions=len(closes),
        window_sessions=window_sessions,
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
    params = {
        "apikey": api_key,
        "from": lookback_start.date().isoformat(),
        "to": now.date().isoformat(),
    }
    if market.code == "u":
        params["symbol"] = symbol
    try:
        # Reference dates must be fresh, even when candidate history is cached.
        data = get_json(url, params)
        rows = data if market.code == "u" else data["historical"]
        dates = [dt.date.fromisoformat(row["date"]) for row in rows]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise ScreeningWindowError(
            f"Could not load FMP reference history for {symbol}."
        ) from exc
    window = select_screening_window(market, now, lookback_start, dates)
    sys.stderr.write(
        f"Selected TV_SORT_WINDOW={window.tv_sort_window}.\n"
        f"  {market.timezone}: lookback {lookback_start.isoformat()} to {now.isoformat()}; "
        f"selected window starts {window.tv_sort_start.isoformat()}.\n"
        f"  Completed trading days: lookback={window.lookback_sessions}, "
        f"selected window={window.window_sessions}; "
        f"candidate pool multiplier={window.candidate_pool_multiplier:.6g}.\n"
    )
    return window


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
    window: ScreeningWindow,
) -> float | None:
    recent_rows = [
        row
        for row in rows
        if window.lookback_start
        <= trading_close_time(market, dt.date.fromisoformat(row[0]))
        <= window.now
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
    window: ScreeningWindow,
) -> Candidate:
    data = cached_get_json(
        FMP_STABLE_URL,
        params=[
            ("apikey", api_key),
            ("from", window.request_start),
            ("to", window.request_end),
            ("symbol", symbol),
        ],
    )
    daily_notionals = [(row["date"], row["vwap"] * row["volume"]) for row in data]
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
    window: ScreeningWindow,
) -> list[Candidate]:
    sys.stderr.write("Fetching trading dollar data from FMP...\n")
    return [
        fetch_trading_dollar_fmp(market, symbol, description, api_key, window)
        for symbol, description in stocks
    ]


def fetch_taiwan_stock_daily(stock_id: str, start: str, end: str, *, client: Any):
    key = (__name__, "fetch_taiwan_stock_daily", (stock_id, start, end), ())
    if key in cache:
        return cache[key]
    df = client.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=start,
        end_date=end,
    )
    cache.set(key, df, 43200)
    return df


def fetch_trading_dollar_tw(
    market: MarketConfig,
    stock_id: str,
    description: str,
    window: ScreeningWindow,
    *,
    client: Any,
) -> Candidate:
    df = fetch_taiwan_stock_daily(
        stock_id, window.request_start, window.request_end, client=client
    )
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
    finmind_key = os.environ.get("FINMIND_KEY")
    if not finmind_key:
        raise ValueError("FINMIND_KEY is required for Taiwanese liquidity data.")

    from FinMind.data import DataLoader

    client = DataLoader()
    client.login_by_token(finmind_key)

    sys.stderr.write("Fetching trading dollar data from FinMind...\n")
    with ThreadPoolExecutor(max_workers=FINMIND_THREADS) as pool:
        return list(
            pool.map(
                lambda s: fetch_trading_dollar_tw(
                    market, s[0], s[1], window, client=client
                ),
                stocks,
            )
        )


def load_top_candidates(
    market: MarketConfig,
    api_key: str | None,
    top_n_symbols: int,
    top_n_results: int,
    window: ScreeningWindow,
) -> list[Candidate]:
    validate_candidate_pool_sizes(top_n_symbols, top_n_results)

    sys.stderr.write("Fetching symbols from TradingView...\n")
    stocks = fetch_symbols_from_tv(
        market=market,
        top_n_symbols=top_n_symbols,
        sort_window=window.tv_sort_window,
    )
    sys.stderr.write(f"Found {len(stocks)} symbols\n")

    if market.liquidity_source == "finmind":
        liquidity = fetch_trading_dollars_tw(market, stocks, window)
    else:
        liquidity = fetch_trading_dollars_fmp(market, stocks, api_key, window)

    return top_candidates(liquidity, top_n_results)


def score_candidates(
    candidates: Iterable[Candidate],
    scorer: Callable[[Candidate], Candidate | None],
    *,
    max_workers: int | None = None,
) -> list[Candidate]:
    if max_workers is None:
        scored = map(scorer, candidates)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            scored = list(executor.map(scorer, candidates))
    return [candidate for candidate in scored if candidate is not None]


def load_candidates() -> list[Candidate]:
    market = current_market_config()
    api_key = os.environ.get("FMP_API_KEY")
    window = load_screening_window(market, LAST_N, api_key)
    pool_size = candidate_pool_size(RESULT_LIMIT, window.candidate_pool_multiplier)
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
    growth_candidates = score_candidates(
        candidates, score_growth, max_workers=GROWTH_CONCURRENCY
    )
    selected_candidates = select_growth_candidates(growth_candidates)
    scored_candidates = score_candidates(selected_candidates, score_valuation)
    ranked_candidates = rank_candidates(scored_candidates)

    write_results(ranked_candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

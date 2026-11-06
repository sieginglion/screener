import io
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, Mock, call, patch
from zoneinfo import ZoneInfo

import clear_ticker_cache
import quadrant


class FakeTaiwanDailyData:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, columns):
        if columns != ["date", "Trading_money"]:
            raise KeyError(columns)
        return self

    def itertuples(self, index=False):
        if index:
            raise ValueError("index must be disabled")
        return iter(
            [
                types.SimpleNamespace(date=date, Trading_money=trading_money)
                for date, trading_money in self.rows
            ]
        )


class CachedGetJsonTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.cache = quadrant.Cache(self.temporary_directory.name)
        self.cache_patch = patch.object(quadrant, "cache", self.cache)
        self.cache_patch.start()

    def tearDown(self):
        self.cache_patch.stop()
        self.cache.close()
        self.temporary_directory.cleanup()

    def test_returns_json_and_reuses_a_cached_response(self):
        url = "https://example.com/prices"
        params = [("symbol", "ABC"), ("apikey", "test-key")]
        payload = [{"date": "2026-07-17", "vwap": 10, "volume": 2}]
        response = Mock()
        response.json.return_value = payload

        with patch.object(quadrant.httpx, "get", return_value=response) as get:
            self.assertEqual(quadrant.cached_get_json(url, params), payload)
            self.assertEqual(quadrant.cached_get_json(url, params), payload)

        get.assert_called_once_with(url, params=dict(params), timeout=None)
        response.raise_for_status.assert_called_once_with()
        response.json.assert_called_once_with()

    def test_does_not_cache_a_failed_response(self):
        url = "https://example.com/prices"
        params = [("symbol", "ABC"), ("apikey", "test-key")]
        request = quadrant.httpx.Request("GET", url)
        response = quadrant.httpx.Response(500, request=request)

        with patch.object(quadrant.httpx, "get", return_value=response) as get:
            for _ in range(2):
                with self.assertRaises(quadrant.httpx.HTTPStatusError):
                    quadrant.cached_get_json(url, params)

        self.assertEqual(get.call_count, 2)


class FetchScoreTests(unittest.TestCase):
    def test_returns_the_raw_json_response(self):
        params = {"market": "u", "symbol": "ABC"}
        payload = [1, None, 3]
        response = Mock()
        response.json.return_value = payload

        with patch.object(quadrant.httpx, "get", return_value=response) as get:
            self.assertEqual(quadrant.fetch_score("scores", params), payload)

        get.assert_called_once_with(
            f"{quadrant.SCORING_BASE_URL}/scores",
            params=params,
            timeout=None,
        )
        response.raise_for_status.assert_called_once_with()
        response.json.assert_called_once_with()

    def test_propagates_http_errors(self):
        url = f"{quadrant.SCORING_BASE_URL}/growth"
        response = quadrant.httpx.Response(
            500, request=quadrant.httpx.Request("GET", url)
        )

        with patch.object(quadrant.httpx, "get", return_value=response):
            with self.assertRaises(quadrant.httpx.HTTPStatusError):
                quadrant.fetch_score("growth", {"market": "u", "symbol": "ABC"})


class CacheClearingTests(unittest.TestCase):
    def test_liquidity_cache_matcher_recognizes_cached_json_entries(self):
        key = (
            quadrant.__name__,
            "cached_get_json",
            (quadrant.FMP_STABLE_URL,),
            (),
        )

        self.assertTrue(clear_ticker_cache.key_matches_liquidity(key))


class MarketConfigurationTests(unittest.TestCase):
    def test_current_market_config_selects_each_supported_market(self):
        for code, expected in quadrant.MARKET_CONFIG_BY_CODE.items():
            with self.subTest(market=code), patch.object(quadrant, "MARKET", code):
                self.assertIs(quadrant.current_market_config(), expected)

    def test_current_market_config_falls_back_to_us(self):
        with patch.object(quadrant, "MARKET", "unsupported"):
            self.assertIs(
                quadrant.current_market_config(),
                quadrant.MARKET_CONFIG_BY_CODE["u"],
            )

    def test_today_for_market_uses_the_configured_timezone(self):
        expected_today = quadrant.dt.date(2026, 7, 17)

        for market in quadrant.MARKET_CONFIG_BY_CODE.values():
            with (
                self.subTest(market=market.code),
                patch.object(quadrant, "current_market_config") as current_config,
                patch.object(quadrant.dt, "datetime") as datetime,
            ):
                datetime.now.return_value.date.return_value = expected_today

                self.assertEqual(quadrant.today_for_market(market), expected_today)
                datetime.now.assert_called_once_with(ZoneInfo(market.timezone))
                current_config.assert_not_called()

    def test_candidate_records_its_market(self):
        for code, market in quadrant.MARKET_CONFIG_BY_CODE.items():
            with self.subTest(market=code):
                candidate = quadrant.Candidate(
                    symbol="ABC", description="Example", market=market.code
                )
                self.assertEqual(candidate.market, market.code)

    def test_market_configs_capture_provider_specific_behavior(self):
        us = quadrant.MARKET_CONFIG_BY_CODE["u"]
        taiwan = quadrant.MARKET_CONFIG_BY_CODE["t"]
        japan = quadrant.MARKET_CONFIG_BY_CODE["j"]

        self.assertEqual(us.liquidity_source, "fmp")
        self.assertTrue(us.exclude_intraday_fmp_row)
        self.assertTrue(us.include_bitcoin)

        self.assertEqual(taiwan.liquidity_source, "finmind")
        self.assertFalse(taiwan.exclude_intraday_fmp_row)
        self.assertFalse(taiwan.include_bitcoin)

        self.assertEqual(japan.liquidity_source, "fmp")
        self.assertTrue(japan.fmp_legacy)
        self.assertEqual(japan.fmp_symbol_suffix, ".T")


class TradingViewHeaderTests(unittest.TestCase):
    common_headers = {
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
    market_specific_headers = {
        "u": {
            "origin": "https://www.tradingview.com",
            "referer": "https://www.tradingview.com/",
            "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        },
        "t": {
            "origin": "https://tw.tradingview.com",
            "referer": "https://tw.tradingview.com/",
            "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
        },
        "j": {
            "origin": "https://jp.tradingview.com",
            "referer": "https://jp.tradingview.com/",
            "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
        },
    }

    def test_effective_headers_match_the_market_contract(self):
        for market, market_headers in self.market_specific_headers.items():
            with self.subTest(market=market):
                self.assertEqual(
                    quadrant.MARKET_CONFIG_BY_CODE[market].tradingview_headers,
                    self.common_headers | market_headers,
                )

    def test_symbol_fetch_forwards_each_market_header_set(self):
        response = Mock()
        response.json.return_value = {
            "data": [{"d": [{"name": "BRK.B", "description": "Berkshire"}]}]
        }
        client = MagicMock()
        client.__enter__.return_value = client
        client.post.return_value = response

        with patch("quadrant.httpx.Client", return_value=client):
            for market in quadrant.MARKET_CONFIG_BY_CODE.values():
                self.assertEqual(
                    quadrant.fetch_symbols_from_tv(market, top_n_symbols=1),
                    [("BRK-B", "Berkshire")],
                )

        self.assertEqual(len(client.post.call_args_list), 3)
        for market, request in zip(
            quadrant.MARKET_CONFIG_BY_CODE.values(), client.post.call_args_list
        ):
            with self.subTest(market=market.tradingview_market):
                self.assertEqual(request.args[0], market.tradingview_url)
                self.assertEqual(request.kwargs["headers"], market.tradingview_headers)
                self.assertEqual(
                    request.kwargs["json"],
                    quadrant.tradingview_payload(market, limit=1),
                )


class LoadTopCandidatesTests(unittest.TestCase):
    market = quadrant.MARKET_CONFIG_BY_CODE["u"]
    stocks = [("A", "Alpha; Inc."), ("B", "Beta")]

    def test_loader_rejects_a_pool_that_cannot_be_reranked(self):
        with patch("quadrant.fetch_symbols_from_tv") as symbols:
            with self.assertRaisesRegex(ValueError, "top_n_symbols must be greater"):
                quadrant.load_top_candidates(
                    market=self.market,
                    api_key="test-key",
                    top_n_symbols=2,
                    top_n_results=2,
                )

        symbols.assert_not_called()

    def test_top_candidates_excludes_insufficient_liquidity_data(self):
        result = quadrant.top_candidates(
            [
                quadrant.Candidate("A", "Alpha", trading_dollars=None),
                quadrant.Candidate("B", "Beta", trading_dollars=2),
                quadrant.Candidate("C", "Charlie", trading_dollars=1),
            ],
            limit=3,
        )

        self.assertEqual(
            result,
            [
                quadrant.Candidate(symbol="B", description="Beta", trading_dollars=2),
                quadrant.Candidate(
                    symbol="C", description="Charlie", trading_dollars=1
                ),
            ],
        )

    def test_fmp_loader_fetches_symbols_and_ranks_by_liquidity(self):
        today_date = quadrant.dt.date(2026, 7, 17)
        from_date = (
            today_date - quadrant.dt.timedelta(days=quadrant.LOOKBACK_DAYS - 1)
        ).isoformat()

        def trading_dollars(market, symbol, description, from_date, api_key):
            return quadrant.Candidate(
                symbol=symbol,
                description=description,
                market=market.code,
                trading_dollars={"A": 1, "B": 2}[symbol],
            )

        with (
            patch(
                "quadrant.today_for_market",
                return_value=today_date,
            ) as today,
            patch(
                "quadrant.fetch_symbols_from_tv", return_value=self.stocks
            ) as symbols,
            patch(
                "quadrant.fetch_trading_dollar_fmp",
                side_effect=trading_dollars,
            ) as fetch,
        ):
            result = quadrant.load_top_candidates(
                market=self.market,
                api_key="test-key",
                top_n_symbols=3,
                top_n_results=2,
            )

        self.assertEqual(
            result,
            [
                quadrant.Candidate(symbol="B", description="Beta", trading_dollars=2),
                quadrant.Candidate(
                    symbol="A", description="Alpha; Inc.", trading_dollars=1
                ),
            ],
        )
        symbols.assert_called_once_with(market=self.market, top_n_symbols=3)
        today.assert_called_once_with(self.market)
        self.assertEqual(
            fetch.call_args_list,
            [
                call(self.market, "A", "Alpha; Inc.", from_date, "test-key"),
                call(self.market, "B", "Beta", from_date, "test-key"),
            ],
        )

    def test_loader_omits_candidates_without_enough_liquidity_data(self):
        def trading_dollars(market, symbol, description, from_date, api_key):
            return quadrant.Candidate(
                symbol=symbol,
                description=description,
                market=market.code,
                trading_dollars={"A": None, "B": 2}[symbol],
            )

        with (
            patch(
                "quadrant.today_for_market", return_value=quadrant.dt.date(2026, 7, 17)
            ),
            patch("quadrant.fetch_symbols_from_tv", return_value=self.stocks),
            patch(
                "quadrant.fetch_trading_dollar_fmp",
                side_effect=trading_dollars,
            ),
        ):
            result = quadrant.load_top_candidates(
                market=self.market,
                api_key="test-key",
                top_n_symbols=3,
                top_n_results=2,
            )

        self.assertEqual(
            result,
            [
                quadrant.Candidate(
                    symbol="B", description="Beta", market="u", trading_dollars=2
                )
            ],
        )

    def test_taiwan_loader_logs_in_and_ranks_by_liquidity(self):
        market = quadrant.MARKET_CONFIG_BY_CODE["t"]
        today_date = quadrant.dt.date(2026, 7, 17)
        start = (
            today_date - quadrant.dt.timedelta(days=quadrant.LOOKBACK_DAYS - 1)
        ).isoformat()
        finmind = types.ModuleType("FinMind")
        finmind_data = types.ModuleType("FinMind.data")
        loader = Mock()
        data_loader = Mock(return_value=loader)
        setattr(finmind, "data", finmind_data)
        setattr(finmind_data, "DataLoader", data_loader)

        def trading_dollars(market, symbol, description, start, end):
            return quadrant.Candidate(
                symbol=symbol,
                description=description,
                market=market.code,
                trading_dollars={"A": 1, "B": 2}[symbol],
            )

        with (
            patch.dict(
                sys.modules,
                {"FinMind": finmind, "FinMind.data": finmind_data},
            ),
            patch.dict(quadrant.os.environ, {"FINMIND_KEY": "test-key"}),
            patch.object(quadrant, "finmind_api", None),
            patch(
                "quadrant.today_for_market",
                return_value=today_date,
            ) as today,
            patch(
                "quadrant.fetch_symbols_from_tv", return_value=self.stocks
            ) as symbols,
            patch("quadrant.fetch_trading_dollars_fmp") as fmp,
            patch(
                "quadrant.fetch_trading_dollar_tw",
                side_effect=trading_dollars,
            ) as fetch,
        ):
            result = quadrant.load_top_candidates(
                market=market,
                api_key=None,
                top_n_symbols=3,
                top_n_results=2,
            )

        self.assertEqual(
            result,
            [
                quadrant.Candidate(
                    symbol="B", description="Beta", market="t", trading_dollars=2
                ),
                quadrant.Candidate(
                    symbol="A",
                    description="Alpha; Inc.",
                    market="t",
                    trading_dollars=1,
                ),
            ],
        )
        data_loader.assert_called_once_with()
        loader.login_by_token.assert_called_once_with("test-key")
        symbols.assert_called_once_with(
            market=market,
            top_n_symbols=3,
        )
        fmp.assert_not_called()
        today.assert_called_once_with(market)
        self.assertEqual(
            fetch.call_args_list,
            [
                call(market, "A", "Alpha; Inc.", start, "2026-07-17"),
                call(market, "B", "Beta", start, "2026-07-17"),
            ],
        )


class CandidateLoadingTests(unittest.TestCase):
    def test_bitcoin_is_added_only_for_markets_that_enable_it(self):
        for code, market in quadrant.MARKET_CONFIG_BY_CODE.items():
            source_candidates = [
                quadrant.Candidate(symbol="ABC", description="Example")
            ]
            with (
                self.subTest(market=code),
                patch("quadrant.current_market_config", return_value=market),
                patch(
                    "quadrant.load_top_candidates",
                    return_value=list(source_candidates),
                ) as load_top_candidates,
            ):
                result = quadrant.load_candidates()

            expected = source_candidates + (
                [quadrant.Candidate(symbol="BTC", description="Bitcoin", market="c")]
                if market.include_bitcoin
                else []
            )
            self.assertEqual(result, expected)
            self.assertIs(load_top_candidates.call_args.kwargs["market"], market)


class GrowthScoringTests(unittest.TestCase):
    def test_all_nullable_signed_metric_pairs_are_accepted_and_ranked(self):
        cases = (
            (quadrant.score_growth, ("revenue_growth", "eps_growth"), "growth_rank"),
            (quadrant.score_valuation, ("ps_score", "pe_score"), "valuation_rank"),
        )
        for scorer, fields, rank_field in cases:
            references = [
                replace(
                    quadrant.Candidate(symbol, symbol), **dict.fromkeys(fields, value)
                )
                for symbol, value in (("low", -2), ("zero", 0), ("high", 2))
            ]
            # With reference values -2, 0, 2: -1 ranks third, 0 ties at
            # ranks 2 and 3, and +1 ranks second. Higher values rank better.
            expected_ranks = {-1: 3, 0: 2.5, 1: 2}
            for first in (None, -1, 0, 1):
                for second in (None, -1, 0, 1):
                    with (
                        self.subTest(scorer=scorer.__name__, pair=(first, second)),
                        patch("quadrant.fetch_score", return_value=[first, second]),
                    ):
                        scored = scorer(quadrant.Candidate("target", "Target"))
                        self.assertEqual(
                            scored,
                            replace(
                                quadrant.Candidate("target", "Target"),
                                **dict(zip(fields, (first, second))),
                            ),
                        )
                        assert scored is not None
                        ranked = quadrant.combine_metric_ranks(
                            [*references, scored], fields, rank_field
                        )
                        target = [item for item in ranked if item.symbol == "target"]
                        if first is None and second is None:
                            self.assertEqual(target, [])
                        else:
                            ranks = [
                                expected_ranks[value]
                                for value in (first, second)
                                if value is not None
                            ]
                            self.assertEqual(len(target), 1)
                            self.assertEqual(
                                getattr(target[0], rank_field), sum(ranks) / len(ranks)
                            )

    def test_scores_are_kept_on_the_same_candidate(self):
        candidate = quadrant.Candidate("A", "Alpha", trading_dollars=123)
        with patch("quadrant.fetch_score", side_effect=[[2, -1], [0, None]]):
            grown = quadrant.score_growth(candidate)
            assert grown is not None
            scored = quadrant.score_valuation(grown)
        self.assertEqual(
            scored,
            quadrant.Candidate(
                "A",
                "Alpha",
                trading_dollars=123,
                revenue_growth=2,
                eps_growth=-1,
                ps_score=0,
            ),
        )

    def test_scoring_uses_the_expected_endpoints_and_parameters(self):
        candidate = quadrant.Candidate("A", "Alpha")
        with patch("quadrant.fetch_score", return_value=[None, -2]) as fetch:
            grown = quadrant.score_growth(candidate)
            scored = quadrant.score_valuation(candidate)
        self.assertEqual(
            fetch.call_args_list,
            [
                call("growths", {"market": "u", "symbol": "A"}),
                call(
                    "scores",
                    {
                        "market": "u",
                        "symbol": "A",
                        "q": quadrant.Q,
                        "u": quadrant.VALUATION_SCORE_UPPER,
                    },
                ),
            ],
        )
        assert grown is not None and scored is not None
        self.assertIsNone(grown.revenue_growth)
        self.assertEqual(grown.eps_growth, -2)
        self.assertIsNone(scored.ps_score)
        self.assertEqual(scored.pe_score, -2)

    def test_scoring_failure_skips_the_candidate_with_a_message(self):
        candidate = quadrant.Candidate("A", "Alpha")
        for scorer, label in (
            (quadrant.score_growth, "growth score"),
            (quadrant.score_valuation, "valuation score"),
        ):
            with (
                self.subTest(scorer=scorer.__name__),
                patch("quadrant.fetch_score", side_effect=RuntimeError("unavailable")),
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                self.assertIsNone(scorer(candidate))
                self.assertEqual(
                    stderr.getvalue(), f"Skipping A {label}: unavailable\n"
                )

    def test_invalid_payloads_skip_the_candidate(self):
        candidate = quadrant.Candidate("A", "Alpha")
        for scorer in (quadrant.score_growth, quadrant.score_valuation):
            for payload in (
                "invalid",
                [],
                [1],
                [1, 2, 3],
                ["bad", 1],
                [float("nan"), 1],
                [1, float("inf")],
                [True, None],
            ):
                with (
                    self.subTest(scorer=scorer.__name__, payload=payload),
                    patch("quadrant.fetch_score", return_value=payload),
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    self.assertIsNone(scorer(candidate))
                    self.assertIn("Skipping A", stderr.getvalue())

    def test_growth_applies_bitcoin_multiplier_and_preserves_null(self):
        with (
            patch.object(quadrant, "BTC_GROWTH_MULTIPLIER", 2),
            patch("quadrant.fetch_score", return_value=[-3, None]),
        ):
            bitcoin = quadrant.score_growth(
                quadrant.Candidate("BTC", "Bitcoin", market="c")
            )
            stock = quadrant.score_growth(quadrant.Candidate("A", "Alpha"))
        assert bitcoin is not None and stock is not None
        self.assertEqual(bitcoin.revenue_growth, -6)
        self.assertIsNone(bitcoin.eps_growth)
        self.assertEqual(stock.revenue_growth, -3)

    def test_growth_scoring_always_runs_and_skips_failed_requests(self):
        candidates = [quadrant.Candidate("A", "Alpha"), quadrant.Candidate("B", "Beta")]
        with patch("quadrant.score_growth", side_effect=[candidates[0], None]) as score:
            result = quadrant.score_growth_candidates(candidates)
        self.assertEqual(result, candidates[:1])
        self.assertEqual(
            score.call_args_list, [call(candidate) for candidate in candidates]
        )


class RankingTests(unittest.TestCase):
    def test_independent_growth_ranks_average_available_metrics(self):
        candidates = [
            quadrant.Candidate("A", "Alpha", revenue_growth=100, eps_growth=-2),
            quadrant.Candidate("B", "Beta", revenue_growth=2, eps_growth=0),
            quadrant.Candidate("C", "Charlie", revenue_growth=-1),
            quadrant.Candidate("D", "Delta", eps_growth=-3),
            quadrant.Candidate("E", "Empty"),
        ]
        result = quadrant.combine_metric_ranks(
            candidates, ("revenue_growth", "eps_growth"), "growth_rank"
        )
        self.assertEqual(
            {candidate.symbol: candidate.growth_rank for candidate in result},
            {"A": 1.5, "B": 1.5, "C": 3, "D": 3},
        )

    def test_metric_ranks_average_ties_and_ignore_null(self):
        candidates = [
            quadrant.Candidate("A", "Alpha", revenue_growth=0),
            quadrant.Candidate("B", "Beta", revenue_growth=0),
            quadrant.Candidate("C", "Charlie", revenue_growth=-5),
            quadrant.Candidate("D", "Missing"),
        ]
        self.assertEqual(
            quadrant.metric_ranks(candidates, "revenue_growth"),
            {candidates[0]: 1.5, candidates[1]: 1.5, candidates[2]: 3},
        )

    def test_growth_selection_rounds_up_and_keeps_at_least_one(self):
        for count in (0, 1, 2, 3, 4, 5):
            candidates = [
                quadrant.Candidate(str(i), str(i), revenue_growth=-i)
                for i in reversed(range(count))
            ]
            candidates.append(quadrant.Candidate("missing", "Missing"))
            with self.subTest(count=count):
                selected = quadrant.select_growth_candidates(candidates)
                self.assertEqual(
                    [candidate.symbol for candidate in selected],
                    [str(i) for i in range((count + 1) // 2)],
                )

    def test_growth_selection_breaks_cutoff_ties_alphabetically(self):
        candidates = [
            quadrant.Candidate(symbol, symbol, eps_growth=0)
            for symbol in ("D", "C", "B", "A")
        ]
        self.assertEqual(
            [
                candidate.symbol
                for candidate in quadrant.select_growth_candidates(candidates)
            ],
            ["A", "B"],
        )

    def test_final_ranking_reranks_groups_with_equal_weight(self):
        candidates = [
            quadrant.Candidate("Z", "Zulu", growth_rank=1, ps_score=0, pe_score=-5),
            quadrant.Candidate("B", "Beta", growth_rank=20, ps_score=-1, pe_score=5),
            quadrant.Candidate("A", "Alpha", growth_rank=30, ps_score=2, pe_score=0),
        ]
        # Growth ranks 1,2,3 and valuation ranks 3,2,1 tie; symbols decide.
        # Adding original growth values would incorrectly produce Z,B,A.
        self.assertEqual(
            [candidate.symbol for candidate in quadrant.rank_candidates(candidates)],
            ["A", "B", "Z"],
        )

    def test_final_ranking_uses_either_score_and_excludes_only_missing_groups(self):
        candidates = [
            quadrant.Candidate("A", "Alpha", growth_rank=1, ps_score=-2),
            quadrant.Candidate("B", "Beta", growth_rank=2, pe_score=0),
            quadrant.Candidate("C", "No valuation", growth_rank=3),
            quadrant.Candidate("D", "No growth", ps_score=10, pe_score=10),
        ]
        self.assertEqual(
            [candidate.symbol for candidate in quadrant.rank_candidates(candidates)],
            ["A", "B"],
        )
        self.assertEqual(quadrant.rank_candidates([]), [])

    def test_valuation_ranks_average_available_components(self):
        candidates = [
            quadrant.Candidate("A", "Alpha", ps_score=-1, pe_score=0),
            quadrant.Candidate("B", "Beta", ps_score=0, pe_score=-1),
            quadrant.Candidate("C", "Charlie", ps_score=-2),
            quadrant.Candidate("D", "Delta", pe_score=-2),
        ]
        result = quadrant.combine_metric_ranks(
            candidates, ("ps_score", "pe_score"), "valuation_rank"
        )
        self.assertEqual(
            {candidate.symbol: candidate.valuation_rank for candidate in result},
            {"A": 1.5, "B": 1.5, "C": 3, "D": 3},
        )


class MainTests(unittest.TestCase):
    def test_growth_failures_and_missing_growths_are_removed_before_halving(self):
        for eligible_count in (4, 5):
            symbols = "ABCDE"[:eligible_count]
            candidates = [
                quadrant.Candidate(symbol, symbol) for symbol in symbols + "FMN"
            ]
            growths = {symbol: [-i - 1, None] for i, symbol in enumerate(symbols)}
            growths.update({"M": [1], "N": [None, None]})

            def fetch(path, params):
                if path == "growths":
                    if params["symbol"] == "F":
                        raise RuntimeError("growth unavailable")
                    return growths[params["symbol"]]
                return [0, None]

            with (
                self.subTest(eligible_count=eligible_count),
                patch("quadrant.load_dotenv"),
                patch("quadrant.load_candidates", return_value=candidates),
                patch("quadrant.fetch_score", side_effect=fetch) as fetch_score,
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                self.assertEqual(quadrant.main(), 0)
                expected = list("AB" if eligible_count == 4 else "ABC")
                self.assertCountEqual(
                    [
                        request.args[1]["symbol"]
                        for request in fetch_score.call_args_list
                        if request.args[0] == "growths"
                    ],
                    list(symbols + "FMN"),
                )
                self.assertEqual(
                    [
                        request.args[1]["symbol"]
                        for request in fetch_score.call_args_list
                        if request.args[0] == "scores"
                    ],
                    expected,
                )
                self.assertEqual(
                    stdout.getvalue(),
                    "".join(f"{symbol},{symbol}\n" for symbol in expected),
                )
                self.assertIn(
                    "Skipping F growth score: growth unavailable", stderr.getvalue()
                )
                self.assertIn("Skipping M growth score:", stderr.getvalue())

    def test_conflicting_metrics_determine_selection_and_a_clear_final_winner(self):
        candidates = [quadrant.Candidate(symbol, symbol) for symbol in "HGFEDCBA"]
        growths = {
            "A": [10000, 90],
            "B": [6, 100],
            "C": [5, 70],
            "D": [4, 60],
            "E": [3, 80],
            "F": [2, 50],
            "G": [1, 30],
            "H": [0, 40],
        }
        scores = {
            "A": [10000, -100],
            "B": [2, 100],
            "C": [1, 10],
            "E": [0, 1],
        }

        # Growth averages A=1.5, B=1.5, C=3.5, E=4 beat D=4.5.
        # Valuation averages A=2.5, B=1.5, C=2.5, E=3.5.
        # Final reranked sums: B=2.5, A=4, C=5.5, E=8, with no ties.
        # Averaging raw scores would incorrectly put A ahead of B.
        def fetch(path, params):
            return (growths if path == "growths" else scores)[params["symbol"]]

        with (
            patch("quadrant.load_dotenv"),
            patch("quadrant.load_candidates", return_value=candidates),
            patch("quadrant.fetch_score", side_effect=fetch) as fetch_score,
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            self.assertEqual(quadrant.main(), 0)
        self.assertEqual(
            [
                request.args[1]["symbol"]
                for request in fetch_score.call_args_list
                if request.args[0] == "scores"
            ],
            ["A", "B", "C", "E"],
        )
        self.assertEqual(stdout.getvalue(), "B,B\nA,A\nC,C\nE,E\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_main_scores_only_best_half_without_backfilling_and_writes_csv(self):
        candidates = [
            quadrant.Candidate(symbol, description)
            for symbol, description in (
                ("A", "Alpha"),
                ("B", "Beta, Inc."),
                ("C", "Charlie"),
                ("D", "Delta"),
                ("E", "Echo"),
                ("N", "No growth"),
            )
        ]
        growths = {symbol: [5 - i, None] for i, symbol in enumerate("ABCDE")}
        growths["N"] = [None, None]
        scores = {"A": [None, None], "B": [-2, 0], "C": [0, None]}

        def fetch(path, params):
            return (growths if path == "growths" else scores)[params["symbol"]]

        with (
            patch("quadrant.load_dotenv"),
            patch("quadrant.load_candidates", return_value=candidates),
            patch("quadrant.fetch_score", side_effect=fetch) as fetch_score,
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            self.assertEqual(quadrant.main(), 0)
        self.assertEqual(
            [
                request.args[1]["symbol"]
                for request in fetch_score.call_args_list
                if request.args[0] == "scores"
            ],
            ["A", "B", "C"],
        )
        self.assertEqual(stdout.getvalue(), 'B,"Beta, Inc."\nC,Charlie\n')

    def test_main_valuation_failure_does_not_backfill(self):
        candidates = [quadrant.Candidate(symbol, symbol) for symbol in "ABCD"]

        def fetch(path, params):
            if path == "growths":
                return [4 - "ABCD".index(params["symbol"]), None]
            raise RuntimeError("unavailable")

        with (
            patch("quadrant.load_dotenv"),
            patch("quadrant.load_candidates", return_value=candidates),
            patch("quadrant.fetch_score", side_effect=fetch) as fetch_score,
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            self.assertEqual(quadrant.main(), 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            [
                request.args[1]["symbol"]
                for request in fetch_score.call_args_list
                if request.args[0] == "scores"
            ],
            ["A", "B"],
        )


class IntradayLiquidityTests(unittest.TestCase):
    def test_recent_trading_dollars_includes_at_most_last_n_calendar_days(self):
        with patch.object(quadrant, "LAST_N", 2):
            total = quadrant.recent_trading_dollars(
                [
                    ("2026-07-15", 4),
                    ("2026-07-17", 10),
                    ("2026-07-16", 5),
                ],
                "ABC",
                quadrant.dt.date(2026, 7, 17),
            )

        self.assertEqual(total, 15)

    def test_recent_trading_dollars_does_not_require_a_row_for_every_day(self):
        with patch.object(quadrant, "LAST_N", 2):
            total = quadrant.recent_trading_dollars(
                [("2026-07-17", 10)], "ABC", quadrant.dt.date(2026, 7, 17)
            )

        self.assertEqual(total, 10)

    def test_recent_trading_dollars_excludes_rows_outside_the_calendar_window(self):
        with patch.object(quadrant, "LAST_N", 2):
            total = quadrant.recent_trading_dollars(
                [
                    ("2026-07-18", 100),
                    ("2026-07-17", 10),
                    ("2026-07-15", 4),
                    ("2026-07-14", 20),
                ],
                "ABC",
                quadrant.dt.date(2026, 7, 17),
            )

        self.assertEqual(total, 10)

    def test_recent_trading_dollars_skips_when_no_rows_are_in_the_calendar_window(self):
        with (
            patch.object(quadrant, "LAST_N", 2),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            total = quadrant.recent_trading_dollars(
                [("2026-07-18", 100), ("2026-07-14", 20)],
                "ABC",
                quadrant.dt.date(2026, 7, 17),
            )

        self.assertIsNone(total)
        self.assertEqual(
            stderr.getvalue(), "Skipping ABC trading dollar data: no rows in 2 days\n"
        )

    def test_fmp_and_finmind_produce_the_same_total_for_equal_notionals(self):
        fmp_data = [
            {"date": "2026-07-15", "vwap": 2, "volume": 2},
            {"date": "2026-07-17", "vwap": 10, "volume": 1},
            {"date": "2026-07-16", "vwap": 5, "volume": 1},
        ]
        finmind_data = FakeTaiwanDailyData(
            [
                ("2026-07-15", 4),
                ("2026-07-17", 10),
                ("2026-07-16", 5),
            ]
        )

        with (
            patch.object(quadrant, "LAST_N", 2),
            patch("quadrant.new_york_regular_session_in_progress", return_value=False),
            patch("quadrant.taipei_regular_session_in_progress", return_value=False),
            patch(
                "quadrant.today_for_market", return_value=quadrant.dt.date(2026, 7, 17)
            ),
            patch("quadrant.cached_get_json", return_value=fmp_data),
            patch("quadrant.fetch_taiwan_stock_daily", return_value=finmind_data),
        ):
            fmp_result = quadrant.fetch_trading_dollar_fmp(
                quadrant.MARKET_CONFIG_BY_CODE["u"],
                "ABC",
                "Example Corp.",
                "2026-05-22",
                "test-key",
            )
            finmind_result = quadrant.fetch_trading_dollar_tw(
                quadrant.MARKET_CONFIG_BY_CODE["t"],
                "1234",
                "Example Corp.",
                "2026-05-22",
                "2026-07-17",
            )

        self.assertEqual(fmp_result.trading_dollars, 15)
        self.assertEqual(finmind_result.trading_dollars, 15)

    def test_fmp_and_finmind_include_available_rows_in_the_calendar_day_window(self):
        fmp_data = [{"date": "2026-07-17", "vwap": 10, "volume": 1}]
        finmind_data = FakeTaiwanDailyData([("2026-07-17", 10)])

        with (
            patch.object(quadrant, "LAST_N", 2),
            patch("quadrant.new_york_regular_session_in_progress", return_value=False),
            patch("quadrant.taipei_regular_session_in_progress", return_value=False),
            patch(
                "quadrant.today_for_market", return_value=quadrant.dt.date(2026, 7, 17)
            ),
            patch("quadrant.cached_get_json", return_value=fmp_data),
            patch("quadrant.fetch_taiwan_stock_daily", return_value=finmind_data),
        ):
            fmp_result = quadrant.fetch_trading_dollar_fmp(
                quadrant.MARKET_CONFIG_BY_CODE["u"],
                "ABC",
                "Example Corp.",
                "2026-05-22",
                "test-key",
            )
            finmind_result = quadrant.fetch_trading_dollar_tw(
                quadrant.MARKET_CONFIG_BY_CODE["t"],
                "1234",
                "Example Corp.",
                "2026-05-22",
                "2026-07-17",
            )

        self.assertEqual(
            fmp_result,
            quadrant.Candidate("ABC", "Example Corp.", market="u", trading_dollars=10),
        )
        self.assertEqual(
            finmind_result,
            quadrant.Candidate("1234", "Example Corp.", market="t", trading_dollars=10),
        )

    def test_new_york_regular_session_boundaries(self):
        timezone = quadrant.NEW_YORK_TIMEZONE

        for hour, minute, expected in [
            (9, 29, False),
            (9, 30, True),
            (15, 59, True),
            (16, 0, False),
        ]:
            with self.subTest(hour=hour, minute=minute):
                now = quadrant.dt.datetime(2026, 7, 17, hour, minute, tzinfo=timezone)
                self.assertEqual(
                    quadrant.new_york_regular_session_in_progress(now), expected
                )

    def test_taipei_regular_session_boundaries(self):
        timezone = quadrant.TAIPEI_TIMEZONE

        for hour, minute, expected in [
            (8, 59, False),
            (9, 0, True),
            (13, 29, True),
            (13, 30, False),
        ]:
            with self.subTest(hour=hour, minute=minute):
                now = quadrant.dt.datetime(2026, 7, 17, hour, minute, tzinfo=timezone)
                self.assertEqual(
                    quadrant.taipei_regular_session_in_progress(now), expected
                )

    def test_finmind_excludes_todays_row_during_taipei_regular_session(self):
        data = FakeTaiwanDailyData(
            [
                ("2026-07-17", 100),
                ("2026-07-16", 5),
                ("2026-07-15", 4),
            ]
        )

        with (
            patch.object(quadrant, "LAST_N", 2),
            patch("quadrant.taipei_regular_session_in_progress", return_value=True),
            patch(
                "quadrant.today_for_market", return_value=quadrant.dt.date(2026, 7, 17)
            ) as today,
            patch("quadrant.fetch_taiwan_stock_daily", return_value=data),
        ):
            result = quadrant.fetch_trading_dollar_tw(
                quadrant.MARKET_CONFIG_BY_CODE["t"],
                "1234",
                "Example Corp.",
                "2026-05-22",
                "2026-07-17",
            )

        today.assert_called_once_with(quadrant.MARKET_CONFIG_BY_CODE["t"])
        self.assertEqual(
            result,
            quadrant.Candidate("1234", "Example Corp.", market="t", trading_dollars=5),
        )

    def test_fmp_excludes_todays_row_during_new_york_regular_session(self):
        data = [
            {"date": "2026-07-17", "vwap": 100, "volume": 1},
            {"date": "2026-07-16", "vwap": 5, "volume": 1},
            {"date": "2026-07-15", "vwap": 4, "volume": 1},
        ]

        with (
            patch.object(quadrant, "LAST_N", 2),
            patch("quadrant.new_york_regular_session_in_progress", return_value=True),
            patch(
                "quadrant.today_for_market", return_value=quadrant.dt.date(2026, 7, 17)
            ) as today,
            patch("quadrant.cached_get_json", return_value=data) as fetch,
        ):
            result = quadrant.fetch_trading_dollar_fmp(
                quadrant.MARKET_CONFIG_BY_CODE["u"],
                "ABC",
                "Example Corp.",
                "2026-05-22",
                "test-key",
            )

        today.assert_called_once_with(quadrant.MARKET_CONFIG_BY_CODE["u"])
        fetch.assert_called_once_with(
            quadrant.FMP_STABLE_URL,
            params=[
                ("apikey", "test-key"),
                ("from", "2026-05-22"),
                ("symbol", "ABC"),
            ],
        )
        self.assertEqual(
            result,
            quadrant.Candidate("ABC", "Example Corp.", market="u", trading_dollars=5),
        )

    def test_fmp_keeps_todays_row_outside_the_us_market(self):
        data = {
            "historical": [
                {"date": "2026-07-17", "vwap": 100, "volume": 1},
                {"date": "2026-07-16", "vwap": 5, "volume": 1},
            ]
        }

        with (
            patch.object(quadrant, "LAST_N", 2),
            patch(
                "quadrant.new_york_regular_session_in_progress", return_value=True
            ) as in_session,
            patch(
                "quadrant.today_for_market", return_value=quadrant.dt.date(2026, 7, 17)
            ),
            patch("quadrant.cached_get_json", return_value=data) as fetch,
        ):
            result = quadrant.fetch_trading_dollar_fmp(
                quadrant.MARKET_CONFIG_BY_CODE["j"],
                "7203",
                "Toyota",
                "2026-05-22",
                "test-key",
            )

        in_session.assert_not_called()
        fetch.assert_called_once_with(
            f"{quadrant.FMP_LEGACY_URL}/7203.T",
            params=[
                ("apikey", "test-key"),
                ("from", "2026-05-22"),
            ],
        )
        self.assertEqual(
            result,
            quadrant.Candidate("7203", "Toyota", market="j", trading_dollars=105),
        )


if __name__ == "__main__":
    unittest.main()

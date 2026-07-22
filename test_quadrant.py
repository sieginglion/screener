import io
import sys
import tempfile
import types
import unittest
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
                quadrant.LiquidityResult("A", "Alpha", None),
                quadrant.LiquidityResult("B", "Beta", 2),
                quadrant.LiquidityResult("C", "Charlie", 1),
            ],
            limit=3,
            market="u",
        )

        self.assertEqual(
            result,
            [
                quadrant.Candidate(symbol="B", description="Beta"),
                quadrant.Candidate(symbol="C", description="Charlie"),
            ],
        )

    def test_fmp_loader_fetches_symbols_and_ranks_by_liquidity(self):
        def trading_dollars(market, symbol, description, from_date, api_key):
            return quadrant.LiquidityResult(
                symbol,
                description,
                {"A": 1, "B": 2}[symbol],
            )

        with (
            patch(
                "quadrant.today_for_market",
                return_value=quadrant.dt.date(2026, 7, 17),
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
                quadrant.Candidate(symbol="B", description="Beta"),
                quadrant.Candidate(symbol="A", description="Alpha; Inc."),
            ],
        )
        symbols.assert_called_once_with(market=self.market, top_n_symbols=3)
        today.assert_called_once_with(self.market)
        self.assertEqual(
            fetch.call_args_list,
            [
                call(self.market, "A", "Alpha; Inc.", "2026-05-22", "test-key"),
                call(self.market, "B", "Beta", "2026-05-22", "test-key"),
            ],
        )

    def test_loader_omits_candidates_without_enough_liquidity_data(self):
        def trading_dollars(market, symbol, description, from_date, api_key):
            return quadrant.LiquidityResult(
                symbol,
                description,
                {"A": None, "B": 2}[symbol],
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
            [quadrant.Candidate(symbol="B", description="Beta", market="u")],
        )

    def test_taiwan_loader_logs_in_and_ranks_by_liquidity(self):
        market = quadrant.MARKET_CONFIG_BY_CODE["t"]
        finmind = types.ModuleType("FinMind")
        finmind_data = types.ModuleType("FinMind.data")
        loader = Mock()
        data_loader = Mock(return_value=loader)
        finmind.data = finmind_data
        finmind_data.DataLoader = data_loader

        def trading_dollars(symbol, description, start, end):
            return quadrant.LiquidityResult(
                symbol,
                description,
                {"A": 1, "B": 2}[symbol],
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
                return_value=quadrant.dt.date(2026, 7, 17),
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
                quadrant.Candidate(symbol="B", description="Beta", market="t"),
                quadrant.Candidate(symbol="A", description="Alpha; Inc.", market="t"),
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
                call("A", "Alpha; Inc.", "2026-05-22", "2026-07-17"),
                call("B", "Beta", "2026-05-22", "2026-07-17"),
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
    def test_scores_are_kept_on_the_same_candidate(self):
        candidate = quadrant.Candidate(symbol="A", description="Alpha")

        with patch("quadrant.fetch_score", side_effect=[2, [3]]):
            growth_candidate = quadrant.score_growth(candidate)
            scored_candidate = quadrant.score_valuation(growth_candidate)

        self.assertEqual(
            scored_candidate,
            quadrant.Candidate(
                symbol="A",
                description="Alpha",
                growth_score=2,
                valuation_score=3,
            ),
        )
        self.assertEqual(scored_candidate.power, 6)

    def test_scoring_uses_the_expected_endpoint_parameters_and_parser(self):
        candidate = quadrant.Candidate(symbol="A", description="Alpha", market="u")

        with patch("quadrant.fetch_score", return_value=2) as fetch_score:
            quadrant.score_growth(candidate)

        fetch_score.assert_called_once_with(
            "growth",
            {"market": "u", "symbol": "A"},
        )

        with patch("quadrant.fetch_score", return_value=[2]) as fetch_score:
            quadrant.score_valuation(candidate)

        fetch_score.assert_called_once_with(
            "scores",
            {"market": "u", "symbol": "A", "q": quadrant.Q},
        )

    def test_scoring_failure_skips_the_candidate_with_a_message(self):
        candidate = quadrant.Candidate(symbol="A", description="Alpha")

        for scorer, score_label in (
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
                stderr.getvalue(),
                f"Skipping A {score_label}: unavailable\n",
            )

    def test_invalid_score_payload_skips_the_candidate_with_a_message(self):
        candidate = quadrant.Candidate(symbol="A", description="Alpha")

        for scorer, payload, score_label in (
            (quadrant.score_growth, "not-a-number", "growth score"),
            (quadrant.score_valuation, [], "valuation score"),
        ):
            with (
                self.subTest(scorer=scorer.__name__),
                patch("quadrant.fetch_score", return_value=payload),
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                self.assertIsNone(scorer(candidate))

            self.assertIn(f"Skipping A {score_label}:", stderr.getvalue())

    def test_growth_applies_the_bitcoin_multiplier_only_to_bitcoin(self):
        bitcoin = quadrant.Candidate(symbol="BTC", description="Bitcoin", market="c")
        stock = quadrant.Candidate(symbol="A", description="Alpha")

        with (
            patch.object(quadrant, "BTC_GROWTH_MULTIPLIER", 2),
            patch("quadrant.fetch_score", return_value=3),
        ):
            bitcoin_result = quadrant.score_growth(bitcoin)
            stock_result = quadrant.score_growth(stock)

        self.assertEqual(bitcoin_result.growth_score, 6)
        self.assertEqual(stock_result.growth_score, 3)

    def test_growth_scoring_can_be_disabled(self):
        candidates = [
            quadrant.Candidate(symbol="A", description="Alpha"),
            quadrant.Candidate(symbol="B", description="Beta"),
        ]

        with (
            patch.object(quadrant, "GROWTH_ENABLED", False),
            patch("quadrant.score_growth") as score_growth,
        ):
            result = quadrant.score_growth_candidates(candidates)

        score_growth.assert_not_called()
        self.assertEqual(
            result,
            [
                quadrant.Candidate(symbol="A", description="Alpha", growth_score=1),
                quadrant.Candidate(symbol="B", description="Beta", growth_score=1),
            ],
        )


class MainTests(unittest.TestCase):
    def test_main_writes_only_positive_complete_candidates_by_descending_power(self):
        source_candidates = [quadrant.Candidate(symbol="source", description="Source")]
        scored_candidates = [
            quadrant.Candidate(
                symbol="A", description="Alpha", growth_score=2, valuation_score=3
            ),
            quadrant.Candidate(
                symbol="B",
                description="Beta, Inc.",
                growth_score=3,
                valuation_score=3,
            ),
            quadrant.Candidate(symbol="C", description="Incomplete", growth_score=2),
            quadrant.Candidate(
                symbol="D", description="Zero", growth_score=1, valuation_score=0
            ),
            quadrant.Candidate(
                symbol="E", description="Negative", growth_score=-1, valuation_score=2
            ),
        ]

        with (
            patch("quadrant.load_dotenv"),
            patch("quadrant.load_candidates", return_value=source_candidates),
            patch(
                "quadrant.score_growth_candidates", return_value=scored_candidates
            ) as score_growth,
            patch(
                "quadrant.score_all_valuations", return_value=scored_candidates
            ) as score_valuations,
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            self.assertEqual(quadrant.main(), 0)

        score_growth.assert_called_once_with(source_candidates)
        score_valuations.assert_called_once_with(scored_candidates)
        self.assertEqual(stdout.getvalue(), 'B,"Beta, Inc."\nA,Alpha\n')


class IntradayLiquidityTests(unittest.TestCase):
    def test_recent_trading_dollars_uses_only_the_latest_complete_sessions(self):
        with patch.object(quadrant, "LAST_N", 2):
            total = quadrant.recent_trading_dollars(
                [
                    ("2026-07-15", 4),
                    ("2026-07-17", 10),
                    ("2026-07-16", 5),
                ],
                "ABC",
            )

        self.assertEqual(total, 15)

    def test_recent_trading_dollars_requires_all_sessions(self):
        with (
            patch.object(quadrant, "LAST_N", 2),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            total = quadrant.recent_trading_dollars([("2026-07-17", 10)], "ABC")

        self.assertIsNone(total)
        self.assertEqual(
            stderr.getvalue(),
            "Skipping ABC trading dollar data: only 1 rows, need 2\n",
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
                "1234",
                "Example Corp.",
                "2026-05-22",
                "2026-07-17",
            )

        self.assertEqual(fmp_result.trading_dollars, 15)
        self.assertEqual(finmind_result.trading_dollars, 15)

    def test_fmp_and_finmind_exclude_insufficient_rows(self):
        fmp_data = [{"date": "2026-07-17", "vwap": 10, "volume": 1}]
        finmind_data = FakeTaiwanDailyData([("2026-07-17", 10)])

        with (
            patch.object(quadrant, "LAST_N", 2),
            patch("quadrant.new_york_regular_session_in_progress", return_value=False),
            patch("quadrant.cached_get_json", return_value=fmp_data),
            patch("quadrant.fetch_taiwan_stock_daily", return_value=finmind_data),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            fmp_result = quadrant.fetch_trading_dollar_fmp(
                quadrant.MARKET_CONFIG_BY_CODE["u"],
                "ABC",
                "Example Corp.",
                "2026-05-22",
                "test-key",
            )
            finmind_result = quadrant.fetch_trading_dollar_tw(
                "1234",
                "Example Corp.",
                "2026-05-22",
                "2026-07-17",
            )

        self.assertEqual(
            fmp_result,
            quadrant.LiquidityResult("ABC", "Example Corp.", None),
        )
        self.assertEqual(
            finmind_result,
            quadrant.LiquidityResult("1234", "Example Corp.", None),
        )
        self.assertEqual(
            stderr.getvalue(),
            "Skipping ABC trading dollar data: only 1 rows, need 2\n"
            "Skipping 1234 trading dollar data: only 1 rows, need 2\n",
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
            quadrant.LiquidityResult("ABC", "Example Corp.", 9),
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
            quadrant.LiquidityResult("7203", "Toyota", 105),
        )


if __name__ == "__main__":
    unittest.main()

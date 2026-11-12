import io
import tempfile
import types
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, Mock, call, patch
from zoneinfo import ZoneInfo

import pandas as pd

import clear_ticker_cache
import quadrant


class FakeTaiwanDailyData:
    def __init__(self, rows):
        self.rows = rows
        self.columns = ["date", "Trading_money"]

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

    def test_finmind_cache_is_reused_across_clients_with_the_existing_key(self):
        rows = pd.DataFrame({"date": ["2026-07-17"], "Trading_money": [10]})
        first_client = Mock()
        first_client.taiwan_stock_daily.return_value = rows
        second_client = Mock()
        args = ("2330", "2026-07-10", "2026-07-17")

        pd.testing.assert_frame_equal(
            quadrant.fetch_taiwan_stock_daily(*args, client=first_client), rows
        )
        key = (quadrant.__name__, "fetch_taiwan_stock_daily", args, ())
        self.assertIn(key, self.cache)
        pd.testing.assert_frame_equal(
            quadrant.fetch_taiwan_stock_daily(*args, client=second_client), rows
        )
        first_client.taiwan_stock_daily.assert_called_once_with(
            stock_id=args[0], start_date=args[1], end_date=args[2]
        )
        second_client.taiwan_stock_daily.assert_not_called()

    def test_finmind_does_not_cache_failed_requests(self):
        client = Mock()
        client.taiwan_stock_daily.side_effect = RuntimeError("unavailable")
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                quadrant.fetch_taiwan_stock_daily(
                    "2330", "2026-07-10", "2026-07-17", client=client
                )
        self.assertEqual(client.taiwan_stock_daily.call_count, 2)


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

        self.assertEqual(us.liquidity_source, "fmp")
        self.assertEqual(us.timezone, "America/New_York")
        self.assertTrue(us.include_bitcoin)

        self.assertEqual(taiwan.liquidity_source, "finmind")
        self.assertEqual(taiwan.timezone, "Asia/Taipei")
        self.assertFalse(taiwan.include_bitcoin)


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
                    quadrant.fetch_symbols_from_tv(
                        market, top_n_symbols=1, sort_window="1W"
                    ),
                    [("BRK-B", "Berkshire")],
                )

        self.assertEqual(len(client.post.call_args_list), 2)
        for market, request in zip(
            quadrant.MARKET_CONFIG_BY_CODE.values(), client.post.call_args_list
        ):
            with self.subTest(market=market.tradingview_market):
                self.assertEqual(request.args[0], market.tradingview_url)
                self.assertEqual(request.kwargs["headers"], market.tradingview_headers)
                self.assertEqual(
                    request.kwargs["json"],
                    quadrant.tradingview_payload(market, limit=1, sort_window="1W"),
                )


def sample_window(code="u", hour=17):
    market = quadrant.MARKET_CONFIG_BY_CODE[code]
    now = quadrant.dt.datetime(2026, 7, 17, hour, tzinfo=ZoneInfo(market.timezone))
    dates = [quadrant.dt.date(2026, 7, day) for day in (10, 13, 14, 15, 16, 17)]
    return quadrant.select_screening_window(
        market, now, now - quadrant.dt.timedelta(days=7), dates
    )


class LoadTopCandidatesTests(unittest.TestCase):
    stocks = [("A", "Alpha; Inc."), ("B", "Beta")]

    def test_loader_rejects_a_pool_that_cannot_be_reranked(self):
        with patch("quadrant.fetch_symbols_from_tv") as symbols:
            with self.assertRaisesRegex(ValueError, "top_n_symbols must be greater"):
                quadrant.load_top_candidates(
                    quadrant.MARKET_CONFIG_BY_CODE["u"],
                    "test-key",
                    2,
                    2,
                    window=sample_window(),
                )
        symbols.assert_not_called()

    def test_top_candidates_excludes_insufficient_liquidity_data(self):
        candidates = [
            quadrant.Candidate("A", "Alpha", trading_dollars=None),
            quadrant.Candidate("B", "Beta", trading_dollars=2),
            quadrant.Candidate("C", "Charlie", trading_dollars=1),
        ]
        self.assertEqual(quadrant.top_candidates(candidates, 3), candidates[1:])

    def test_fmp_loader_passes_window_and_ranks_liquidity(self):
        market = quadrant.MARKET_CONFIG_BY_CODE["u"]
        window = sample_window()

        def trading_dollars(market, symbol, description, api_key, window):
            return quadrant.Candidate(
                symbol,
                description,
                market=market.code,
                trading_dollars={"A": 1, "B": 2}[symbol],
            )

        with (
            patch(
                "quadrant.fetch_symbols_from_tv", return_value=self.stocks
            ) as symbols,
            patch(
                "quadrant.fetch_trading_dollar_fmp", side_effect=trading_dollars
            ) as fetch,
        ):
            result = quadrant.load_top_candidates(market, "test-key", 3, 2, window)
        self.assertEqual([c.symbol for c in result], ["B", "A"])
        symbols.assert_called_once_with(
            market=market, top_n_symbols=3, sort_window="1W"
        )
        self.assertEqual(
            fetch.call_args_list,
            [
                call(
                    market,
                    symbol,
                    description,
                    "test-key",
                    window,
                )
                for symbol, description in self.stocks
            ],
        )

    def test_taiwan_loader_logs_in_and_passes_window(self):
        market = quadrant.MARKET_CONFIG_BY_CODE["t"]
        window = sample_window("t")
        candidates = [
            quadrant.Candidate("A", "Alpha; Inc.", market="t"),
            quadrant.Candidate("B", "Beta", market="t", trading_dollars=2),
        ]
        with (
            patch.dict(quadrant.os.environ, {"FINMIND_KEY": "test-key"}),
            patch("FinMind.data.DataLoader") as loader,
            patch(
                "quadrant.fetch_symbols_from_tv", return_value=self.stocks
            ) as symbols,
            patch("quadrant.fetch_trading_dollar_tw", side_effect=candidates) as fetch,
            patch("quadrant.fetch_trading_dollars_fmp") as fmp,
        ):
            result = quadrant.load_top_candidates(market, "test-key", 3, 2, window)
        self.assertEqual(result, candidates[1:])
        loader.return_value.login_by_token.assert_called_once_with("test-key")
        symbols.assert_called_once_with(
            market=market, top_n_symbols=3, sort_window="1W"
        )
        self.assertEqual(
            fetch.call_args_list,
            [
                call(market, symbol, description, window, client=loader.return_value)
                for symbol, description in self.stocks
            ],
        )
        fmp.assert_not_called()


class CandidateLoadingTests(unittest.TestCase):
    def test_bitcoin_and_automatic_pool_are_market_specific(self):
        for code, market in quadrant.MARKET_CONFIG_BY_CODE.items():
            window = sample_window(code)
            source = [quadrant.Candidate("ABC", "Example", market=code)]
            with (
                self.subTest(market=code),
                patch("quadrant.current_market_config", return_value=market),
                patch(
                    "quadrant.load_screening_window", return_value=window
                ) as automatic,
                patch(
                    "quadrant.load_top_candidates", return_value=list(source)
                ) as load,
            ):
                result = quadrant.load_candidates()
            expected = source + (
                [quadrant.Candidate("BTC", "Bitcoin", market="c")]
                if code == "u"
                else []
            )
            self.assertEqual(result, expected)
            self.assertIs(load.call_args.kwargs["window"], window)
            multiplier = window.candidate_pool_multiplier
            self.assertEqual(
                load.call_args.kwargs["top_n_symbols"],
                quadrant.candidate_pool_size(quadrant.RESULT_LIMIT, multiplier),
            )
            automatic.assert_called_once_with(
                market, quadrant.LAST_N, quadrant.os.environ.get("FMP_API_KEY")
            )


class GrowthScoringTests(unittest.TestCase):
    def test_first_metric_is_required_and_second_metric_is_optional(self):
        cases = (
            (
                quadrant.score_growth,
                ("revenue_growth", "eps_growth"),
                "growth_rank",
                (-2, 0, 2),
                {-1.0: 3.0, 0.0: 2.5, 1.0: 2.0},
            ),
            (
                quadrant.score_valuation,
                ("ps_score", "pe_score"),
                "valuation_rank",
                (0, 0.5, 0.75),
                {0.0: 3.5, 0.5: 2.5, 0.9: 1.0},
            ),
        )
        for scorer, fields, rank_field, reference_values, expected_ranks in cases:
            references = [
                replace(
                    quadrant.Candidate(symbol, symbol), **dict.fromkeys(fields, value)
                )
                for symbol, value in zip(("low", "middle", "high"), reference_values)
            ]
            for first in (None, *expected_ranks):
                for second in (None, *expected_ranks):
                    with (
                        self.subTest(scorer=scorer.__name__, pair=(first, second)),
                        patch("quadrant.fetch_score", return_value=[first, second]),
                    ):
                        scored = scorer(quadrant.Candidate("target", "Target"))
                        if first is None:
                            self.assertIsNone(scored)
                            continue
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
        with patch("quadrant.fetch_score", side_effect=[[2.0, -1.0], [0, None]]):
            grown = quadrant.score_growth(candidate)
            assert grown is not None
            scored = quadrant.score_valuation(grown)
        self.assertEqual(
            scored,
            quadrant.Candidate(
                "A",
                "Alpha",
                trading_dollars=123,
                revenue_growth=2.0,
                eps_growth=-1.0,
                ps_score=0,
            ),
        )

    def test_scoring_uses_the_expected_endpoints_and_parameters(self):
        candidate = quadrant.Candidate("A", "Alpha")
        with patch("quadrant.fetch_score", side_effect=[[0, -2], [0, 0.75]]) as fetch:
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
                    },
                ),
            ],
        )
        assert grown is not None and scored is not None
        self.assertEqual(grown.revenue_growth, 0)
        self.assertEqual(grown.eps_growth, -2.0)
        self.assertEqual(scored.ps_score, 0)
        self.assertEqual(scored.pe_score, 0.75)

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

    def test_unusable_score_pairs_skip_the_candidate(self):
        candidate = quadrant.Candidate("A", "Alpha")
        for scorer in (quadrant.score_growth, quadrant.score_valuation):
            for payload in (
                "invalid",
                [],
                [1],
                [1, 2, 3],
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
            patch("quadrant.fetch_score", return_value=[-3.0, None]),
        ):
            bitcoin = quadrant.score_growth(
                quadrant.Candidate("BTC", "Bitcoin", market="c")
            )
            stock = quadrant.score_growth(quadrant.Candidate("A", "Alpha"))
        assert bitcoin is not None and stock is not None
        self.assertEqual(bitcoin.revenue_growth, -6)
        self.assertIsNone(bitcoin.eps_growth)
        self.assertEqual(stock.revenue_growth, -3)

    def test_batch_scoring_preserves_order_and_skips_failed_requests(self):
        candidates = [quadrant.Candidate(symbol, symbol) for symbol in "ABC"]
        for workers in (None, 2):
            with self.subTest(max_workers=workers):
                score = Mock(side_effect=lambda c: None if c.symbol == "B" else c)
                result = quadrant.score_candidates(
                    iter(candidates), score, max_workers=workers
                )
                self.assertEqual(result, [candidates[0], candidates[2]])
                self.assertEqual(score.call_count, len(candidates))
                score.assert_has_calls([call(c) for c in candidates], any_order=True)
                score.reset_mock()
                self.assertEqual(
                    quadrant.score_candidates([], score, max_workers=workers), []
                )
                score.assert_not_called()


class RankingTests(unittest.TestCase):
    def test_independent_growth_ranks_average_available_metrics(self):
        candidates = [
            quadrant.Candidate("A", "Alpha", revenue_growth=100, eps_growth=-2),
            quadrant.Candidate("B", "Beta", revenue_growth=2, eps_growth=0),
            quadrant.Candidate("C", "Charlie", revenue_growth=-1),
            quadrant.Candidate("D", "Delta", eps_growth=1000),
            quadrant.Candidate("E", "Empty"),
        ]
        result = quadrant.combine_metric_ranks(
            candidates, ("revenue_growth", "eps_growth"), "growth_rank"
        )
        self.assertEqual(
            {candidate.symbol: candidate.growth_rank for candidate in result},
            {"A": 1.5, "B": 1.5, "C": 3},
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
            quadrant.Candidate(symbol, symbol, revenue_growth=0)
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
            quadrant.Candidate("Z", "Zulu", growth_rank=1, ps_score=0.2, pe_score=0),
            quadrant.Candidate("B", "Beta", growth_rank=20, ps_score=0.1, pe_score=0.4),
            quadrant.Candidate(
                "A", "Alpha", growth_rank=30, ps_score=0.3, pe_score=0.2
            ),
            quadrant.Candidate("C", "C", growth_rank=2, ps_score=-1, pe_score=-1),
            quadrant.Candidate("D", "D", growth_rank=3, ps_score=-2, pe_score=-2),
            quadrant.Candidate("E", "E", growth_rank=4, ps_score=-3, pe_score=-3),
        ]
        # Only A, B, Z survive valuation selection.
        # Their growth ranks 1,2,3 and valuation ranks 3,2,1 tie; symbols decide.
        # Adding original growth values would incorrectly produce Z,B,A.
        self.assertEqual(
            [candidate.symbol for candidate in quadrant.rank_candidates(candidates)],
            ["A", "B", "Z"],
        )

    def test_final_ranking_requires_first_score_before_halving(
        self,
    ):
        candidates = [
            quadrant.Candidate("A", "Alpha", growth_rank=1, ps_score=0),
            quadrant.Candidate("B", "Beta", growth_rank=2, pe_score=0.2),
            quadrant.Candidate("C", "No valuation", growth_rank=3),
            quadrant.Candidate("D", "No growth", ps_score=0.9, pe_score=0.9),
            quadrant.Candidate("E", "E", growth_rank=4, ps_score=-1),
            quadrant.Candidate("F", "F", growth_rank=5, pe_score=-0.2),
        ]
        self.assertEqual(
            [candidate.symbol for candidate in quadrant.rank_candidates(candidates)],
            ["A"],
        )
        self.assertEqual(quadrant.rank_candidates([]), [])

    def test_valuation_cutoff_rounds_up_and_keeps_at_least_one(self):
        for count in range(6):
            candidates = [
                quadrant.Candidate(str(i), str(i), growth_rank=1, ps_score=-i)
                for i in reversed(range(count))
            ]
            candidates.append(quadrant.Candidate("missing", "Missing", growth_rank=1))
            with self.subTest(count=count):
                self.assertEqual(
                    [item.symbol for item in quadrant.rank_candidates(candidates)],
                    [str(i) for i in range((count + 1) // 2)],
                )

    def test_valuation_cutoff_breaks_ties_alphabetically(self):
        candidates = [
            quadrant.Candidate(symbol, symbol, growth_rank=1, ps_score=0, pe_score=0)
            for symbol in "DCBA"
        ]
        self.assertEqual(
            [item.symbol for item in quadrant.rank_candidates(candidates)],
            ["A", "B"],
        )

    def test_valuation_ranks_average_available_components(self):
        candidates = [
            quadrant.Candidate("A", "Alpha", ps_score=0.5, pe_score=0.9),
            quadrant.Candidate("B", "Beta", ps_score=0.9, pe_score=0.5),
            quadrant.Candidate("C", "Charlie", ps_score=0),
            quadrant.Candidate("D", "Delta", pe_score=1000),
        ]
        result = quadrant.combine_metric_ranks(
            candidates, ("ps_score", "pe_score"), "valuation_rank"
        )
        self.assertEqual(
            {candidate.symbol: candidate.valuation_rank for candidate in result},
            {"A": 1.5, "B": 1.5, "C": 3},
        )


class MainTests(unittest.TestCase):
    def test_growth_failures_and_missing_growths_are_removed_before_halving(self):
        for eligible_count in (4, 5):
            symbols = "ABCDE"[:eligible_count]
            candidates = [
                quadrant.Candidate(symbol, symbol) for symbol in symbols + "FMN"
            ]
            growths = {symbol: [-i - 1.0, None] for i, symbol in enumerate(symbols)}
            growths.update({"M": [1], "N": [None, 1000]})

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
                    "".join(
                        f"{symbol},{symbol}\n"
                        for symbol in expected[: (len(expected) + 1) // 2]
                    ),
                )
                self.assertIn(
                    "Skipping F growth score: growth unavailable", stderr.getvalue()
                )
                self.assertIn("Skipping M growth score:", stderr.getvalue())

    def test_conflicting_metrics_determine_selection_and_a_clear_final_winner(self):
        candidates = [quadrant.Candidate(symbol, symbol) for symbol in "HGFEDCBA"]
        growths = {
            "A": [10000.0, 90.0],
            "B": [6.0, 100.0],
            "C": [5.0, 70.0],
            "D": [4.0, 60.0],
            "E": [3.0, 80.0],
            "F": [2.0, 50.0],
            "G": [1.0, 30.0],
            "H": [0.0, 40.0],
        }
        scores = {
            "A": [0.99, 0],
            "B": [0.02, 0.9],
            "C": [0.01, 0.1],
            "E": [0, 0.01],
        }

        # Growth averages A=1.5, B=1.5, C=3.5, E=4 beat D=4.5.
        # Valuation averages A=2.5, B=1.5, C=2.5, E=3.5.
        # B and A survive valuation selection; B wins on valuation.
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
        self.assertEqual(stdout.getvalue(), "B,B\nA,A\n")
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
        growths = {symbol: [5.0 - i, None] for i, symbol in enumerate("ABCDE")}
        growths["N"] = [None, None]
        scores = {"A": [None, None], "B": [0, 0.5], "C": [0, None]}

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
        self.assertEqual(stdout.getvalue(), 'B,"Beta, Inc."\n')

    def test_valuation_failures_and_missing_scores_are_removed_before_halving(self):
        candidates = [quadrant.Candidate(symbol, symbol) for symbol in "ABCDEFGHIJKL"]
        for eligible_count in (4, 5):

            def fetch(path, params):
                symbol = params["symbol"]
                if path == "growths":
                    return [-ord(symbol), None]
                if symbol == "F":
                    return [None, 1000]
                if symbol == "E" and eligible_count == 4:
                    raise RuntimeError("valuation unavailable")
                return [ord(symbol), None]

            with (
                self.subTest(eligible_count=eligible_count),
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
                    list("ABCDEF"),
                )
                expected = "CD" if eligible_count == 4 else "CDE"
                self.assertEqual(
                    stdout.getvalue(),
                    "".join(f"{symbol},{symbol}\n" for symbol in expected),
                )
                if eligible_count == 4:
                    self.assertIn(
                        "Skipping E valuation score: valuation unavailable",
                        stderr.getvalue(),
                    )

    def test_main_valuation_failure_does_not_backfill(self):
        candidates = [quadrant.Candidate(symbol, symbol) for symbol in "ABCD"]

        def fetch(path, params):
            if path == "growths":
                return [4.0 - "ABCD".index(params["symbol"]), None]
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
    def test_finmind_skips_empty_or_incomplete_responses(self):
        window = sample_window("t")
        for data, missing_columns in (
            (pd.DataFrame(), "date, Trading_money"),
            (pd.DataFrame({"date": ["2026-07-17"]}), "Trading_money"),
            (pd.DataFrame({"Trading_money": [10]}), "date"),
        ):
            with (
                self.subTest(columns=list(data.columns)),
                patch("quadrant.fetch_taiwan_stock_daily", return_value=data),
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                result = quadrant.fetch_trading_dollar_tw(
                    quadrant.MARKET_CONFIG_BY_CODE["t"],
                    "7953",
                    "Example Corp.",
                    window,
                    client=Mock(),
                )
            self.assertEqual(
                result, quadrant.Candidate("7953", "Example Corp.", market="t")
            )
            self.assertEqual(quadrant.top_candidates([result], limit=1), [])
            self.assertEqual(
                stderr.getvalue(),
                "Skipping 7953 trading dollar data: "
                f"FinMind response missing columns: {missing_columns}\n",
            )

    def test_finmind_empty_response_does_not_stop_other_symbols(self):
        market = quadrant.MARKET_CONFIG_BY_CODE["t"]
        data_by_symbol = {
            "7953": pd.DataFrame(),
            "2330": pd.DataFrame({"date": ["2026-07-17"], "Trading_money": [10]}),
        }
        with (
            patch.dict(quadrant.os.environ, {"FINMIND_KEY": "test-key"}),
            patch("FinMind.data.DataLoader"),
            patch(
                "quadrant.fetch_taiwan_stock_daily",
                side_effect=lambda symbol, start, end, *, client: data_by_symbol[
                    symbol
                ],
            ),
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            candidates = quadrant.fetch_trading_dollars_tw(
                market, [("7953", "No data"), ("2330", "TSMC")], sample_window("t")
            )
        self.assertEqual(
            quadrant.top_candidates(candidates, limit=2),
            [quadrant.Candidate("2330", "TSMC", market="t", trading_dollars=10)],
        )

    def test_fmp_and_finmind_use_identical_close_boundaries(self):
        for code in ("u", "t"):
            market = quadrant.MARKET_CONFIG_BY_CODE[code]
            close = quadrant.trading_close_time(market, quadrant.dt.date(2026, 7, 17))
            for delta, expected in ((-1, 15), (0, 115), (1, 111)):
                now = close + quadrant.dt.timedelta(seconds=delta)
                dates = [quadrant.dt.date(2026, 7, d) for d in (10, 13, 16, 17)]
                window = quadrant.select_screening_window(
                    market, now, now - quadrant.dt.timedelta(days=7), dates
                )
                # The oldest close is included at the exact lower boundary too.
                rows = [
                    ("2026-07-09", 1000),
                    ("2026-07-10", 4),
                    ("2026-07-13", 1),
                    ("2026-07-16", 10),
                    ("2026-07-17", 100),
                    ("2026-07-18", 1000),
                ]
                with (
                    self.subTest(code=code, delta=delta),
                    patch(
                        "quadrant.cached_get_json",
                        return_value=[
                            {"date": date, "vwap": value, "volume": 1}
                            for date, value in rows
                        ],
                    ) as fmp,
                    patch(
                        "quadrant.fetch_taiwan_stock_daily",
                        return_value=FakeTaiwanDailyData(rows),
                    ) as finmind,
                ):
                    if code == "u":
                        result = quadrant.fetch_trading_dollar_fmp(
                            market,
                            "ABC",
                            "Example",
                            "test-key",
                            window,
                        )
                        self.assertIn(
                            ("to", window.request_end), fmp.call_args.kwargs["params"]
                        )
                    else:
                        client = Mock()
                        result = quadrant.fetch_trading_dollar_tw(
                            market, "1234", "Example", window, client=client
                        )
                        finmind.assert_called_once_with(
                            "1234",
                            window.request_start,
                            window.request_end,
                            client=client,
                        )
                self.assertEqual(result.trading_dollars, expected)
                self.assertEqual(
                    window.request_end, "2026-07-16" if delta < 0 else "2026-07-17"
                )

    def test_no_available_liquidity_rows_skips_symbol(self):
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            result = quadrant.recent_trading_dollars(
                [("2026-07-01", 10)],
                "ABC",
                quadrant.MARKET_CONFIG_BY_CODE["u"],
                sample_window(),
            )
        self.assertIsNone(result)
        self.assertIn("Skipping ABC trading dollar data", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

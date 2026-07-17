import sys
import types
import unittest
from unittest.mock import Mock, call, patch

import quadrant


class LoadTopCandidatesTests(unittest.TestCase):
    tv_config = quadrant.TradingViewConfig(
        url="https://example.com/scan",
        headers={},
        market_name="test",
        language="en",
    )
    stocks = [("A", "Alpha; Inc."), ("B", "Beta")]

    def test_loader_rejects_a_pool_that_cannot_be_reranked(self):
        with (
            patch("quadrant.current_tv_config") as config,
            patch("quadrant.fetch_symbols_from_tv") as symbols,
        ):
            with self.assertRaisesRegex(ValueError, "top_n_symbols must be greater"):
                quadrant.load_top_candidates(
                    api_key="test-key",
                    top_n_symbols=2,
                    top_n_results=2,
                )

        config.assert_not_called()
        symbols.assert_not_called()

    def test_fmp_loader_fetches_symbols_and_ranks_by_liquidity(self):
        def trading_dollars(symbol, description, from_date, api_key):
            return symbol, description, {"A": 1, "B": 2}[symbol]

        with (
            patch("quadrant.current_tv_config", return_value=self.tv_config),
            patch(
                "quadrant.today_for_market",
                return_value=quadrant.dt.date(2026, 7, 17),
            ),
            patch(
                "quadrant.fetch_symbols_from_tv", return_value=self.stocks
            ) as symbols,
            patch(
                "quadrant.fetch_trading_dollar_fmp",
                side_effect=trading_dollars,
            ) as fetch,
        ):
            result = quadrant.load_top_candidates(
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
        symbols.assert_called_once_with(tv_config=self.tv_config, top_n_symbols=3)
        self.assertEqual(
            fetch.call_args_list,
            [
                call("A", "Alpha; Inc.", "2026-05-22", "test-key"),
                call("B", "Beta", "2026-05-22", "test-key"),
            ],
        )

    def test_taiwan_loader_logs_in_and_ranks_by_liquidity(self):
        finmind = types.ModuleType("FinMind")
        finmind_data = types.ModuleType("FinMind.data")
        loader = Mock()
        data_loader = Mock(return_value=loader)
        finmind.data = finmind_data
        finmind_data.DataLoader = data_loader

        def trading_dollars(symbol, description, start, end):
            return symbol, description, {"A": 1, "B": 2}[symbol]

        with (
            patch.dict(
                sys.modules,
                {"FinMind": finmind, "FinMind.data": finmind_data},
            ),
            patch.dict(quadrant.os.environ, {"FINMIND_KEY": "test-key"}),
            patch.object(quadrant, "finmind_api", None),
            patch.object(quadrant, "MARKET", "t"),
            patch("quadrant.current_tv_config", return_value=self.tv_config),
            patch(
                "quadrant.today_for_market",
                return_value=quadrant.dt.date(2026, 7, 17),
            ),
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
                api_key=None,
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
        data_loader.assert_called_once_with()
        loader.login_by_token.assert_called_once_with("test-key")
        symbols.assert_called_once_with(tv_config=self.tv_config, top_n_symbols=3)
        fmp.assert_not_called()
        self.assertEqual(
            fetch.call_args_list,
            [
                call("A", "Alpha; Inc.", "2026-05-22", "2026-07-17"),
                call("B", "Beta", "2026-05-22", "2026-07-17"),
            ],
        )


class GrowthScoringTests(unittest.TestCase):
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
                quadrant.GrowthCandidate(candidate=candidates[0], growth_score=1),
                quadrant.GrowthCandidate(candidate=candidates[1], growth_score=1),
            ],
        )


class IntradayLiquidityTests(unittest.TestCase):
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
        response = Mock()
        response.json.return_value = [
            {"date": "2026-07-17", "vwap": 100, "volume": 1},
            {"date": "2026-07-16", "vwap": 5, "volume": 1},
            {"date": "2026-07-15", "vwap": 4, "volume": 1},
        ]

        with (
            patch.object(quadrant, "MARKET", "u"),
            patch.object(quadrant, "LAST_N", 2),
            patch("quadrant.new_york_regular_session_in_progress", return_value=True),
            patch(
                "quadrant.today_for_market", return_value=quadrant.dt.date(2026, 7, 17)
            ),
            patch("quadrant.cached_httpx_get", return_value=response),
        ):
            result = quadrant.fetch_trading_dollar_fmp(
                "ABC", "Example Corp.", "2026-05-22", "test-key"
            )

        self.assertEqual(result, ("ABC", "Example Corp.", 9))

    def test_fmp_keeps_todays_row_outside_the_us_market(self):
        response = Mock()
        response.json.return_value = {
            "historical": [
                {"date": "2026-07-17", "vwap": 100, "volume": 1},
                {"date": "2026-07-16", "vwap": 5, "volume": 1},
            ]
        }

        with (
            patch.object(quadrant, "MARKET", "j"),
            patch.object(quadrant, "LAST_N", 2),
            patch(
                "quadrant.new_york_regular_session_in_progress", return_value=True
            ) as in_session,
            patch("quadrant.cached_httpx_get", return_value=response),
        ):
            result = quadrant.fetch_trading_dollar_fmp(
                "7203", "Toyota", "2026-05-22", "test-key"
            )

        in_session.assert_not_called()
        self.assertEqual(result, ("7203", "Toyota", 105))


if __name__ == "__main__":
    unittest.main()

import datetime as dt
import io
import math
import tempfile
import unittest
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pandas as pd

import quadrant


def weekdays(start: str, end: str) -> list[dt.date]:
    first = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    return [
        day
        for offset in range((last - first).days + 1)
        if (day := first + dt.timedelta(days=offset)).weekday() < 5
    ]


def local_now(code: str, value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value).replace(
        tzinfo=ZoneInfo(quadrant.MARKET_CONFIG_BY_CODE[code].timezone)
    )


class WindowSelectionTests(unittest.TestCase):
    def select(self, value, last_n=14, dates=None, code="u"):
        if dates is None:
            dates = weekdays("2026-06-01", "2026-10-31")
        now = local_now(code, value)
        return quadrant.select_screening_window(
            quadrant.MARKET_CONFIG_BY_CODE[code],
            now,
            now - dt.timedelta(days=last_n),
            dates,
        )

    def test_allowed_values_and_rejected_types(self):
        for value in (7, 14, 28):
            with self.subTest(value=value):
                quadrant.validate_last_n(value)
        for value in (0, -7, 8, 30, 7.0, "7", True, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    quadrant.ScreeningWindowError, "7, 14, or 28"
                ):
                    quadrant.validate_last_n(value)

    def test_longer_eligible_window_and_multiplier(self):
        for code in ("u", "t"):
            with self.subTest(code=code):
                window = self.select("2026-07-10T17:00", code=code)
                self.assertEqual(window.tv_sort_window, "1M")
                self.assertEqual(window.lookback_sessions, 10)
                self.assertEqual(window.window_sessions, 8)
                self.assertEqual(window.candidate_pool_multiplier, 10 / 8 * 1.1)
                self.assertEqual(
                    quadrant.candidate_pool_size(101, window.candidate_pool_multiplier),
                    math.ceil(101 * 10 / 8 * 1.1),
                )

    def test_month_ineligible_and_shorter_week_expands_pool(self):
        window = self.select("2026-07-17T17:00")
        self.assertEqual(window.tv_sort_window, "1W")
        self.assertEqual(window.lookback_sessions, 10)
        self.assertEqual(window.window_sessions, 5)
        self.assertEqual(window.candidate_pool_multiplier, 2.2)

    def test_ties_prefer_week_even_with_different_start_dates(self):
        for code in ("u", "t"):
            with self.subTest(code=code):
                window = self.select("2026-08-04T17:00", 7, code=code)
                self.assertEqual(window.tv_sort_window, "1W")
                self.assertEqual(
                    window.tv_sort_start, local_now(code, "2026-08-03T00:00")
                )
                self.assertEqual(window.window_sessions, 2)

    def test_week_and_month_reset_at_local_midnight(self):
        sunday = self.select("2026-07-19T23:59:59", 7)
        self.assertEqual(sunday.tv_sort_start, local_now("u", "2026-07-13T00:00"))
        with self.assertRaisesRegex(
            quadrant.ScreeningWindowError, "eligible TradingView"
        ):
            self.select("2026-07-20T00:00", 7)
        first = self.select("2026-10-01T00:00", 14)
        self.assertEqual(first.tv_sort_window, "1W")
        self.assertEqual(first.tv_sort_start, local_now("u", "2026-09-28T00:00"))
        later = self.select("2026-10-06T17:00", 14)
        self.assertEqual(later.tv_sort_window, "1M")
        self.assertEqual(later.tv_sort_start, local_now("u", "2026-10-01T00:00"))

    def test_month_start_is_eligible_at_exact_lower_boundary_only(self):
        exact = self.select("2026-07-29T00:00", 28)
        self.assertEqual(exact.lookback_start, exact.tv_sort_start)
        self.assertEqual(exact.tv_sort_window, "1M")
        after = self.select("2026-07-29T00:00:00.000001", 28)
        self.assertEqual(after.tv_sort_window, "1W")
        self.assertGreater(after.tv_sort_start, after.lookback_start)

    def test_month_end_28_day_window_strictly_excludes_month(self):
        for value in ("2026-07-30T17:00", "2026-07-31T17:00", "2026-08-31T17:00"):
            with self.subTest(now=value):
                window = self.select(value, 28)
                self.assertEqual(window.tv_sort_window, "1W")
                self.assertGreaterEqual(window.tv_sort_start, window.lookback_start)

    def test_duplicate_future_and_missing_dates(self):
        dates = [dt.date(2026, 7, d) for d in (13, 13, 15, 17, 20)]
        window = self.select("2026-07-17T15:00", 7, dates)
        self.assertEqual(window.lookback_sessions, 2)
        self.assertEqual(window.window_sessions, 2)
        self.assertEqual(window.request_end, "2026-07-15")

    def test_zero_counts_fail_separately(self):
        with self.assertRaisesRegex(quadrant.ScreeningWindowError, "within LAST_N"):
            self.select("2026-07-17T17:00", dates=[])
        with self.assertRaisesRegex(
            quadrant.ScreeningWindowError, "eligible TradingView"
        ):
            self.select("2026-07-20T15:00", 7)

    def test_close_boundaries_are_inclusive_to_the_microsecond(self):
        for code, close in (("u", "16:00"), ("t", "13:30")):
            for delta, expected in ((-1, 2), (0, 3), (1, 2)):
                now = local_now(code, f"2026-07-17T{close}") + dt.timedelta(
                    microseconds=delta
                )
                dates = [dt.date(2026, 7, d) for d in (10, 13, 17)]
                with self.subTest(code=code, delta=delta):
                    window = quadrant.select_screening_window(
                        quadrant.MARKET_CONFIG_BY_CODE[code],
                        now,
                        now - dt.timedelta(days=7),
                        dates,
                    )
                    self.assertEqual(window.lookback_sessions, expected)

    def test_local_lookback_and_close_offsets_across_both_dst_changes(self):
        market = quadrant.MARKET_CONFIG_BY_CODE["u"]
        for before, after, elapsed_hours, old_offset, new_offset in (
            ("2026-03-06", "2026-03-13", 167, -5, -4),
            ("2026-10-30", "2026-11-06", 169, -4, -5),
        ):
            now = local_now("u", f"{after}T16:00")
            dates = weekdays(before, after)
            window = quadrant.select_screening_window(
                market, now, now - dt.timedelta(days=7), dates
            )
            with self.subTest(after=after):
                self.assertEqual(
                    window.lookback_start, local_now("u", f"{before}T16:00")
                )
                self.assertEqual(
                    (
                        now.astimezone(dt.UTC)
                        - window.lookback_start.astimezone(dt.UTC)
                    ).total_seconds()
                    / 3600,
                    elapsed_hours,
                )
                self.assertEqual(window.lookback_sessions, 6)
                self.assertEqual(
                    quadrant.trading_close_time(market, dates[0]).utcoffset(),
                    dt.timedelta(hours=old_offset),
                )
                self.assertEqual(
                    quadrant.trading_close_time(market, dates[-1]).utcoffset(),
                    dt.timedelta(hours=new_offset),
                )


class ReferenceHistoryTests(unittest.TestCase):
    def test_reference_loading_subtracts_days_in_local_time_across_dst(self):
        market = quadrant.MARKET_CONFIG_BY_CODE["u"]
        for before, after in (
            ("2026-03-06", "2026-03-13"),
            ("2026-10-30", "2026-11-06"),
        ):
            now = local_now("u", f"{after}T16:00")
            with (
                self.subTest(after=after),
                patch.object(quadrant.dt, "datetime", wraps=dt.datetime) as clock,
                patch("quadrant.get_json", return_value=[{"date": after}]),
            ):
                clock.now.return_value = now
                window = quadrant.load_screening_window(market, 7, "test-key")
            self.assertEqual(window.lookback_start, local_now("u", f"{before}T16:00"))

    def test_candidate_pipeline_uses_one_clock(self):
        for code in ("u", "t"):
            market = quadrant.MARKET_CONFIG_BY_CODE[code]
            now = local_now(code, "2026-07-17T12:00")
            reference = [
                {"date": date.isoformat()}
                for date in weekdays("2026-07-10", "2026-07-17")
            ]
            data = [
                {"date": "2026-07-16", "vwap": 5, "volume": 2},
                {"date": "2026-07-17", "vwap": 1000, "volume": 1},
            ]
            with (
                self.subTest(code=code),
                patch("quadrant.current_market_config", return_value=market),
                patch.object(quadrant, "LAST_N", 7),
                patch.object(quadrant, "RESULT_LIMIT", 2),
                patch.dict(
                    quadrant.os.environ,
                    {"FMP_API_KEY": "test-key", "FINMIND_KEY": "test-key"},
                ),
                patch.object(quadrant.dt, "datetime", wraps=dt.datetime) as clock,
                patch(
                    "quadrant.get_json",
                    return_value=reference
                    if code == "u"
                    else {"historical": reference},
                ),
                patch(
                    "quadrant.fetch_symbols_from_tv", return_value=[("ABC", "Example")]
                ) as tv,
                patch("quadrant.cached_get_json", return_value=data) as fmp,
                patch(
                    "quadrant.fetch_taiwan_stock_daily",
                    return_value=pd.DataFrame(
                        {
                            "date": ["2026-07-16", "2026-07-17"],
                            "Trading_money": [10, 1000],
                        }
                    ),
                ) as finmind,
                patch("FinMind.data.DataLoader") as loader,
            ):
                clock.now.side_effect = [now]
                candidates = quadrant.load_candidates()
            clock.now.assert_called_once_with(ZoneInfo(market.timezone))
            tv.assert_called_once_with(market=market, top_n_symbols=3, sort_window="1W")
            self.assertEqual(candidates[0].trading_dollars, 10)
            self.assertEqual(
                [c.symbol for c in candidates],
                ["ABC", "BTC"] if code == "u" else ["ABC"],
            )
            if code == "u":
                self.assertIn(("to", "2026-07-16"), fmp.call_args.kwargs["params"])
            else:
                finmind.assert_called_once_with(
                    "ABC", "2026-07-10", "2026-07-16", client=loader.return_value
                )

    def test_endpoints_parameters_freshness_and_one_local_clock_capture(self):
        for code in ("u", "t"):
            market = quadrant.MARKET_CONFIG_BY_CODE[code]
            now = local_now(code, "2026-07-17T12:00")
            rows = [
                {"date": date.isoformat()}
                for date in weekdays("2026-07-03", "2026-07-17")
            ]
            response = Mock()
            response.json.return_value = rows if code == "u" else {"historical": rows}
            with (
                self.subTest(code=code),
                patch.object(quadrant.dt, "datetime", wraps=dt.datetime) as clock,
                patch("quadrant.httpx.get", return_value=response) as get,
                patch(
                    "quadrant.cached_get_json",
                    side_effect=AssertionError("reference cached"),
                ),
            ):
                clock.now.return_value = now
                for run in range(2):
                    window = quadrant.load_screening_window(market, 14, "test-key")
                    self.assertEqual(clock.now.call_count, run + 1)
                clock.now.assert_called_with(ZoneInfo(market.timezone))
                params = {
                    "apikey": "test-key",
                    "from": "2026-07-03",
                    "to": "2026-07-17",
                }
                url = f"{quadrant.FMP_LEGACY_URL}/2330.TW"
                if code == "u":
                    url = quadrant.FMP_STABLE_URL
                    params["symbol"] = "NVDA"
                self.assertEqual(get.call_count, 2)
                get.assert_called_with(url, params=params, timeout=None)
                self.assertEqual(window.now, now)
                self.assertEqual(window.request_end, "2026-07-16")

    def test_invalid_config_and_missing_credentials_fail_before_network(self):
        with patch("quadrant.get_json") as get:
            for code in ("u", "t"):
                market = quadrant.MARKET_CONFIG_BY_CODE[code]
                with self.assertRaisesRegex(quadrant.ScreeningWindowError, "LAST_N"):
                    quadrant.load_screening_window(market, 8, "test-key")
                with self.assertRaisesRegex(
                    quadrant.ScreeningWindowError, "FMP_API_KEY"
                ):
                    quadrant.load_screening_window(market, 7, None)
            get.assert_not_called()

    def test_bad_history_and_http_failure_stop_cleanly(self):
        for payload in ({"error": "bad"}, [{"date": "bad-date"}], [{}], None):
            with (
                self.subTest(payload=payload),
                patch("quadrant.get_json", return_value=payload),
            ):
                with self.assertRaisesRegex(
                    quadrant.ScreeningWindowError, "reference history"
                ):
                    quadrant.load_screening_window(
                        quadrant.MARKET_CONFIG_BY_CODE["u"], 7, "test-key"
                    )
        with patch(
            "quadrant.get_json", side_effect=quadrant.httpx.ConnectError("unavailable")
        ):
            with self.assertRaisesRegex(
                quadrant.ScreeningWindowError, "reference history"
            ):
                quadrant.load_screening_window(
                    quadrant.MARKET_CONFIG_BY_CODE["t"], 7, "test-key"
                )

    def test_failure_produces_no_csv_and_no_candidate_requests(self):
        with (
            patch("quadrant.load_dotenv"),
            patch(
                "quadrant.load_screening_window",
                side_effect=quadrant.ScreeningWindowError(
                    "No completed trading days within LAST_N."
                ),
            ),
            patch("quadrant.fetch_symbols_from_tv") as tv,
            patch("quadrant.fetch_score") as score,
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            self.assertEqual(quadrant.main(), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Screening stopped: No completed trading days", stderr.getvalue())
        tv.assert_not_called()
        score.assert_not_called()


class CompletedHistoryCacheTests(unittest.TestCase):
    def test_completed_date_bounds_cache_and_advance_at_close(self):
        for code in ("u", "t"):
            market = quadrant.MARKET_CONFIG_BY_CODE[code]
            close = quadrant.trading_close_time(market, dt.date(2026, 7, 17))
            dates = weekdays("2026-07-03", "2026-07-17")
            before = quadrant.select_screening_window(
                market,
                close - dt.timedelta(seconds=1),
                close - dt.timedelta(days=14, seconds=1),
                dates,
            )
            after = quadrant.select_screening_window(
                market, close, close - dt.timedelta(days=14), dates
            )
            # The provider includes the unfinished row despite the request bounds.
            rows = [
                {"date": "2026-07-16", "vwap": 10, "volume": 1},
                {"date": "2026-07-17", "vwap": 100, "volume": 1},
            ]
            response = Mock()
            response.json.return_value = rows
            finmind = Mock()
            finmind.taiwan_stock_daily.return_value = pd.DataFrame(
                {"date": ["2026-07-16", "2026-07-17"], "Trading_money": [10, 100]}
            )
            with (
                self.subTest(code=code),
                tempfile.TemporaryDirectory() as path,
                quadrant.Cache(path) as cache,
                patch.object(quadrant, "cache", cache),
                patch("quadrant.httpx.get", return_value=response) as get,
            ):

                def fetch(window):
                    if code == "u":
                        return quadrant.fetch_trading_dollar_fmp(
                            market,
                            "ABC",
                            "Example",
                            "test-key",
                            window,
                        )
                    return quadrant.fetch_trading_dollar_tw(
                        market, "1234", "Example", window, client=finmind
                    )

                self.assertEqual(fetch(before).trading_dollars, 10)
                self.assertEqual(fetch(before).trading_dollars, 10)
                self.assertEqual(fetch(after).trading_dollars, 110)
                self.assertEqual(fetch(after).trading_dollars, 110)
                requests = (
                    get.call_args_list
                    if code == "u"
                    else finmind.taiwan_stock_daily.call_args_list
                )
                self.assertEqual(len(requests), 2)
                if code == "u":
                    self.assertEqual(
                        [r.kwargs["params"]["to"] for r in requests],
                        ["2026-07-16", "2026-07-17"],
                    )
                    self.assertTrue(
                        all(
                            r.kwargs["params"]["from"] == "2026-07-03" for r in requests
                        )
                    )
                else:
                    self.assertEqual(
                        [r.kwargs["end_date"] for r in requests],
                        ["2026-07-16", "2026-07-17"],
                    )
                    self.assertTrue(
                        all(r.kwargs["start_date"] == "2026-07-03" for r in requests)
                    )
                # A different lower date must also generate a separate cache entry.
                shifted = quadrant.select_screening_window(
                    market, close, close - dt.timedelta(days=7), dates
                )
                fetch(shifted)
                self.assertEqual(
                    get.call_count
                    if code == "u"
                    else finmind.taiwan_stock_daily.call_count,
                    3,
                )


if __name__ == "__main__":
    unittest.main()

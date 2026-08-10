import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np

import sp500_prices


HOLDINGS_CSV = """iShares Core S&P 500 ETF
Fund Holdings as of,"Jul 30, 2026"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Quantity
"AAA","Alpha","Technology","Equity","1,000","2.5","10"
"BRKB","Berkshire","Financials","Equity","500","1.5","5"
"AAA","Alpha","Technology","Equity","900","2.0","9"
"GOOG","Alphabet Class C","Technology","Equity","800","2.0","8"
"GOOGL","Alphabet Class A","Technology","Equity","700","1.5","7"
"CASH","Cash","-","Cash","1","0.1","1"
"""


class Sp500HoldingsTests(unittest.TestCase):
    def test_uses_the_latest_holdings_export(self):
        self.assertNotIn("asOfDate=", sp500_prices.HOLDINGS_URL)

    def test_parses_sorts_deduplicates_and_normalizes_holdings(self):
        self.assertEqual(
            sp500_prices.sp500_holdings(HOLDINGS_CSV),
            [
                sp500_prices.Holding("AAA", 4.5),
                sp500_prices.Holding("GOOG", 3.5),
                sp500_prices.Holding("BRK-B", 1.5),
            ],
        )

    def test_normalizes_blackrock_class_share_symbols(self):
        self.assertEqual(sp500_prices.normalize_symbol("BFB"), "BF-B")
        self.assertEqual(sp500_prices.normalize_symbol("BRKB"), "BRK-B")
        self.assertEqual(sp500_prices.normalize_symbol("GOOGL"), "GOOG")

    def test_rejects_pages_without_the_holdings_table(self):
        with self.assertRaisesRegex(ValueError, "CSV header"):
            sp500_prices.sp500_holdings("Ticker,Name")

    def test_reads_tickers_from_the_first_csv_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "u.16.csv"
            path.write_text("aaa,Alpha Inc.\nBRK.B,Berkshire\n", encoding="utf-8")
            self.assertEqual(
                sp500_prices.csv_holdings(path),
                [
                    sp500_prices.Holding("AAA", 0.0),
                    sp500_prices.Holding("BRK-B", 0.0),
                ],
            )

    def test_applies_sp500_weights_to_csv_holdings(self):
        input_holdings = [
            sp500_prices.Holding("BRK-B", 0.0),
            sp500_prices.Holding("MISSING", 0.0),
            sp500_prices.Holding("AAA", 0.0),
        ]
        sp500 = [
            sp500_prices.Holding("AAA", 4.5),
            sp500_prices.Holding("BRK-B", 1.5),
        ]
        self.assertEqual(
            sp500_prices.apply_sp500_weights(input_holdings, sp500),
            (
                [
                    sp500_prices.Holding("BRK-B", 1.5),
                    sp500_prices.Holding("AAA", 4.5),
                ],
                ["MISSING"],
            ),
        )

    def test_rejects_duplicate_csv_tickers_after_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tickers.csv"
            path.write_text("GOOG,Alphabet\nGOOGL,Alphabet\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate ticker 'GOOG'"):
                sp500_prices.csv_holdings(path)


class PriceFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_requests_extra_prices_and_returns_365_smas(self):
        raw_price_count = sp500_prices.PRICE_COUNT + sp500_prices.SMA_WINDOW - 1
        prices = list(range(raw_price_count))
        request = httpx.Request("GET", "https://portman.test/prices")
        response = httpx.Response(200, json=prices, request=request)

        class Client:
            async def get(self, url, params):
                self.url = url
                self.params = params
                return response

        client = Client()
        result = await sp500_prices.fetch_prices(
            cast(httpx.AsyncClient, client),
            asyncio.Semaphore(1),
            "https://portman.test/",
            "BRK-B",
        )

        self.assertEqual(
            result,
            [
                index + (sp500_prices.SMA_WINDOW - 1) / 2
                for index in range(sp500_prices.PRICE_COUNT)
            ],
        )
        self.assertEqual(client.url, "https://portman.test/prices")
        self.assertEqual(
            client.params,
            {"market": "u", "symbol": "BRK-B", "n": raw_price_count},
        )


class CorrelationMapTests(unittest.TestCase):
    def test_rejects_constant_price_histories(self):
        prices = [[100.0] * sp500_prices.PRICE_COUNT] * 2
        with self.assertRaisesRegex(ValueError, "constant price histories"):
            sp500_prices.daily_return_correlations(prices)

    def test_bubble_area_scales_with_the_square_of_price(self):
        stocks = [
            sp500_prices.Stock(
                "AAA",
                2,
                [50.0] + [100.0] * (sp500_prices.PRICE_COUNT - 1),
                0,
                0,
            )
        ]
        self.assertEqual(sp500_prices.bubble_areas(stocks, 0)[0], 0.5)
        self.assertEqual(sp500_prices.bubble_areas(stocks, 1)[0], 2)

    def test_bubble_colors_use_the_median_absolute_market_cap_change(self):
        stocks = [
            sp500_prices.Stock("AAA", 2.0, [100.0, 102.0], 0, 0),
            sp500_prices.Stock("BBB", 1.0, [100.0, 98.5], 0, 0),
            sp500_prices.Stock("CCC", 1.0, [100.0, 101.0], 0, 0),
            sp500_prices.Stock("DDD", 1.0, [100.0, 99.5], 0, 0),
            sp500_prices.Stock("EEE", 1.0, [100.0, 100.2], 0, 0),
        ]
        alpha = round(255 * sp500_prices.BUBBLE_ALPHA)
        self.assertEqual(
            sp500_prices.bubble_fills(stocks, 1),
            [
                (*sp500_prices.BUBBLE_UP_COLOR, alpha),
                (*sp500_prices.BUBBLE_DOWN_COLOR, alpha),
                (*sp500_prices.BUBBLE_FLAT_COLOR, alpha),
                (*sp500_prices.BUBBLE_FLAT_COLOR, alpha),
                (*sp500_prices.BUBBLE_FLAT_COLOR, alpha),
            ],
        )

    def test_places_perfectly_correlated_stocks_closer_together(self):
        prices = [
            [100, 101, 102.01, 103.0301, 104.060401]
            + [105.10100501] * (sp500_prices.PRICE_COUNT - 5),
            [200, 202, 204.02, 206.0602, 208.120802]
            + [210.20201002] * (sp500_prices.PRICE_COUNT - 5),
            [100, 99, 98.01, 97.0299, 96.059601]
            + [95.09900499] * (sp500_prices.PRICE_COUNT - 5),
        ]
        correlations = sp500_prices.daily_return_correlations(prices)
        coordinates = sp500_prices.correlation_coordinates(correlations)
        self.assertEqual(coordinates.shape, (3, 2))
        self.assertLess(
            np.linalg.norm(coordinates[0] - coordinates[1]),
            np.linalg.norm(coordinates[0] - coordinates[2]),
        )

    def test_writes_an_mp4(self):
        stocks = [
            sp500_prices.Stock(
                "AAA", 2.5, [100.0, 110.0, 120.0], 0.0, 1.0
            ),
            sp500_prices.Stock(
                "BBB", 1.5, [100.0, 90.0, 80.0], 1.0, 0.0
            ),
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sp500_prices, "PRICE_COUNT", 3),
        ):
            path = Path(directory) / "map.mp4"
            sp500_prices.write_video(stocks, path)
            self.assertGreater(path.stat().st_size, 0)


class FailedPriceTests(unittest.IsolatedAsyncioTestCase):
    async def test_skips_a_stock_with_unavailable_price_history(self):
        holdings = [
            sp500_prices.Holding("AAA", 3),
            sp500_prices.Holding("HONA", 2),
            sp500_prices.Holding("BBB", 1),
        ]
        prices = [float(value) for value in range(1, sp500_prices.PRICE_COUNT + 1)]

        with (
            patch.object(
                sp500_prices,
                "fetch_sp500_holdings",
                new=AsyncMock(return_value=holdings),
            ),
            patch.object(
                sp500_prices,
                "fetch_prices",
                new=AsyncMock(side_effect=[prices, ValueError("no history"), prices]),
            ),
        ):
            stocks = await sp500_prices.run(
                "http://portman.test", "https://holdings.test", 3, 1
            )

        self.assertEqual([stock.symbol for stock in stocks], ["AAA", "BBB"])

    async def test_skips_a_stock_with_a_constant_price_history(self):
        holdings = [
            sp500_prices.Holding("AAA", 3),
            sp500_prices.Holding("BBB", 1),
            sp500_prices.Holding("CCC", 1),
        ]
        changing_prices = [
            float(value) for value in range(1, sp500_prices.PRICE_COUNT + 1)
        ]
        constant_prices = [100.0] * sp500_prices.PRICE_COUNT

        with (
            patch.object(
                sp500_prices,
                "fetch_sp500_holdings",
                new=AsyncMock(return_value=holdings),
            ),
            patch.object(
                sp500_prices,
                "fetch_prices",
                new=AsyncMock(
                    side_effect=[changing_prices, constant_prices, changing_prices]
                ),
            ),
        ):
            stocks = await sp500_prices.run(
                "http://portman.test", "https://holdings.test", 3, 1
            )

        self.assertEqual([stock.symbol for stock in stocks], ["AAA", "CCC"])


class CommandLineTests(unittest.TestCase):
    def test_reads_one_optional_holdings_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tickers.csv"
            path.write_text("AAA,Alpha Inc.\n", encoding="utf-8")
            source_holdings = [sp500_prices.Holding("AAA", 0.0)]
            with (
                patch.object(
                    sp500_prices, "run", new=AsyncMock(return_value=[])
                ) as run,
                patch.object(sp500_prices, "write_video") as write_video,
            ):
                self.assertEqual(sp500_prices.main([str(path)]), 0)

        run.assert_awaited_once_with(
            sp500_prices.PORTMAN_BASE_URL,
            sp500_prices.HOLDINGS_URL,
            sp500_prices.HOLDING_COUNT,
            sp500_prices.DEFAULT_CONCURRENCY,
            source_holdings,
        )
        write_video.assert_called_once_with([], Path(sp500_prices.VIDEO_PATH))

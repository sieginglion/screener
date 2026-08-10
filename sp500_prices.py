#!/usr/bin/env python3
"""Create a daily-return correlation-map video from Portman prices.

Run with the Portman backend available at http://localhost:8080:

    uv run python sp500_prices.py

Or create a map for tickers in the first column of a headerless CSV:

    uv run python sp500_prices.py 16.csv

"""

import argparse
import asyncio
import csv
import io
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Configuration
HOLDINGS_URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/"
    "api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&"
    "targetSite=us-ishares&locale=en_US&portfolioId=239726&"
    "userType=individual&component=holdings"
)
PORTMAN_BASE_URL = "http://localhost:8080"
DEFAULT_CONCURRENCY = 8
VIDEO_PATH = "sp500-correlation-map.mp4"
K = 3_000.0
DOT_SIZE_PRICE_EXPONENT = 2.0
VIDEO_SIZE = (1920, 1440)
VIDEO_BITRATE = 12_000
LABEL_FONT_SIZE = 18
LABEL_ALPHA = 0.75
BUBBLE_ALPHA = 0.5
BUBBLE_UP_COLOR = (22, 163, 74)
BUBBLE_DOWN_COLOR = (220, 38, 38)
BUBBLE_FLAT_COLOR = (107, 114, 128)

FRAME_RATE = 16
HOLDING_COUNT = 256
SMA_WINDOW = 7
PRICE_COUNT = 92


@dataclass(frozen=True)
class Holding:
    symbol: str
    weight: float


@dataclass(frozen=True)
class Stock:
    """A holding with its price history and correlation-map position."""

    symbol: str
    weight: float
    prices: list[float]
    x: float
    y: float


def normalize_symbol(symbol: str) -> str:
    """Use FMP's dash notation for multiple share classes (for example BRK-B)."""
    symbol = symbol.replace(".", "-")
    return {
        "BFB": "BF-B",
        "BRKB": "BRK-B",
        "GOOGL": "GOOG",
    }.get(symbol, symbol)


def parse_weight(value: str) -> float:
    return float(value.replace("%", "").replace(",", "").strip())


def csv_field(row: dict[str | None, str | list[str] | None], name: str) -> str:
    value = row.get(name)
    return value if isinstance(value, str) else ""


def sp500_holdings(document: str) -> list[Holding]:
    header = "Ticker,Name,Sector,Asset Class,Market Value,Weight (%)"
    try:
        holdings_csv = document[document.index(header) :]
    except ValueError as exc:
        raise ValueError("Could not find the BlackRock holdings CSV header") from exc

    holdings: dict[str, Holding] = {}
    for row in csv.DictReader(io.StringIO(holdings_csv)):
        if csv_field(row, "Asset Class") != "Equity":
            continue
        symbol = normalize_symbol(csv_field(row, "Ticker"))
        try:
            weight = parse_weight(csv_field(row, "Weight (%)"))
        except ValueError:
            continue
        if not symbol or weight <= 0:
            continue
        holding = Holding(symbol, weight)
        if previous := holdings.get(symbol):
            holding = Holding(symbol, previous.weight + weight)
        holdings[symbol] = holding

    if not holdings:
        raise ValueError("BlackRock's holdings CSV had no equity positions")
    return sorted(holdings.values(), key=lambda holding: holding.weight, reverse=True)


def csv_holdings(path: Path) -> list[Holding]:
    """Read tickers from the first column of a headerless CSV."""
    holdings: list[Holding] = []
    seen_symbols: set[str] = set()
    with path.open(newline="", encoding="utf-8") as source:
        for line_number, fields in enumerate(csv.reader(source), start=1):
            if not fields or not any(field.strip() for field in fields):
                raise ValueError(f"{path}:{line_number}: blank rows are not supported")

            symbol = normalize_symbol(fields[0].strip().upper())
            if not symbol:
                raise ValueError(f"{path}:{line_number}: ticker is empty")
            if symbol in seen_symbols:
                raise ValueError(f"{path}:{line_number}: duplicate ticker {symbol!r}")

            seen_symbols.add(symbol)
            holdings.append(Holding(symbol, 0.0))

    if not holdings:
        raise ValueError(f"{path}: no tickers found")
    return holdings


def apply_sp500_weights(
    input_holdings: list[Holding], sp500: list[Holding]
) -> tuple[list[Holding], list[str]]:
    """Apply S&P 500 weights while preserving the input CSV's order."""
    sp500_by_symbol = {holding.symbol: holding for holding in sp500}
    weighted_holdings = []
    missing_symbols = []
    for input_holding in input_holdings:
        sp500_holding = sp500_by_symbol.get(input_holding.symbol)
        if sp500_holding is None:
            missing_symbols.append(input_holding.symbol)
            continue
        weighted_holdings.append(Holding(input_holding.symbol, sp500_holding.weight))
    return weighted_holdings, missing_symbols


async def fetch_prices(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    base_url: str,
    symbol: str,
) -> list[float]:
    raw_price_count = PRICE_COUNT + SMA_WINDOW - 1
    async with semaphore:
        response = await client.get(
            f"{base_url.rstrip('/')}/prices",
            params={"market": "u", "symbol": symbol, "n": raw_price_count},
        )
    response.raise_for_status()
    prices = response.json()
    if (
        not isinstance(prices, list)
        or len(prices) != raw_price_count
        or not all(
            isinstance(price, int | float) and not isinstance(price, bool)
            for price in prices
        )
    ):
        raise ValueError(f"Expected {raw_price_count} numeric prices, got {prices!r}")

    return (np.convolve(prices, np.ones(SMA_WINDOW), "valid") / SMA_WINDOW).tolist()


def daily_return_correlations(prices: list[list[float]]) -> np.ndarray:
    """Calculate Pearson correlations of daily returns of 7-day SMAs."""
    price_array = np.asarray(prices, dtype=float)
    if price_array.ndim != 2 or price_array.shape[1] != PRICE_COUNT:
        raise ValueError(f"Expected price rows with {PRICE_COUNT} values")

    returns = price_array[:, 1:] / price_array[:, :-1] - 1
    if not np.all(np.isfinite(returns)) or np.any(np.std(returns, axis=1) == 0.0):
        raise ValueError("Cannot correlate non-finite or constant price histories")
    correlations = np.corrcoef(returns)
    if not np.all(np.isfinite(correlations)):
        raise ValueError("Cannot correlate non-finite or constant price histories")
    np.fill_diagonal(correlations, 1.0)
    return correlations


def usable_price_history(prices: list[float]) -> bool:
    price_array = np.asarray(prices, dtype=float)
    if not np.all(np.isfinite(price_array)) or np.any(price_array <= 0):
        return False
    returns = price_array[1:] / price_array[:-1] - 1
    return np.std(returns) != 0.0


def correlation_coordinates(correlations: np.ndarray) -> np.ndarray:
    """Project a Pearson correlation matrix into two dimensions with classical MDS."""
    if correlations.ndim != 2 or correlations.shape[0] != correlations.shape[1]:
        raise ValueError("Expected a square correlation matrix")

    distance_squared = 2 * (1 - correlations)
    count = len(correlations)
    center = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * center @ distance_squared @ center
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    dimensions = np.argsort(eigenvalues)[-2:][::-1]
    return eigenvectors[:, dimensions] * np.sqrt(np.maximum(eigenvalues[dimensions], 0))


def bubble_fills(stocks: list[Stock], frame: int) -> list[tuple[int, int, int, int]]:
    """Color moves using the median absolute post-SMA market-cap change."""
    market_cap_changes = [
        stock.weight
        * (stock.prices[frame] - stock.prices[frame - 1])
        / stock.prices[-1]
        for stock in stocks
    ]
    threshold = float(np.median(np.abs(market_cap_changes)))
    alpha = round(255 * BUBBLE_ALPHA)
    fills = []
    for market_cap_change in market_cap_changes:
        if market_cap_change > threshold:
            color = BUBBLE_UP_COLOR
        elif market_cap_change < -threshold:
            color = BUBBLE_DOWN_COLOR
        else:
            color = BUBBLE_FLAT_COLOR
        fills.append((*color, alpha))
    return fills


def static_layers(stocks: list[Stock]) -> tuple[Image.Image, Image.Image, np.ndarray]:
    width, height = VIDEO_SIZE
    coordinates = np.array([(stock.x, stock.y) for stock in stocks], dtype=float)
    minimums = coordinates.min(axis=0)
    ranges = np.ptp(coordinates, axis=0)
    padding = np.maximum(ranges, 1) * 0.02
    normalized = (coordinates - minimums + padding) / (ranges + 2 * padding)

    left, top, right, bottom = 40, 65, width - 40, height - 90
    points = normalized * (right - left, bottom - top) + (left, top)
    points[:, 1] = bottom - (points[:, 1] - top)

    background = Image.new("RGBA", VIDEO_SIZE, "white")
    background_draw = ImageDraw.Draw(background)
    font = ImageFont.load_default(size=LABEL_FONT_SIZE)
    background_draw.text(
        (width / 2, 25),
        "S&P 500 daily-return correlation map",
        fill="black",
        font=font,
        anchor="mm",
    )
    background_draw.text(
        (width / 2, height - 50),
        "Nearby stocks have more-correlated daily returns",
        fill="black",
        font=font,
        anchor="mm",
    )

    labels = Image.new("RGBA", VIDEO_SIZE, (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(labels)
    for stock, (x, y) in zip(stocks, points, strict=True):
        label_draw.text(
            (x, y),
            stock.symbol,
            fill=(0, 0, 0, round(255 * LABEL_ALPHA)),
            font=font,
            anchor="mm",
        )
    return background, labels, points


def write_video(stocks: list[Stock], path: Path) -> None:
    background, labels, points = static_layers(stocks)
    width, height = background.size
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{width}x{height}",
            "-r",
            str(FRAME_RATE),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-b:v",
            f"{VIDEO_BITRATE}k",
            "-preset",
            "faster",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        assert ffmpeg.stdin is not None
        for frame in range(1, PRICE_COUNT):
            areas = np.array(
                [
                    stock.weight
                    * (stock.prices[frame] / stock.prices[-1])
                    ** DOT_SIZE_PRICE_EXPONENT
                    for stock in stocks
                ]
            ) * K
            fills = bubble_fills(stocks, frame)
            image = background.copy()
            bubbles = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(bubbles)
            for (x, y), area, fill in zip(points, areas, fills, strict=True):
                radius = float(np.sqrt(area / np.pi))
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=fill,
                )
            image = Image.alpha_composite(image, bubbles)
            image = Image.alpha_composite(image, labels)
            caption = ImageDraw.Draw(image)
            caption.text(
                (width // 2 - 260, height - 24),
                f"Day {frame} of {PRICE_COUNT - 1}",
                fill="black",
            )
            ffmpeg.stdin.write(image.tobytes())
            if frame == 1 or frame % 25 == 0 or frame == PRICE_COUNT - 1:
                sys.stderr.write(f"Rendered frame {frame} of {PRICE_COUNT - 1}\n")
    finally:
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise subprocess.CalledProcessError(ffmpeg.returncode, ffmpeg.args)


async def run(
    base_url: str,
    holdings_url: str,
    holding_count: int,
    concurrency: int,
    source_holdings: list[Holding] | None = None,
) -> list[Stock]:
    if holding_count <= 0:
        raise ValueError("holding_count must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    headers = {"User-Agent": "portman-screener/1.0"}
    async with httpx.AsyncClient(
        follow_redirects=True, headers=headers, timeout=60
    ) as client:
        response = await client.get(holdings_url)
        response.raise_for_status()
        sp500 = sp500_holdings(response.text)
        if source_holdings is None:
            holdings = sp500[:holding_count]
        else:
            holdings, missing_symbols = apply_sp500_weights(
                source_holdings[:holding_count], sp500
            )
            for symbol in missing_symbols:
                sys.stderr.write(f"Skipping {symbol}: not in the S&P 500 holdings\n")
        if source_holdings is None and len(holdings) != holding_count:
            raise ValueError(
                "BlackRock returned only "
                f"{len(holdings)} equity holdings; expected {holding_count}"
            )

        semaphore = asyncio.Semaphore(concurrency)
        results = await asyncio.gather(
            *(
                fetch_prices(client, semaphore, base_url, holding.symbol)
                for holding in holdings
            ),
            return_exceptions=True,
        )

    stock_prices = []
    for holding, prices in zip(holdings, results, strict=True):
        if isinstance(prices, Exception):
            sys.stderr.write(f"Skipping {holding.symbol}: {prices}\n")
            continue
        assert isinstance(prices, list)
        if not usable_price_history(prices):
            sys.stderr.write(f"Skipping {holding.symbol}: unusable price history\n")
            continue
        stock_prices.append((holding, prices))

    if len(stock_prices) < 2:
        raise ValueError(
            "Need at least two stocks with usable price histories to build a correlation map"
        )

    correlations = daily_return_correlations([prices for _, prices in stock_prices])
    coordinates = correlation_coordinates(correlations)
    return [
        Stock(
            holding.symbol,
            holding.weight,
            prices,
            float(x),
            float(y),
        )
        for (holding, prices), (x, y) in zip(stock_prices, coordinates, strict=True)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create an S&P 500 daily-return correlation-map video."
    )
    parser.add_argument(
        "holdings_csv",
        nargs="?",
        type=Path,
        help="headerless CSV whose first column is a ticker; uses top S&P 500 holdings if omitted",
    )
    args = parser.parse_args(argv)
    try:
        source_holdings = (
            csv_holdings(args.holdings_csv) if args.holdings_csv is not None else None
        )
        results = asyncio.run(
            run(
                PORTMAN_BASE_URL,
                HOLDINGS_URL,
                HOLDING_COUNT,
                DEFAULT_CONCURRENCY,
                source_holdings,
            )
        )
        write_video(results, Path(VIDEO_PATH))
    except (
        OSError,
        csv.Error,
        httpx.HTTPError,
        RuntimeError,
        ValueError,
        FileNotFoundError,
        subprocess.CalledProcessError,
    ) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    sys.stderr.write(f"Wrote {VIDEO_PATH}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

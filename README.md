# Screener

Screen liquid stocks, rank their economic moats with an AI panel, and compare ranked lists.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

Use `uv run` for all commands. Create an uncommitted `config.py` with the settings required by `quadrant.py`, including market: `u` (US), `t` (Taiwan), or `j` (Japan).

Add only the credentials you need to `.env`:

```dotenv
FMP_API_KEY=...       # US/Japan liquidity
FINMIND_KEY=...       # Taiwan liquidity
OPENAI_API_KEY=...    # Moat ranking panel
GOOGLE_API_KEY=...    # Or GEMINI_API_KEY
XAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Usage

Start the scoring service at `http://localhost:8080`, then generate a `ticker,description` CSV:

```sh
uv run python quadrant.py > 16.csv
```

Rank its moats with OpenAI, Gemini, xAI, and Anthropic:

```sh
uv run python moat_ranker.py < 16.csv > 16-moats.md
```

Find 16 shared top-ranked tickers from two lists:

```sh
uv run python top_k_intersection.py 16.csv 8.csv --target-count 16 -o moat.csv
```

Clear cached data when needed:

```sh
uv run python clear_ticker_cache.py AAPL
uv run python clear_ticker_cache.py --portman
uv run python clear_ticker_cache.py --liquidity
```

## Checks

```sh
uv run --locked python -m unittest discover -v
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
```

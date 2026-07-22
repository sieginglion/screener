# Screener

Scripts for finding liquid, high-scoring stocks, ranking their economic moats
with an AI model panel, and comparing ranked ticker lists.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) to manage dependencies and
the local virtual environment. Python 3.12 is selected automatically.

```sh
uv sync
```

This creates a local `.venv/`; use `uv run` rather than activating it
manually.

The screener reads local configuration from `config.py`, which is intentionally
not committed. It must define the settings imported by `quadrant.py`, including
the market (`u` for US, `t` for Taiwan, or `j` for Japan) and ranking limits.

Create a `.env` file for the credentials relevant to the commands you use:

```dotenv
# Required for US and Japanese liquidity data
FMP_API_KEY=...

# Required for Taiwanese liquidity data
FINMIND_KEY=...

# Required by moat_ranker.py
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
XAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

`GEMINI_API_KEY` may be used instead of `GOOGLE_API_KEY`.

## Commands

### Generate a ranked stock list

Start the local scoring service expected at `http://localhost:8080`, then run:

```sh
./run.sh > 16.csv
# Equivalent: uv run python quadrant.py > 16.csv
```

The command writes `ticker,description` CSV records to standard output and
caches network responses in `.cache/`.

### Rank moats with the model panel

Provide one `ticker,description` CSV record per line on standard input:

```sh
uv run python moat_ranker.py < 16.csv > 16-moats.md
```

The panel uses OpenAI, Google Gemini, xAI, and Anthropic. Use `--help` to see
the optional model override flags.

### Find the shared top-ranked tickers

Find the smallest equal-size prefix of two ranked CSV files that has exactly
16 tickers in common, then write the matching rows from the first file:

```sh
uv run python top_k_intersection.py 16.csv 8.csv --target-count 16 -o moat.csv
```

The defaults are `16.csv`, `8.csv`, `moat.csv`, and a target count of 16.

### Clear cached data

```sh
uv run python clear_ticker_cache.py AAPL
uv run python clear_ticker_cache.py --portman
uv run python clear_ticker_cache.py --liquidity
```

## Tests

```sh
uv run --locked python -m unittest discover -v
```

## Code quality

Check linting, formatting, and types with:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
```

Apply Ruff's safe lint fixes and formatter with:

```sh
uv run ruff check --fix .
uv run ruff format .
```

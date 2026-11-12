# Screener

Screen liquid stocks, rank their economic moats or pricing power with an AI panel, and compare ranked lists.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

Use `uv run` for commands; `./ranker.py` invokes it automatically. Create an uncommitted `config.py` with the settings required by `quadrant.py`, including market: `u` (US) or `t` (Taiwan).

Add only the credentials you need to `.env`:

```dotenv
FMP_API_KEY=...       # US/Taiwan reference history; US liquidity
FINMIND_KEY=...       # Taiwan liquidity
OPENAI_API_KEY=...    # Ranking panel
GOOGLE_API_KEY=...    # Or GEMINI_API_KEY
XAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Usage

Start the scoring service at `http://localhost:8080`, then generate a `ticker,description` CSV:

```sh
uv run python quadrant.py > 16.csv
```

Set `LAST_N` to `7`, `14`, or `28`. The screener captures the
market-local time once and subtracts `LAST_N` days, keeping the same local clock
time across daylight saving changes. It fetches fresh FMP reference dates
(NVDA for US, legacy 2330.TW for Taiwan) and counts completed trading days whose
local close falls within the inclusive lookback boundaries: 16:00 New York or
13:30 Taipei.

TradingView's `1W` starts Monday at local midnight; `1M` starts on the first.
Only windows starting at or after the lookback boundary are eligible, even at
month-end with `LAST_N = 28`. The window covering the most completed reference
trading days wins; ties prefer `1W`. The candidate-pool multiplier is
`lookback_sessions / window_sessions * 1.1`, and
the pool size is `ceil(RESULT_LIMIT * multiplier)`.

US FMP and Taiwan FinMind liquidity use the same inclusive close-time
boundaries. Their 12-hour cached requests end at the latest completed reference
date, so in-progress sessions are not requested. Reference history is
never cached. Missing FMP credentials, unusable reference history, or zero
completed trading days in either count stop screening with an error and no CSV.
Taiwan also requires `FINMIND_KEY` for liquidity.

The screener ranks revenue and EPS growth separately, averages their available
ranks, and requests PS and PE scores only for the best-growth half (rounded up).
It combines PS and PE ranks the same way and keeps the best-valuation half
(rounded up), breaking cutoff ties alphabetically by ticker. It then reranks
growth and valuation among the survivors and adds those ranks with equal weight.
Higher growth and higher valuation scores receive lower numeric ranks (1 is
best); lower valuation means a higher score. Score values from the service are
used without validation. The
`/scores` request passes `market`, `symbol`, and `q`. Candidates missing revenue
growth or the PS score are omitted before ranking the corresponding group and
applying its cutoff. If EPS growth or the PE score is missing, the first metric's
rank is used alone. Failed scoring requests are also omitted, without backfilling.

Create an S&P 500 correlation-map video using BlackRock's latest S&P 500
holdings export, or for tickers in the first column of a headerless CSV (bubble
sizes use S&P 500 weights):

```sh
uv run python sp500_prices.py
uv run python sp500_prices.py 16.csv
```

Rank its moats or pricing power with OpenAI, Gemini, xAI, and Anthropic.
Choose `-p` for pricing power (ability of raising prices) or `-m` for moat
(durability of profitability); one is required:

```sh
./ranker.py -m < 16.csv > 16-moats.md
./ranker.py -p < 16.csv > 16-pricing.md
```

Each batch of up to four stocks receives two panel rounds. The second asks
`think again`, with all four first-round answers tagged by model in shared
assistant history. Revised answers are concatenated by model with newline
separators into four `<context model="…">` blocks for the final ranking panel.
GPT performs the only merge, with all four tagged rankings as assistant history
and the merge instruction as the next user message.

Add `--grok-priority` to request priority processing for Grok at xAI's premium
token pricing. Standard processing is the default:

```sh
./ranker.py -p --grok-priority < 16.csv > 16-pricing.md
```

Find 16 shared top-ranked tickers from two lists:

```sh
uv run python top_k_intersection.py 16.csv 8.csv --target-count 16 -o moat.csv
```

Average the original row ranks of two lists, keeping only tickers present in
both. Output includes the ticker and name, sorted from lowest average rank
to highest (ties use ticker order), without a header, on stdout. Save it with
shell redirection:

```sh
uv run python average_rankings.py ./t.08.csv ./t.16.csv > averaged.csv
```

Combine ranked CSV files with raw linear reverse-rank scores. Every row is
used; in a list of length ``n``, scores decline from ``n`` for the first row to
``1`` for the last:

```sh
uv run python aggregate_rankings.py -o aggregated.csv
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

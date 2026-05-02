# Quadrant Screener Spec

## Goal

Build `quadrant.py`, a command-line screener that finds liquid securities in the high-growth, low-valuation quadrant and writes the ranked result set as CSV to standard output.

The strategy is fixed:

- Prefer higher growth scores.
- Prefer lower valuation scores.
- Rank final survivors by a power score.

## Allowed Dependencies

`quadrant.py` may use:

- Python standard library modules.
- Third-party runtime dependencies already used by the project, including `httpx` and `python-dotenv`.
- `config.py` for configuration.
- `run_u_analyze.py` for candidate universe construction and cached HTTP requests.

The script must not depend on any other project files.

## Configuration

Read these values from `config.py`:

- `MARKET`
- `RESULT_LIMIT`
- `CANDIDATE_POOL_MULTIPLIER`
- `PEAK_CUTOFF_RATIO`
- `Q`

Environment variables must be loaded with `load_dotenv()` before building the candidate universe.

`quadrant.py` must not validate configuration values, candidate rows, scoring response shapes, cutoff counts, or computed scores. It should use the configured and returned values directly; natural exceptions from failed operations may propagate unless this spec explicitly says to skip the candidate.

## Candidate Universe

Use the existing liquid candidate universe builder from `run_u_analyze.py`.

The candidate pool size is:

```text
ceil(RESULT_LIMIT * CANDIDATE_POOL_MULTIPLIER)
```

Do not validate `RESULT_LIMIT`, `CANDIDATE_POOL_MULTIPLIER`, or the computed candidate pool size.

The script must call:

```python
load_top_company_names(
    api_key=os.environ.get("FMP_API_KEY"),
    top_n_symbols=pool_size,
    top_n_results=RESULT_LIMIT,
)
```

The calculated `pool_size` is the number of symbols to fetch from TradingView before liquidity filtering.

`load_top_company_names` must then fetch individual trading volume data for those symbols and return only the most liquid `RESULT_LIMIT` candidates.

Candidate loading errors must fail fast.

Each returned row is expected to be a string in this form:

```text
<symbol> <description>
```

Convert each row into a candidate containing only:

- `symbol`
- `description`

Candidates must not store market.

## Bitcoin Candidate

When `MARKET == "u"`, append Bitcoin as an extra candidate:

```text
symbol: BTC
description: Bitcoin
```

Do not append Bitcoin for any other market.

Bitcoin is added after loading the most liquid `RESULT_LIMIT` candidates, so it may increase the number of candidates entering growth scoring by one.

## Scoring Service

Scores come from a local HTTP service on `localhost:8080`.

For scoring requests, select market at request time:

- `BTC` uses market `c`.
- Every other symbol uses the configured `MARKET`.

Growth endpoint:

```text
GET http://localhost:8080/growths?market=<market>&symbol=<symbol>
```

Valuation endpoint:

```text
GET http://localhost:8080/scores?market=<market>&symbol=<symbol>&q=<Q>
```

Both endpoints return JSON arrays. Use the first array element as the score and ignore the rest.

## Scoring Errors

Candidate universe loading errors must fail fast.

Scoring service errors must not fail the whole run:

- If a `/growths` request fails, skip that candidate for the growth ranking.
- If a `/scores` request fails, skip that candidate for the valuation ranking.

Skipped scoring responses must not reduce the stage cutoff count. Each cutoff count is based on the number of candidates attempted at that stage, not the number of successful scoring responses.

## Screening Logic

Filtering has two stages.

### Growth Stage

Attempt to score every loaded candidate with `/growths`.

Let `growth_attempt_count` be the number of candidates that entered the growth stage, including Bitcoin when it was appended.

Sort successful growth-scored candidates by `growth_score` from highest to lowest.

Keep:

```text
ceil(growth_attempt_count * PEAK_CUTOFF_RATIO)
```

If fewer successful growth-scored candidates exist than the keep count, keep all successful growth-scored candidates.

### Valuation Stage

Attempt to score every growth survivor with `/scores`.

Let `valuation_attempt_count` be the number of candidates that entered the valuation stage.

Sort successful valuation-scored candidates by `valuation_score` from lowest to highest.

Keep:

```text
ceil(valuation_attempt_count * PEAK_CUTOFF_RATIO)
```

If fewer successful valuation-scored candidates exist than the keep count, keep all successful valuation-scored candidates.

## Power Score

Each final survivor receives:

```text
power = growth_score * (0.625 - valuation_score)
```

Final survivors are sorted by `power` from highest to lowest.

## Output

Write CSV to standard output.

The header must be exactly:

```csv
symbol,description,power
```

Each row must contain:

- Candidate symbol.
- Candidate description.
- Power formatted with six significant digits.

No other data should be written to standard output.

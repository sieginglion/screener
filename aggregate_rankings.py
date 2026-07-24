#!/usr/bin/env python3
"""Aggregate ranked ticker CSV files using normalized reverse-rank scores.

For a file containing ``n`` rows, the first row receives ``n / (n * (n + 1) / 2)``
and the final row receives ``1 / (n * (n + 1) / 2)``.  Thus every input file
contributes a total score of one.  Tickers appearing in multiple files have
their scores summed and are written in descending total-score order.
"""

import argparse
import csv
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Sequence


def read_ranked_csv(path: Path) -> list[tuple[str, str]]:
    """Return ``(ticker, description)`` rows, rejecting invalid input."""
    rows: list[tuple[str, str]] = []
    seen_tickers: set[str] = set()

    with path.open(newline="", encoding="utf-8") as source:
        for line_number, fields in enumerate(csv.reader(source), start=1):
            if not fields or not any(field.strip() for field in fields):
                raise ValueError(f"{path}:{line_number}: blank rows are not supported")

            ticker = fields[0].strip().upper()
            if not ticker:
                raise ValueError(f"{path}:{line_number}: ticker is empty")
            if ticker in seen_tickers:
                raise ValueError(f"{path}:{line_number}: duplicate ticker {ticker!r}")

            seen_tickers.add(ticker)
            description = fields[1].strip() if len(fields) > 1 else ""
            rows.append((ticker, description))

    if not rows:
        raise ValueError(f"{path}: contains no rows")
    return rows


def aggregate(paths: Sequence[Path]) -> list[tuple[str, str, Fraction]]:
    """Score every input independently and return combined ticker totals."""
    scores: defaultdict[str, Fraction] = defaultdict(Fraction)
    descriptions: dict[str, str] = {}

    for path in paths:
        rows = read_ranked_csv(path)
        row_count = len(rows)
        denominator = row_count * (row_count + 1) // 2

        for index, (ticker, description) in enumerate(rows):
            scores[ticker] += Fraction(row_count - index, denominator)
            if ticker not in descriptions or not descriptions[ticker]:
                descriptions[ticker] = description

    return sorted(
        ((ticker, descriptions[ticker], score) for ticker, score in scores.items()),
        key=lambda row: (-row[2], row[0]),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="ranked CSV files (defaults to every CSV in the current directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("aggregated.csv"),
        help="destination CSV (default: aggregated.csv)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    input_paths = args.inputs or sorted(Path.cwd().glob("*.csv"))
    input_paths = [path for path in input_paths if path.resolve() != output]

    if not input_paths:
        print("error: no input CSV files found", file=sys.stderr)
        return 1

    try:
        aggregated = aggregate(input_paths)
        with args.output.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerows(
                (ticker, description, f"{float(score):.12f}")
                for ticker, description, score in aggregated
            )
    except (OSError, ValueError, csv.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"aggregated {len(input_paths)} files into {len(aggregated)} tickers: "
        f"{args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

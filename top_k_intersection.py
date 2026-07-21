#!/usr/bin/env python3
"""Write the ticker intersection of the top rows of two ranked CSV files.

By default, find the smallest k where the first k rows of ``16.csv`` and
``8.csv`` share 16 tickers, then write those records from ``16.csv`` to
``moat.csv``.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Record:
    """A CSV row identified by its ticker in the first column."""

    ticker: str
    fields: tuple[str, ...]


def read_records(path: Path) -> list[Record]:
    """Read records from *path*, rejecting blank or duplicate tickers."""
    records: list[Record] = []
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
            records.append(Record(ticker=ticker, fields=tuple(fields)))

    return records


def find_first_k(
    first: Sequence[Record], second: Sequence[Record], target_count: int
) -> int:
    """Return the smallest shared prefix length with ``target_count`` tickers."""
    if target_count < 0:
        raise ValueError("target count must not be negative")
    if target_count == 0:
        return 0

    first_seen: set[str] = set()
    second_seen: set[str] = set()
    shared: set[str] = set()

    for index, (first_record, second_record) in enumerate(zip(first, second), start=1):
        first_seen.add(first_record.ticker)
        if first_record.ticker in second_seen:
            shared.add(first_record.ticker)

        second_seen.add(second_record.ticker)
        if second_record.ticker in first_seen:
            shared.add(second_record.ticker)

        if len(shared) == target_count:
            return index

    raise ValueError(
        f"no k up to {min(len(first), len(second))} has exactly "
        f"{target_count} shared tickers"
    )


def intersect_top_k(
    first: Sequence[Record], second: Sequence[Record], k: int
) -> list[Record]:
    """Return first-list records shared by both top-``k`` prefixes, in rank order."""
    if k < 0:
        raise ValueError("k must not be negative")

    second_tickers = {record.ticker for record in second[:k]}
    return [record for record in first[:k] if record.ticker in second_tickers]


def positive_integer(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", nargs="?", type=Path, default=Path("16.csv"))
    parser.add_argument("second", nargs="?", type=Path, default=Path("8.csv"))
    parser.add_argument("-o", "--output", type=Path, default=Path("moat.csv"))
    parser.add_argument(
        "--target-count",
        type=positive_integer,
        default=16,
        help="shared-ticker count used when finding k (default: 16)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        first = read_records(args.first)
        second = read_records(args.second)
        k = find_first_k(first, second, args.target_count)
        intersection = intersect_top_k(first, second, k)

        with args.output.open("w", newline="", encoding="utf-8") as destination:
            csv.writer(destination).writerows(record.fields for record in intersection)
    except (OSError, ValueError, csv.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"k={k}; wrote {len(intersection)} shared tickers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

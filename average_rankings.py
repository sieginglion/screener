#!/usr/bin/env python3
"""Average row-position ranks for tickers present in both headerless CSV files."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

from top_k_intersection import read_records


def average_rankings(first: Path, second: Path) -> list[tuple[str, ...]]:
    """Keep shared tickers, sorted by mean original rank, then ticker for ties."""
    first_records = read_records(first)
    second_ranks = {
        record.ticker: rank for rank, record in enumerate(read_records(second), start=1)
    }
    # Sorting by rank sums is equivalent to sorting by their averages.
    shared = [
        (rank + second_ranks[record.ticker], record)
        for rank, record in enumerate(first_records, start=1)
        if record.ticker in second_ranks
    ]
    shared.sort(key=lambda item: (item[0], item[1].ticker))
    return [record.fields for _, record in shared]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="first ranked CSV (rank 1 first)")
    parser.add_argument("second", type=Path, help="second ranked CSV (rank 1 first)")
    args = parser.parse_args(argv)
    try:
        rows = average_rankings(args.first, args.second)
        csv.writer(sys.stdout).writerows(rows)
    except (OSError, ValueError, csv.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import csv
import sys

from dotenv import load_dotenv

from peak import fetch_growth, fetch_score

TARGET_SCORE = 0.625


def first_number(values: object, label: str) -> float:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} response missing first value")

    value = values[0]
    if value is None:
        raise ValueError(f"{label} first value is null")

    return float(value)


def read_stocks() -> list[tuple[str, list[str]]]:
    stocks: list[tuple[str, list[str]]] = []
    reader = csv.reader(sys.stdin)

    for row in reader:
        if not row:
            continue

        symbol = row[0].strip()
        if not symbol:
            continue

        stocks.append((symbol, row[1:]))

    return stocks


def main() -> int:
    load_dotenv()
    stocks = read_stocks()

    if not stocks:
        return 0

    sys.stderr.write(f"Fetching growths and scores for {len(stocks)} stocks...\n")
    results: list[tuple[str, list[str], float]] = []

    for symbol, rest in stocks:
        try:
            growth = first_number(fetch_growth(symbol), "growths")
            score = first_number(fetch_score(symbol), "scores")
            value = growth * (TARGET_SCORE - score)
        except Exception as exc:
            sys.stderr.write(f"Skipping {symbol}: {type(exc).__name__}: {exc}\n")
        else:
            results.append((symbol, rest, value))

    results.sort(key=lambda item: item[2], reverse=True)

    writer = csv.writer(sys.stdout)
    for symbol, rest, value in results:
        writer.writerow([symbol, *rest, value])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

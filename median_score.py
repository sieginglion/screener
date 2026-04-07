#!/usr/bin/env python3
import sys


def load_scores() -> list[float]:
    scores: list[float] = []

    for line_number, raw_line in enumerate(sys.stdin, start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split(";")
        if len(parts) < 2:
            raise ValueError(
                f"stdin:{line_number}: expected at least 2 semicolon-delimited fields"
            )

        try:
            scores.append(float(parts[1]))
        except ValueError as exc:
            raise ValueError(
                f"stdin:{line_number}: invalid score value {parts[1]!r}"
            ) from exc

    if not scores:
        raise ValueError("stdin: no scores found")

    return scores


def median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2

    if count % 2 == 1:
        return ordered[middle]

    return (ordered[middle - 1] + ordered[middle]) / 2.0


def main() -> int:
    try:
        print(f"{median(load_scores()):.6f}")
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

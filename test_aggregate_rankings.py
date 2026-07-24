import csv
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from aggregate_rankings import aggregate, main


class AggregateRankingsTests(unittest.TestCase):
    def write_csv(
        self, directory: Path, name: str, rows: list[tuple[str, str]]
    ) -> Path:
        path = directory / name
        with path.open("w", newline="", encoding="utf-8") as destination:
            csv.writer(destination).writerows(rows)
        return path

    def test_uses_normalized_reverse_rank_and_sorts_totals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first = self.write_csv(
                directory,
                "first.csv",
                [("NVDA", "NVIDIA"), ("MSFT", "Microsoft"), ("AAPL", "Apple")],
            )
            second = self.write_csv(
                directory,
                "second.csv",
                [("MSFT", "Microsoft"), ("NVDA", "NVIDIA")],
            )

            result = aggregate([first, second])

        self.assertEqual(
            result,
            [
                ("MSFT", "Microsoft", Fraction(1)),
                ("NVDA", "NVIDIA", Fraction(5, 6)),
                ("AAPL", "Apple", Fraction(1, 6)),
            ],
        )

    def test_writes_a_sorted_csv_without_a_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = self.write_csv(
                directory,
                "source.csv",
                [("NVDA", "NVIDIA"), ("MSFT", "Microsoft")],
            )
            output = directory / "result.csv"

            self.assertEqual(main([str(source), "-o", str(output)]), 0)

            with output.open(newline="", encoding="utf-8") as result:
                self.assertEqual(
                    list(csv.reader(result)),
                    [
                        ["NVDA", "NVIDIA", "0.666666666667"],
                        ["MSFT", "Microsoft", "0.333333333333"],
                    ],
                )


if __name__ == "__main__":
    unittest.main()

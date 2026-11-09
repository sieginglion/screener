import csv
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from average_rankings import average_rankings, main


class AverageRankingsTests(unittest.TestCase):
    def test_averages_original_positions_and_drops_missing_tickers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first = directory / "first.csv"
            second = directory / "second.csv"
            first.write_text("A,Alpha\nX,Excluded\nB,Beta\nC,Gamma\n", encoding="utf-8")
            second.write_text(
                "B,Beta\nC,Gamma\nY,Excluded\nZ,Excluded\nA,Alpha\n", encoding="utf-8"
            )

            # B averages 2; A and C average 3, breaking ties by ticker.
            self.assertEqual(
                average_rankings(first, second),
                [("B", "Beta"), ("A", "Alpha"), ("C", "Gamma")],
            )
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main([str(first), str(second)]), 0)
            self.assertEqual(
                list(csv.reader(StringIO(stdout.getvalue()))),
                [["B", "Beta"], ["A", "Alpha"], ["C", "Gamma"]],
            )

    def test_no_shared_tickers_produces_empty_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first = directory / "first.csv"
            second = directory / "second.csv"
            first.write_text("A,Alpha\n", encoding="utf-8")
            second.write_text("B,Beta\n", encoding="utf-8")
            self.assertEqual(average_rankings(first, second), [])


if __name__ == "__main__":
    unittest.main()

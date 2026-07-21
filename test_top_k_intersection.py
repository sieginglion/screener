import csv
import tempfile
import unittest
from pathlib import Path

import top_k_intersection


class TopKIntersectionTests(unittest.TestCase):
    def test_finds_smallest_k_and_matches_by_ticker(self):
        first = [
            top_k_intersection.Record("AAA", ("AAA", "Alpha")),
            top_k_intersection.Record("BSX", ("BSX", "Boston Scientific")),
            top_k_intersection.Record("CCC", ("CCC", "Charlie")),
        ]
        second = [
            top_k_intersection.Record("BSX", ("BSX", "Boston Scientific")),
            top_k_intersection.Record("AAA", ("AAA", "Alpha, Inc.")),
            top_k_intersection.Record("DDD", ("DDD", "Delta")),
        ]

        k = top_k_intersection.find_first_k(first, second, target_count=2)
        intersection = top_k_intersection.intersect_top_k(first, second, k)

        self.assertEqual(k, 2)
        self.assertEqual([record.ticker for record in intersection], ["AAA", "BSX"])

    def test_cli_writes_first_file_rows_in_rank_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.csv"
            second_path = root / "second.csv"
            output_path = root / "moat.csv"
            first_path.write_text(
                "AAA,Alpha\nBSX,Boston Scientific\nCCC,Charlie\n", encoding="utf-8"
            )
            second_path.write_text(
                'BSX,"Boston Scientific"\nAAA,"Alpha, Inc."\nDDD,Delta\n',
                encoding="utf-8",
            )

            result = top_k_intersection.main(
                [
                    str(first_path),
                    str(second_path),
                    "--target-count",
                    "2",
                    "--output",
                    str(output_path),
                ]
            )

            with output_path.open(newline="", encoding="utf-8") as output:
                rows = list(csv.reader(output))

        self.assertEqual(result, 0)
        self.assertEqual(rows, [["AAA", "Alpha"], ["BSX", "Boston Scientific"]])

    def test_read_records_rejects_duplicate_tickers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            path.write_text("AAA,Alpha\naaa,Alpha again\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate ticker 'AAA'"):
                top_k_intersection.read_records(path)


if __name__ == "__main__":
    unittest.main()

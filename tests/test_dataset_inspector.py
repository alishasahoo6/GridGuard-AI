"""Unit tests for dataset inspection, date parsing, and malformed row detection."""

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.dataset_inspector import (
    find_calendar_gaps,
    inspect_dataset,
    inspect_malformed_rows,
    parse_and_sort_date_columns,
)


class TestDatasetInspector(unittest.TestCase):
    """Test suite for dataset inspection and date parsing functions."""

    def test_parse_and_sort_date_columns_chronological(self) -> None:
        """Verify that date columns are sorted strictly by date, not lexical string order."""
        # Lexical order: '2014/1/1', '2014/1/10', '2014/1/2'
        cols = ["CONS_NO", "FLAG", "2014/1/10", "2014/1/1", "2014/1/2"]
        sorted_dates, non_dates, fmt = parse_and_sort_date_columns(cols)

        self.assertEqual(non_dates, ["CONS_NO", "FLAG"])
        self.assertEqual(len(sorted_dates), 3)
        self.assertEqual(sorted_dates[0][1], "2014/1/1")
        self.assertEqual(sorted_dates[1][1], "2014/1/2")
        self.assertEqual(sorted_dates[2][1], "2014/1/10")
        self.assertEqual(sorted_dates[0][0], datetime(2014, 1, 1))

    def test_parse_day_month_year_format(self) -> None:
        """Verify parsing of European/international DD-MM-YY date headers."""
        cols = ["CONS_NO", "03-01-15", "01-01-15", "02-01-15", "CHK_STATE"]
        sorted_dates, non_dates, fmt = parse_and_sort_date_columns(cols)

        self.assertEqual(non_dates, ["CONS_NO", "CHK_STATE"])
        self.assertEqual(len(sorted_dates), 3)
        self.assertEqual(sorted_dates[0][1], "01-01-15")
        self.assertEqual(sorted_dates[1][1], "02-01-15")
        self.assertEqual(sorted_dates[2][1], "03-01-15")

    def test_find_calendar_gaps(self) -> None:
        """Verify detection of missing dates in a chronological sequence."""
        # Sequence missing 2015-01-02
        dates = [
            (datetime(2015, 1, 1), "01-01-15"),
            (datetime(2015, 1, 3), "03-01-15"),
            (datetime(2015, 1, 4), "04-01-15"),
        ]
        expected_days, missing = find_calendar_gaps(dates)
        self.assertEqual(expected_days, 4)
        self.assertEqual(missing, ["2015-01-02"])

    def test_inspect_malformed_sequential_row(self) -> None:
        """Verify detection of malformed row containing missing ID and sequential indices."""
        data = {
            "CONS_NO": [np.nan, "C101"],
            "D1": [1.0, 5.2],
            "D2": [2.0, 3.4],
            "D3": [3.0, 4.1],
            "D4": [4.0, 6.0],
            "D5": [5.0, 7.2],
            "CHK_STATE": [np.nan, 0.0],
        }
        df = pd.DataFrame(data)
        date_cols = ["D1", "D2", "D3", "D4", "D5"]
        malformed = inspect_malformed_rows(df, "CONS_NO", date_cols)

        self.assertEqual(len(malformed), 1)
        self.assertEqual(malformed[0]["row_index"], 0)
        self.assertTrue(malformed[0]["is_missing_id"])
        self.assertTrue(malformed[0]["is_sequential_index_row"])

    def test_inspect_dataset_on_fixture(self) -> None:
        """Verify end-to-end inspection on a controlled temporary CSV fixture."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_csv = Path(tmp_dir) / "test_data.csv"
            df = pd.DataFrame({
                "CONS_NO": ["C01", "C02", "C03"],
                "FLAG": [0, 1, 0],
                "2015/1/1": [10.5, 0.0, np.nan],
                "2015/1/2": [12.0, -1.5, 8.0],
            })
            df.to_csv(tmp_csv, index=False)

            result = inspect_dataset(tmp_csv, dataset_name="test_data.csv")
            self.assertEqual(result.num_rows, 3)
            self.assertEqual(result.num_columns, 4)
            self.assertEqual(result.num_unique_consumers, 3)
            self.assertEqual(result.date_columns_count, 2)
            self.assertEqual(result.label_distribution.get("0"), 2)
            self.assertEqual(result.label_distribution.get("1"), 1)
            self.assertEqual(result.negative_consumption_cells, 1)
            self.assertEqual(result.zero_consumption_cells, 1)
            self.assertEqual(result.missing_consumption_cells, 1)


if __name__ == "__main__":
    unittest.main()

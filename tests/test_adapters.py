"""Unit and integration tests for dataset adapters, transformations, and hygiene."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from config.settings import RAW_DATA_DIR
from src.data.adapters import (
    DataCsvAdapter,
    ElectricityTheftAdapter,
)
from src.data.schemas import QualityFlag


class TestDatasetAdapters(unittest.TestCase):
    """Test suite for wide-to-long transformation, label extraction, and quarantine logic."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_data_csv_adapter_label_extraction_and_stats(self) -> None:
        """Verify label extraction, provenance, and summary statistics on synthetic fixture."""
        fixture_path = self.tmp_path / "data_fixture.csv"
        df = pd.DataFrame({
            "CONS_NO": ["HASH001", "HASH002"],
            "FLAG": [0, 1],
            "2014/1/10": [10.0, np.nan],
            "2014/1/1": [5.0, 0.0],
            "2014/1/2": [np.nan, 20.0],
        })
        df.to_csv(fixture_path, index=False)

        adapter = DataCsvAdapter(file_path=fixture_path)
        labels_df, quarantined_df = adapter.extract_consumer_labels()

        self.assertEqual(len(labels_df), 2)
        self.assertTrue(quarantined_df.empty)

        # Check provenance
        self.assertEqual(labels_df.iloc[0]["source_dataset"], "data.csv")
        self.assertEqual(labels_df.iloc[0]["source_label_column"], "FLAG")
        self.assertEqual(labels_df.iloc[0]["source_label"], 0.0)
        self.assertEqual(labels_df.iloc[1]["source_label"], 1.0)

        # Check chronological ordering was applied (2014/1/1, 2014/1/2, 2014/1/10)
        self.assertEqual(adapter.sorted_date_cols[0][1], "2014/1/1")
        self.assertEqual(adapter.sorted_date_cols[1][1], "2014/1/2")
        self.assertEqual(adapter.sorted_date_cols[2][1], "2014/1/10")

        # Check consumption metrics
        self.assertEqual(labels_df.iloc[0]["recorded_days"], 2)
        self.assertEqual(labels_df.iloc[0]["missing_days"], 1)
        self.assertEqual(labels_df.iloc[1]["zero_days"], 1)

    def test_electricity_theft_adapter_quarantines_malformed_row(self) -> None:
        """Verify that Row 0 (missing ID and sequential integers) is quarantined."""
        fixture_path = self.tmp_path / "theft_fixture.csv"
        df = pd.DataFrame({
            "CONS_NO": [np.nan, "100234567"],
            "01-01-15": [1.0, 4.5],
            "02-01-15": [2.0, -2.0],
            "03-01-15": [3.0, 0.0],
            "CHK_STATE": [np.nan, 1.0],
        })
        df.to_csv(fixture_path, index=False)

        adapter = ElectricityTheftAdapter(file_path=fixture_path)
        labels_df, quarantined_df = adapter.extract_consumer_labels()

        # Row 0 must be quarantined
        self.assertEqual(len(quarantined_df), 1)
        self.assertIn("MALFORMED_HEADER_INDEX_ROW", quarantined_df.iloc[0]["quarantine_reason"])

        # Row 1 must be retained
        self.assertEqual(len(labels_df), 1)
        self.assertEqual(labels_df.iloc[0]["consumer_id"], "100234567")
        self.assertEqual(labels_df.iloc[0]["source_label_column"], "CHK_STATE")
        self.assertEqual(labels_df.iloc[0]["source_label"], 1.0)
        self.assertEqual(labels_df.iloc[0]["negative_days"], 1)
        self.assertEqual(labels_df.iloc[0]["zero_days"], 1)

    def test_stream_standardized_long_format_schema_and_flags(self) -> None:
        """Verify wide-to-long transformation preserves schema, nulls, and flags."""
        fixture_path = self.tmp_path / "long_fixture.csv"
        df = pd.DataFrame({
            "CONS_NO": ["C101"],
            "FLAG": [1],
            "2014/1/1": [0.0],
            "2014/1/2": [-5.0],
            "2014/1/3": [np.nan],
            "2014/1/4": [1500.0],
            "2014/1/5": [12.5],
        })
        df.to_csv(fixture_path, index=False)

        adapter = DataCsvAdapter(file_path=fixture_path)
        adapter.load_and_inspect_header()

        chunks = list(adapter.stream_standardized_long_format(chunk_size=10))
        long_df = pd.concat(chunks, ignore_index=True)

        self.assertEqual(len(long_df), 5)
        expected_cols = [
            "consumer_id",
            "timestamp",
            "consumption_value",
            "source_dataset",
            "source_label",
            "source_label_column",
            "data_quality_flags",
        ]
        self.assertEqual(list(long_df.columns), expected_cols)

        # 2014-01-01 -> Zero reading
        row_zero = long_df[long_df["timestamp"] == "2014-01-01"].iloc[0]
        self.assertEqual(row_zero["consumption_value"], 0.0)
        self.assertEqual(row_zero["data_quality_flags"], QualityFlag.ZERO_READING.value)

        # 2014-01-02 -> Negative reading
        row_neg = long_df[long_df["timestamp"] == "2014-01-02"].iloc[0]
        self.assertEqual(row_neg["consumption_value"], -5.0)
        self.assertEqual(row_neg["data_quality_flags"], QualityFlag.NEGATIVE_READING.value)

        # 2014-01-03 -> Missing reading (must be NaN, NOT 0.0)
        row_nan = long_df[long_df["timestamp"] == "2014-01-03"].iloc[0]
        self.assertTrue(pd.isna(row_nan["consumption_value"]))
        self.assertEqual(row_nan["data_quality_flags"], QualityFlag.MISSING_VALUE.value)

        # 2014-01-04 -> High Outlier
        row_out = long_df[long_df["timestamp"] == "2014-01-04"].iloc[0]
        self.assertEqual(row_out["consumption_value"], 1500.0)
        self.assertEqual(row_out["data_quality_flags"], QualityFlag.OUTLIER_HIGH.value)

        # 2014-01-05 -> Valid reading
        row_val = long_df[long_df["timestamp"] == "2014-01-05"].iloc[0]
        self.assertEqual(row_val["consumption_value"], 12.5)
        self.assertEqual(row_val["data_quality_flags"], QualityFlag.VALID.value)

    def test_raw_files_unmodified(self) -> None:
        """Verify that raw dataset files exist and were never modified or overwritten."""
        p1 = RAW_DATA_DIR / "data.csv"
        p2 = RAW_DATA_DIR / "Electricity_Theft_Data.csv"

        self.assertTrue(p1.exists())
        self.assertTrue(p2.exists())

        # Exact byte sizes matching original download
        self.assertEqual(p1.stat().st_size, 175194613)
        self.assertEqual(p2.stat().st_size, 15141941)

    def test_real_data_subset_integration(self) -> None:
        """Practical integration test: Stream 3 consumers from each real dataset."""
        p1 = RAW_DATA_DIR / "data.csv"
        p2 = RAW_DATA_DIR / "Electricity_Theft_Data.csv"

        if p1.exists():
            adapter1 = DataCsvAdapter(p1)
            chunks1 = list(adapter1.stream_standardized_long_format(chunk_size=10, max_consumers=3))
            df1 = pd.concat(chunks1, ignore_index=True)
            self.assertEqual(df1["consumer_id"].nunique(), 3)
            self.assertEqual(len(df1), 3 * 1034)

        if p2.exists():
            adapter2 = ElectricityTheftAdapter(p2)
            chunks2 = list(adapter2.stream_standardized_long_format(chunk_size=10, max_consumers=3))
            df2 = pd.concat(chunks2, ignore_index=True)
            self.assertEqual(df2["consumer_id"].nunique(), 3)
            self.assertEqual(len(df2), 3 * 365)


if __name__ == "__main__":
    unittest.main()

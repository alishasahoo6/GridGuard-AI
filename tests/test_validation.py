"""Unit tests for data validation rules and quality flags."""

import unittest

import numpy as np
import pandas as pd

from src.data.schemas import QualityFlag
from src.data.validation import (
    assign_quality_flag,
    calculate_missing_streaks,
    validate_consumer_id,
    validate_consumer_profile,
    validate_label_value,
    validate_timestamp,
)


class TestValidationRules(unittest.TestCase):
    """Test suite for domain validation rules and quality flagging."""

    def test_validate_consumer_id_valid(self) -> None:
        valid_id = "0387DD8A07E07FDA6271170F86AD9151"
        is_valid, reason = validate_consumer_id(valid_id)
        self.assertTrue(is_valid)
        self.assertIsNone(reason)

    def test_validate_consumer_id_invalid(self) -> None:
        for bad_id in [None, np.nan, "", "   ", "nan", "NULL"]:
            is_valid, reason = validate_consumer_id(bad_id)
            self.assertFalse(is_valid, f"Expected invalid for {bad_id}")
            self.assertIsNotNone(reason)

    def test_validate_timestamp(self) -> None:
        self.assertTrue(validate_timestamp("2015-01-01")[0])
        self.assertTrue(validate_timestamp("2015/1/1")[0])
        self.assertFalse(validate_timestamp(None)[0])
        self.assertFalse(validate_timestamp("not-a-date")[0])

    def test_validate_label_value(self) -> None:
        self.assertTrue(validate_label_value(0)[0])
        self.assertTrue(validate_label_value(1.0)[0])
        self.assertFalse(validate_label_value(None)[0])
        self.assertFalse(validate_label_value(np.nan)[0])
        self.assertFalse(validate_label_value(5)[0])

    def test_assign_quality_flag(self) -> None:
        self.assertEqual(assign_quality_flag(None), QualityFlag.MISSING_VALUE.value)
        self.assertEqual(assign_quality_flag(np.nan), QualityFlag.MISSING_VALUE.value)
        self.assertEqual(assign_quality_flag(0.0), QualityFlag.ZERO_READING.value)
        self.assertEqual(assign_quality_flag(-12.5), QualityFlag.NEGATIVE_READING.value)
        self.assertEqual(assign_quality_flag(25.4), QualityFlag.VALID.value)
        self.assertIn(
            QualityFlag.OUTLIER_HIGH.value, assign_quality_flag(1500.0, outlier_threshold=1000.0)
        )

    def test_calculate_missing_streaks(self) -> None:
        # 10 readings with a streak of 4 missing in the middle
        series = pd.Series([1.0, 2.0, np.nan, np.nan, np.nan, np.nan, 3.0, 4.0, np.nan, np.nan])
        streaks = calculate_missing_streaks(series)
        self.assertEqual(streaks["max_missing_streak"], 4)
        self.assertEqual(streaks["consecutive_missing_at_end"], 2)

    def test_validate_consumer_profile(self) -> None:
        readings = pd.Series([10.0, 0.0, -1.0, np.nan, 20.0])
        report = validate_consumer_profile("METER_001", readings, label=1)
        self.assertFalse(report["is_quarantined"])
        self.assertEqual(report["total_days"], 5)
        self.assertEqual(report["missing_days"], 1)
        self.assertEqual(report["zero_days"], 1)
        self.assertEqual(report["negative_days"], 1)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for dataset compatibility assessment."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data.compatibility import assess_dataset_compatibility


class TestDatasetCompatibility(unittest.TestCase):
    """Test suite for cross-dataset compatibility checks."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_assess_dataset_compatibility_mock(self) -> None:
        """Verify compatibility assessment logic on small controlled fixtures."""
        p1 = self.tmp_path / "mock_data.csv"
        p2 = self.tmp_path / "mock_theft.csv"

        df1 = pd.DataFrame({
            "CONS_NO": ["32CHARHEXHASH0000000000000000001", "32CHARHEXHASH0000000000000000002"],
            "FLAG": [0, 1],
            "2015/1/1": [10.0, 12.0],
            "2015/1/2": [11.0, 14.0],
        })
        df1.to_csv(p1, index=False)

        df2 = pd.DataFrame({
            "CONS_NO": ["8546920085", "617718257"],
            "01-01-15": [5.0, 7.0],
            "02-01-15": [6.0, 8.0],
            "CHK_STATE": [0, 1],
        })
        df2.to_csv(p2, index=False)

        assessment = assess_dataset_compatibility(path_d1=p1, path_d2=p2)

        self.assertIn("compatibility_metrics", assessment)
        self.assertIn("architectural_recommendation", assessment)
        self.assertEqual(
            assessment["compatibility_metrics"]["consumer_id_overlap_count"], 0
        )
        self.assertTrue(
            assessment["compatibility_metrics"]["has_temporal_overlap"]
        )
        self.assertIn("KEEP STRICTLY SEPARATE", assessment["architectural_recommendation"])


if __name__ == "__main__":
    unittest.main()

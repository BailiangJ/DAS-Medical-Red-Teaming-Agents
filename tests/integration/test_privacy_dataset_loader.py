import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from med_red_team.privacy.privacy_dataset_loader import load_privacy_test_cases


class PrivacyDatasetLoaderTests(unittest.TestCase):
    def test_max_samples_counts_valid_nonblank_cases(self):
        frame = pd.DataFrame({
            "Case Plain": ["", "  ", "First valid", None, "Second valid", "Third valid"],
            "Category": ["blank", "blank", "one", "blank", "two", "three"],
            "Diagnosis": ["", "", "d1", "", "d2", "d3"],
        })

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "privacy.xlsx"
            source.touch()
            with patch(
                "med_red_team.privacy.privacy_dataset_loader.pd.read_excel",
                return_value=frame,
            ):
                cases = load_privacy_test_cases(
                    str(source),
                    max_samples=2,
                )

        self.assertEqual(
            [case.original_prompt for case in cases],
            ["First valid", "Second valid"],
        )
        self.assertEqual([case.case_id for case in cases], ["case_001", "case_002"])
        self.assertEqual([case.sample_number for case in cases], [1, 2])
        self.assertEqual([case.category for case in cases], ["one", "two"])

    def test_zero_limit_returns_empty_without_reading_source(self):
        with patch(
            "med_red_team.privacy.privacy_dataset_loader.pd.read_excel"
        ) as read_excel:
            cases = load_privacy_test_cases("missing.xlsx", max_samples=0)

        self.assertEqual(cases, [])
        read_excel.assert_not_called()

    def test_negative_limit_is_invalid(self):
        with self.assertRaises(ValueError):
            load_privacy_test_cases("missing.xlsx", max_samples=-1)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.configs.models.baseline_model import BaselineModel


class BaselineTargetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.model = BaselineModel(
            models_dir=root / "models",
            plots_dir=root / "plots",
        )

    def test_rejects_missing_target_labels_with_actionable_message(self):
        frame = pd.DataFrame({
            "sensor": [0.1, 0.2, 0.3, 0.4],
            "failure": [0, 1, None, 0],
        })

        with self.assertRaisesRegex(ValueError, "contains 1 missing value"):
            self.model.train(frame)

    def test_rejects_singleton_target_class_before_stratified_split(self):
        frame = pd.DataFrame({
            "sensor": [0.1, 0.2, 0.3],
            "failure": [0, 0, 1],
        })

        with self.assertRaisesRegex(ValueError, "at least 2 rows for every class"):
            self.model.train(frame)

    def test_rejects_test_split_too_small_to_represent_each_class(self):
        frame = pd.DataFrame({
            "sensor": list(range(20)),
            "failure": [0] * 10 + [1] * 10,
        })
        model = BaselineModel(
            test_size=0.05,
            models_dir=Path(self.temp_dir.name) / "small-models",
            plots_dir=Path(self.temp_dir.name) / "small-plots",
        )

        with self.assertRaisesRegex(ValueError, "too small for a stratified train/test split"):
            model.train(frame)


if __name__ == "__main__":
    unittest.main()

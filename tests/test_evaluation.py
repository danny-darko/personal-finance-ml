"""Evaluation checks independent of private data or saved model artifacts."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.model_selection import train_test_split

from scripts import evaluate_model


class EvaluationTests(unittest.TestCase):
    def make_training_file(self, root, target):
        path = root / (target + ".csv")
        frame = pd.DataFrame({
            "Description": [
                "market groceries token" + str(index) if index % 2 == 0
                else "rail ticket token" + str(index)
                for index in range(20)
            ],
            target: (["Food", "Travel"] if target == "Category"
                     else ["Market", "Rail"]) * 10,
        })
        frame.to_csv(path, index=False)
        return path, frame

    def test_evaluates_each_target_without_saving_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target in ("Category", "Payee"):
                with self.subTest(target=target):
                    path, frame = self.make_training_file(root, target)
                    with patch("joblib.dump", side_effect=AssertionError("Saved model")):
                        accuracy, report = evaluate_model.evaluate_model(
                            path, target_column=target
                        )
                    self.assertGreaterEqual(accuracy, 0.0)
                    self.assertLessEqual(accuracy, 1.0)
                    self.assertIn("macro avg", report)
                    for label in frame[target].unique():
                        self.assertIn(label, report)

    def test_only_training_descriptions_reach_model_fitting(self):
        from scripts.model_training import fit_model

        with tempfile.TemporaryDirectory() as directory:
            path, frame = self.make_training_file(Path(directory), "Category")
            train, test = train_test_split(
                frame["Description"], test_size=0.2, random_state=42
            )
            with patch("scripts.evaluate_model.fit_model", wraps=fit_model) as fit:
                evaluate_model.evaluate_model(path)
            fitted_descriptions = fit.call_args.args[0]
            self.assertEqual(set(fitted_descriptions), set(train))
            self.assertTrue(set(fitted_descriptions).isdisjoint(test))

    def test_missing_target_column_has_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.make_training_file(Path(directory), "Category")
            with self.assertRaisesRegex(ValueError, "Description and Payee"):
                evaluate_model.evaluate_model(path, target_column="Payee")

    def test_default_paths_follow_the_requested_target(self):
        for target, filename in (
            ("Category", "training_data_category.csv"),
            ("Payee", "training_data_payee.csv"),
        ):
            with self.subTest(target=target):
                with patch("scripts.evaluate_model.load_training_data") as load:
                    load.side_effect = FileNotFoundError("Stop before training")
                    with self.assertRaises(FileNotFoundError):
                        evaluate_model.evaluate_model(target_column=target)
                self.assertEqual(Path(load.call_args.args[0]).name, filename)
                self.assertEqual(load.call_args.args[1], target)


if __name__ == "__main__":
    unittest.main()

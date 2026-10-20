"""Import safety and execution checks using synthetic data and temporary files."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "train_model_category", "train_model_payee", "process_transactions",
    "evaluate_model",
)


class ScriptEntrypointTests(unittest.TestCase):
    def import_script(self, name):
        """Load under guards so a regression cannot touch project artifacts."""
        blocked_operations = (
            "os.makedirs", "pandas.read_csv", "pandas.read_excel",
            "pandas.DataFrame.to_csv", "joblib.load", "joblib.dump",
            "sklearn.linear_model.LogisticRegression.fit",
            "sklearn.linear_model.LogisticRegression.predict",
            "sklearn.feature_extraction.text.TfidfVectorizer.fit_transform",
            "sklearn.feature_extraction.text.TfidfVectorizer.transform",
            "builtins.print",
        )
        with ExitStack() as stack:
            for operation in blocked_operations:
                stack.enter_context(patch(
                    operation,
                    side_effect=AssertionError("Import called " + operation),
                ))
            spec = importlib.util.spec_from_file_location(
                "scripts.import_check_" + name, ROOT / "scripts" / (name + ".py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    # Import each script and check that importing does not start work
    def test_imports_do_not_run_the_workflow(self):
        for name in SCRIPT_NAMES:
            with self.subTest(script=name):
                self.assertTrue(callable(self.import_script(name).main))

    def make_inputs(self, root):
        data = root / "data"
        data.mkdir()
        training = pd.DataFrame({
            "Description": ["market groceries", "rail ticket"] * 10,
            "Category": ["Food", "Travel"] * 10,
            "Payee": ["Market", "Rail"] * 10,
        })
        training[["Description", "Category"]].to_csv(
            data / "training_data_category.csv", index=False
        )
        training[["Description", "Payee"]].to_csv(
            data / "training_data_payee.csv", index=False
        )
        statement = pd.DataFrame({
            "UnusedA": ["", ""],
            "Date": pd.to_datetime(["2026-02-02", "2026-01-01"]),
            "UnusedC": ["", ""],
            "Description": ["market groceries", "rail ticket"],
            "UnusedE": ["", ""],
            "MoneyIn": [None, 25.0],
            "MoneyOut": [12.5, None],
        })
        statement.to_excel(data / "bank_statement.xlsx", startrow=4, index=False)
        return data

    # Check column order, chronological sorting, amounts, predictions, account and memo formatting
    def assert_export(self, path):
        exported = pd.read_csv(path)
        self.assertEqual(list(exported.columns), [
            "Date", "Outflow", "Inflow", "Category", "Account", "Payee", "Memo",
        ])
        self.assertEqual(len(exported), 2)
        self.assertTrue(pd.to_datetime(exported["Date"]).is_monotonic_increasing)
        self.assertEqual(exported.loc[0, "Inflow"], 25.0)
        self.assertEqual(exported.loc[1, "Outflow"], 12.5)
        self.assertTrue(pd.isna(exported.loc[0, "Outflow"]))
        self.assertTrue(pd.isna(exported.loc[1, "Inflow"]))
        self.assertEqual(exported["Category"].tolist(), ["Travel", "Food"])
        self.assertEqual(exported["Payee"].tolist(), ["Rail", "Market"])
        self.assertEqual(exported["Account"].tolist(), ["Main", "Main"])
        self.assertEqual(exported["Memo"].tolist(), [
            "'Rail' rail ticket", "'Market' market groceries",
        ])

    # Checks that another python program can call functions with its own file location
    def test_functions_accept_explicit_paths(self):
        train_model_category = self.import_script("train_model_category")
        train_model_payee = self.import_script("train_model_payee")
        process_transactions = self.import_script("process_transactions")
        evaluate_model = self.import_script("evaluate_model")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            data = self.make_inputs(root)
            models = root / "custom_models"
            for target, module in (
                ("category", train_model_category), ("payee", train_model_payee)
            ):
                module.train_model(
                    training_file=data / ("training_data_" + target + ".csv"),
                    model_file=models / (target + ".pkl"),
                    vectorizer_file=models / (target + "_vectorizer.pkl"),
                )
            output = root / "custom_output" / "transactions.csv"
            result = process_transactions.process_transactions(
                bank_xlsx=data / "bank_statement.xlsx",
                model_file_category=models / "category.pkl",
                vectorizer_file_category=models / "category_vectorizer.pkl",
                model_file_payee=models / "payee.pkl",
                vectorizer_file_payee=models / "payee_vectorizer.pkl",
                output_file=output,
            )
            self.assertEqual(len(result), 2)
            self.assert_export(output)
            accuracy, report = evaluate_model.evaluate_model(
                data / "training_data_category.csv"
            )
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)
            self.assertIn("precision", report)

    def test_scripts_still_run_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inputs(root)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copyfile(ROOT / "scripts" / "model_training.py", scripts / "model_training.py")
            # Run copies so default paths cannot overwrite project artifacts.
            environment = dict(os.environ, PYTHONIOENCODING="utf-8")
            for name in SCRIPT_NAMES:
                script = scripts / (name + ".py")
                shutil.copyfile(ROOT / "scripts" / script.name, script)
                completed = subprocess.run(
                    [sys.executable, str(script)], cwd=root,
                    env=environment, capture_output=True, text=True,
                    encoding="utf-8", timeout=60,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertTrue(completed.stdout.strip())
                if name == "evaluate_model":
                    self.assertIn("Category evaluation", completed.stdout)
                    self.assertIn("Payee evaluation", completed.stdout)
            self.assert_export(root / "output" / "processed_transactions.csv")


if __name__ == "__main__":
    unittest.main()

"""Regression and upload-validation checks for the purchase evaluation service."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from shopper_app.evaluation import (
    DataValidationError,
    FEATURE_COLUMNS,
    KNOWN_CATEGORIES,
    MAX_BYTES,
    MAX_ROWS,
    MODEL_FILES,
    dataset_warnings,
    evaluate_model,
    load_pipeline,
    read_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8")


class PublishedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = read_dataset((ROOT / "test_data.csv").read_bytes())
        cls.expected = pd.read_csv(ROOT / "results/model_comparison.csv", index_col="Model")
        cls.errors = pd.read_csv(ROOT / "results/error_counts.csv", index_col="Model")

    def test_all_five_models_reproduce_published_results(self):
        self.assertEqual(len(MODEL_FILES), 5)
        for name in MODEL_FILES:
            with self.subTest(model=name):
                result = evaluate_model(load_pipeline(name, ROOT), self.frame)
                self.assertEqual(set(result["metrics"]), set(self.expected.columns))
                np.testing.assert_allclose(
                    [result["metrics"][column] for column in self.expected.columns],
                    self.expected.loc[name].to_numpy(dtype=float),
                    rtol=1e-7,
                    atol=1e-9,
                )
                np.testing.assert_array_equal(
                    result["confusion"].ravel(), self.errors.loc[name].to_numpy(dtype=int)
                )
                self.assertEqual(result["confusion"].sum(), len(self.frame))
                predictions = result["predictions"]
                self.assertEqual(len(predictions), len(self.frame))
                self.assertTrue({"PredictedRevenue", "PredictionScore", "Correct"} <= set(predictions))
                pd.testing.assert_frame_equal(predictions[self.frame.columns], self.frame)
                np.testing.assert_array_equal(
                    predictions["Correct"], predictions["PredictedRevenue"] == self.frame["Revenue"]
                )
                np.testing.assert_array_equal(result["y_true"], self.frame["Revenue"].astype(int))
                np.testing.assert_allclose(predictions["PredictionScore"], result["scores"])
                self.assertTrue({"No purchase", "Purchase"} <= set(result["report"].index))

    def test_svm_exports_decision_scores_without_probability_conversion(self):
        model = load_pipeline("Support Vector Machine", ROOT)
        result = evaluate_model(model, self.frame)
        np.testing.assert_allclose(
            result["scores"], model.decision_function(self.frame[FEATURE_COLUMNS])
        )
        self.assertLess(float(np.min(result["scores"])), 0)

    def test_warning_categories_match_each_fitted_encoder_exactly(self):
        for name in MODEL_FILES:
            with self.subTest(model=name):
                prepare = load_pipeline(name, ROOT).named_steps["prepare"]
                columns = next(
                    columns for key, transformer, columns in prepare.transformers_
                    if key == "category"
                )
                observed = dict(zip(
                    columns, prepare.named_transformers_["category"].categories_
                ))
                self.assertEqual(set(KNOWN_CATEGORIES), set(observed) - {"Weekend"})
                for column, expected in KNOWN_CATEGORIES.items():
                    with self.subTest(column=column):
                        self.assertEqual(expected, set(observed[column]))
                self.assertEqual(set(observed["Weekend"]), {False, True})

    def test_single_class_upload_keeps_counts_and_marks_undefined_metrics(self):
        model = load_pipeline("Decision Tree", ROOT)
        for outcome in (False, True):
            with self.subTest(outcome=outcome):
                subset = self.frame.loc[self.frame["Revenue"] == outcome].head(20).reset_index(drop=True)
                result = evaluate_model(model, subset)
                self.assertIsNone(result["metrics"]["AUC"])
                self.assertIsNone(result["metrics"]["MCC"])
                self.assertEqual(result["confusion"].shape, (2, 2))
                self.assertEqual(result["confusion"].sum(), len(subset))
                for metric in ("Accuracy", "Precision", "Recall", "F1"):
                    self.assertTrue(np.isfinite(result["metrics"][metric]))
                self.assertTrue(dataset_warnings(subset))

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            load_pipeline("not-a-model", ROOT)

    def test_corrupted_model_is_rejected_before_deserialization(self):
        name = "Decision Tree"
        with tempfile.TemporaryDirectory(prefix="app-model-check-", dir=ROOT) as directory:
            temporary_root = Path(directory)
            (temporary_root / "model").mkdir()
            (temporary_root / "results").mkdir()
            manifest = json.loads((ROOT / "results/experiment_manifest.json").read_text())
            (temporary_root / "results/experiment_manifest.json").write_text(json.dumps(manifest))
            (temporary_root / "model" / MODEL_FILES[name]).write_bytes(b"altered model bytes")
            with patch("shopper_app.evaluation.joblib.load") as deserialize:
                with self.assertRaises((ValueError, RuntimeError)):
                    load_pipeline(name, temporary_root)
                deserialize.assert_not_called()


class UploadValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(ROOT / "test_data.csv").head(8)

    def test_reordered_and_whitespace_headers_with_utf8_bom(self):
        expected = read_dataset(csv_bytes(self.frame))
        shuffled = self.frame[self.frame.columns[::-1]].copy()
        shuffled.columns = [f" {name} " for name in shuffled.columns]
        actual = read_dataset(b"\xef\xbb\xbf" + csv_bytes(shuffled))
        pd.testing.assert_frame_equal(actual, expected)

    def test_boolean_and_numeric_labels_are_normalized(self):
        frame = self.frame.head(6).copy()
        frame["Revenue"] = ["TRUE", "false", " 1 ", " 0 ", "True", " FALSE "]
        frame["Weekend"] = ["0", "1", "true", "FALSE", " True ", " false "]
        result = read_dataset(csv_bytes(frame))
        self.assertEqual(result["Revenue"].tolist(), [True, False, True, False, True, False])
        self.assertEqual(result["Weekend"].tolist(), [False, True, True, False, True, False])
        self.assertTrue(pd.api.types.is_bool_dtype(result["Revenue"]))
        self.assertTrue(pd.api.types.is_bool_dtype(result["Weekend"]))

    def test_missing_target_and_extra_columns_are_rejected(self):
        for frame in [self.frame.drop(columns="Revenue"), self.frame.assign(Unexpected=1)]:
            with self.subTest(columns=list(frame.columns)):
                with self.assertRaises(DataValidationError):
                    read_dataset(csv_bytes(frame))

    def test_duplicate_headers_are_rejected(self):
        lines = csv_bytes(self.frame).decode().splitlines()
        header = lines[0].split(",")
        header[-1] = header[0]
        payload = (",".join(header) + "\n" + "\n".join(lines[1:])).encode()
        with self.assertRaisesRegex(DataValidationError, "[Dd]uplicate"):
            read_dataset(payload)

    def test_invalid_values_are_rejected(self):
        invalid_values = [
            ("ProductRelated_Duration", "not-a-number"),
            ("ProductRelated_Duration", -1),
            ("ProductRelated_Duration", float("inf")),
            ("Administrative", 0.5),
            ("Browser", 1.5),
            ("Region", 0),
            ("BounceRates", 1.1),
            ("ExitRates", -0.01),
            ("SpecialDay", 1.5),
            ("Revenue", "maybe"),
            ("Weekend", "maybe"),
            ("Month", None),
        ]
        for column, value in invalid_values:
            with self.subTest(column=column, value=value):
                frame = self.frame.copy()
                frame[column] = frame[column].astype(object)
                frame.loc[frame.index[0], column] = value
                with self.assertRaises(DataValidationError):
                    read_dataset(csv_bytes(frame))

    def test_empty_and_malformed_files_are_rejected(self):
        header = csv_bytes(self.frame).splitlines()[0]
        for payload in [b"", b" \n", header + b"\n", header + b'\n"unterminated', b"\xff\xfe\x00"]:
            with self.subTest(payload=payload[:30]):
                with self.assertRaises(DataValidationError):
                    read_dataset(payload)

    def test_extra_or_missing_csv_fields_are_rejected(self):
        lines = csv_bytes(self.frame.head(1)).splitlines()
        for row in [lines[1] + b",extra", b",".join(lines[1].split(b",")[:-1])]:
            with self.subTest(row=row):
                with self.assertRaises(DataValidationError):
                    read_dataset(lines[0] + b"\n" + row)

    def test_upload_size_and_row_limits_are_enforced(self):
        with self.assertRaises(DataValidationError):
            read_dataset(b" " * (MAX_BYTES + 1))
        oversized = pd.concat([self.frame] * (MAX_ROWS // len(self.frame) + 1), ignore_index=True)
        with self.assertRaises(DataValidationError):
            read_dataset(csv_bytes(oversized.head(MAX_ROWS + 1)))

    def test_unseen_categories_remain_evaluable_with_a_warning(self):
        frame = self.frame.copy()
        frame.loc[0, "Month"] = "UnseenMonth"
        frame.loc[0, "VisitorType"] = "UnseenVisitor"
        normalized = read_dataset(csv_bytes(frame))
        self.assertTrue(dataset_warnings(normalized))
        result = evaluate_model(load_pipeline("Decision Tree", ROOT), normalized)
        self.assertEqual(len(result["predictions"]), len(frame))

    def test_duplicate_rows_are_preserved_and_disclosed(self):
        duplicated = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        normalized = read_dataset(csv_bytes(duplicated))
        self.assertEqual(len(normalized), len(duplicated))
        self.assertTrue(any(
            "duplicat" in message.lower() or "repeated rows" in message.lower()
            for message in dataset_warnings(normalized)
        ))


if __name__ == "__main__":
    unittest.main()

"""Regression and upload-validation checks for the purchase evaluation service."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from shopper_app.evaluation import (
    FEATURE_COLUMNS,
    KNOWN_CATEGORIES,
    MAX_BYTES,
    MAX_ROWS,
    MODEL_FILES,
    DataValidationError,
    dataset_fingerprint,
    dataset_warnings,
    evaluate_model,
    load_pipeline,
    model_metadata,
    purchase_score,
    read_dataset,
    schema_guide,
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

    def test_feature_only_data_is_opt_in_and_partial_targets_are_rejected(self):
        features = self.frame.drop(columns="Revenue")
        normalized = read_dataset(csv_bytes(features), require_target=False)
        self.assertEqual(list(normalized), FEATURE_COLUMNS)
        self.assertTrue(any("Revenue is absent" in message for message in dataset_warnings(normalized)))
        with self.assertRaisesRegex(DataValidationError, "Revenue"):
            read_dataset(csv_bytes(features))
        partial = self.frame.copy()
        partial["Revenue"] = partial["Revenue"].astype(object)
        partial.loc[2, "Revenue"] = ""
        with self.assertRaisesRegex(DataValidationError, "CSV row 4"):
            read_dataset(csv_bytes(partial), require_target=False)

    def test_validation_identifies_the_column_and_csv_record(self):
        frame = self.frame.copy()
        frame.loc[3, "BounceRates"] = -0.3
        with self.assertRaisesRegex(DataValidationError, "BounceRates.*CSV row 5"):
            read_dataset(csv_bytes(frame))

    def test_fingerprint_ignores_formatting_but_tracks_data_and_order(self):
        original = read_dataset(csv_bytes(self.frame))
        reordered = self.frame.loc[:, self.frame.columns[::-1]].copy()
        reordered.columns = [f" {column} " for column in reordered]
        normalized = read_dataset(b"\xef\xbb\xbf" + csv_bytes(reordered))
        self.assertEqual(dataset_fingerprint(original), dataset_fingerprint(normalized))
        self.assertNotEqual(dataset_fingerprint(original), dataset_fingerprint(original.iloc[::-1]))
        modified = original.copy()
        modified.loc[0, "PageValues"] += 1
        self.assertNotEqual(dataset_fingerprint(original), dataset_fingerprint(modified))

    def test_schema_guide_matches_validation_fields_and_categories(self):
        guide = schema_guide().set_index("Field")
        self.assertEqual(list(guide.index), [*FEATURE_COLUMNS, "Revenue"])
        self.assertTrue(guide.loc[FEATURE_COLUMNS, "Required"].eq("Yes").all())
        self.assertIn("optional", guide.loc["Revenue", "Required"])
        self.assertEqual(schema_guide(require_target=True).iloc[-1]["Required"], "Yes")
        for column, categories in KNOWN_CATEGORIES.items():
            for category in categories:
                self.assertIn(str(category), guide.loc[column, "Accepted values"])


class FixedPredictions:
    classes_ = np.array([0, 1])

    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)

    def predict(self, features):
        return self.predictions[:len(features)]


class ProbabilityPredictions(FixedPredictions):
    def __init__(self, predictions, probabilities, classes=(0, 1)):
        super().__init__(predictions)
        self.probabilities = np.asarray(probabilities)
        self.classes_ = np.asarray(classes)

    def predict_proba(self, features):
        return self.probabilities[:len(features)]


class DecisionPredictions(FixedPredictions):
    def __init__(self, predictions, decisions, classes=(0, 1)):
        super().__init__(predictions)
        self.decisions = np.asarray(decisions)
        self.classes_ = np.asarray(classes)

    def decision_function(self, features):
        return self.decisions[:len(features)]


class PredictionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = read_dataset((ROOT / "test_data.csv").read_bytes()).head(4).copy()
        cls.frame["Revenue"] = [False, True, False, True]

    def test_feature_only_results_keep_saved_decisions_and_hide_evaluation(self):
        labeled = read_dataset(csv_bytes(self.frame))
        features = read_dataset(csv_bytes(self.frame.drop(columns="Revenue")), require_target=False)
        for name in MODEL_FILES:
            with self.subTest(model=name):
                pipeline = load_pipeline(name, ROOT)
                evaluated = evaluate_model(pipeline, labeled)
                predicted = evaluate_model(pipeline, features)
                np.testing.assert_array_equal(
                    evaluated["predictions"]["PredictedRevenue"], predicted["predictions"]["PredictedRevenue"],
                )
                np.testing.assert_allclose(evaluated["scores"], predicted["scores"])
                self.assertFalse(predicted["labeled"])
                self.assertTrue(predicted["score_available"])
                self.assertTrue(all(value is None for value in predicted["metrics"].values()))
                self.assertIsNone(predicted["confusion"])
                self.assertIsNone(predicted["y_true"])
                self.assertTrue(predicted["report"].empty)
                self.assertTrue(predicted["predictions"]["Correct"].isna().all())
                self.assertTrue(predicted["predictions"]["ResultCategory"].eq("Unlabeled").all())

    def test_positive_class_probability_and_decision_mapping(self):
        model = ProbabilityPredictions([0, 1, 0, 1], [[.2, .8], [.8, .2], [.3, .7], [.7, .3]], classes=(1, 0))
        result = evaluate_model(model, self.frame)
        np.testing.assert_allclose(result["scores"], [.2, .8, .3, .7])
        self.assertEqual(result["metrics"]["AUC"], 1)
        self.assertEqual(result["positive_class"], 1)
        reversed_decision = DecisionPredictions([0, 1, 0, 1], [3, -3, 2, -2], classes=(1, 0))
        np.testing.assert_array_equal(purchase_score(reversed_decision, self.frame), [-3, 3, -2, 2])

    def test_unavailable_scores_preserve_predictions_and_other_metrics(self):
        for model in [
            FixedPredictions([0, 1, 0, 1]),
            DecisionPredictions([0, 1, 0, 1], [0, 1, np.nan, 1]),
        ]:
            with self.subTest(model=type(model).__name__):
                result = evaluate_model(model, self.frame)
                self.assertIsNone(result["scores"])
                self.assertFalse(result["score_available"])
                self.assertEqual(result["score_label"], "Unavailable")
                self.assertIsNone(result["metrics"]["AUC"])
                self.assertEqual(result["metrics"]["Accuracy"], 1)
                self.assertTrue(result["predictions"]["PredictionScore"].isna().all())

    def test_undefined_report_values_remain_missing(self):
        result = evaluate_model(FixedPredictions([0, 0, 0, 0]), self.frame)
        self.assertTrue(np.isnan(result["report"].loc["Purchase", "precision"]))
        self.assertEqual(result["report"].loc["Purchase", "support"], 2)
        self.assertEqual(result["metrics"]["Precision"], 0)
        single = self.frame.copy()
        single["Revenue"] = False
        report = evaluate_model(FixedPredictions([0, 0, 0, 0]), single)["report"]
        self.assertTrue(np.isnan(report.loc["Purchase", "recall"]))
        self.assertTrue(np.isnan(report.loc["Purchase", "f1-score"]))

    def test_operating_point_requires_exact_saved_rule_including_ties(self):
        matched = evaluate_model(DecisionPredictions([0, 1, 0, 1], [-1, 1, 0, 2]), self.frame)
        self.assertEqual(matched["operating_point"]["comparison"], ">")
        self.assertEqual(matched["operating_point"]["threshold"], 0)
        tied = evaluate_model(DecisionPredictions([0, 1, 0, 1], [-1, 0, 0, 2]), self.frame)
        self.assertIsNone(tied["operating_point"])

    def test_invalid_predictions_and_class_mapping_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_model(FixedPredictions([0, .5, 0, 1]), self.frame)
        model = FixedPredictions([0, 1, 0, 1])
        model.classes_ = np.array([1, 2])
        with self.assertRaisesRegex(ValueError, "binary classes"):
            evaluate_model(model, self.frame)

    def test_model_identity_covers_both_classifier_and_preprocessor(self):
        for name in MODEL_FILES:
            with self.subTest(model=name):
                first = model_metadata(load_pipeline(name, ROOT), name)
                self.assertEqual(first, model_metadata(load_pipeline(name, ROOT), name))
        pipeline = load_pipeline("Decision Tree", ROOT)
        before = model_metadata(pipeline, "Decision Tree")
        self.assertEqual(before, model_metadata(deepcopy(pipeline), "Decision Tree"))
        self.assertEqual(before["positive_class"], 1)
        self.assertEqual(before["feature_count"], 17)
        self.assertEqual(before["transformed_feature_count"], 75)
        classifier_changed = deepcopy(pipeline)
        classifier_changed.named_steps["classifier"].max_depth += 1
        updated = model_metadata(classifier_changed)
        self.assertNotEqual(before["identity"], updated["identity"])
        self.assertEqual(before["preprocessing_fingerprint"], updated["preprocessing_fingerprint"])
        preprocessing_changed = deepcopy(pipeline)
        preprocessing_changed.named_steps["prepare"].named_transformers_["numeric"].scale_[0] += 1
        updated = model_metadata(preprocessing_changed)
        self.assertNotEqual(before["identity"], updated["identity"])
        self.assertEqual(before["model_fingerprint"], updated["model_fingerprint"])

    def test_evaluation_identity_uses_current_data_instead_of_stale_frame_attributes(self):
        normalized = read_dataset(csv_bytes(self.frame))
        original_fingerprint = normalized.attrs["dataset_fingerprint"]
        normalized.loc[0, "PageValues"] += 1
        result = evaluate_model(FixedPredictions([0, 1, 0, 1]), normalized)
        self.assertNotEqual(result["dataset_fingerprint"], original_fingerprint)
        self.assertEqual(result["dataset_fingerprint"], dataset_fingerprint(normalized))


if __name__ == "__main__":
    unittest.main()

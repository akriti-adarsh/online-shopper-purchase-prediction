"""Stable identity, matrix drill-down, pagination, and export regression checks."""

import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from shopper_app.evaluation import (
    MODEL_FILES,
    evaluate_model,
    load_pipeline,
    read_dataset,
)
from shopper_app.explorer import (
    export_sessions,
    filter_sessions,
    paginate_sessions,
    session_detail,
)

ROOT = Path(__file__).resolve().parents[1]


class SessionExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = read_dataset((ROOT / "test_data.csv").read_bytes())
        cls.model = load_pipeline("k-Nearest Neighbors", ROOT)
        cls.result = evaluate_model(cls.model, cls.frame)
        cls.predictions = cls.result["predictions"]

    def test_each_matrix_cell_selects_exactly_the_corresponding_rows(self):
        for name in MODEL_FILES:
            result = evaluate_model(load_pipeline(name, ROOT), self.frame)
            categories = ["Correct non-buyers", "False alerts", "Missed buyers", "Identified buyers"]
            counts = [len(filter_sessions(result["predictions"], category)) for category in categories]
            with self.subTest(model=name):
                np.testing.assert_array_equal(counts, result["confusion"].ravel())
                self.assertEqual(sum(counts), len(self.frame))
        self.assertEqual(len(filter_sessions(self.predictions, "Missed buyers")), 258)
        self.assertEqual(len(filter_sessions(self.predictions, "False alerts")), 55)
        self.assertEqual(len(filter_sessions(self.predictions, "Correct predictions")), 2029 + 124)
        self.assertEqual(len(filter_sessions(self.predictions, "Incorrect predictions")), 55 + 258)

    def test_ids_survive_filter_sort_pagination_detail_and_export(self):
        rows = filter_sessions(self.predictions, "Missed buyers", sort_by="PredictionScore", ascending=False)
        first = paginate_sessions(rows, page=1, page_size=10)
        second = paginate_sessions(rows, page=2, page_size=10)
        self.assertEqual(first["count"], 258)
        self.assertEqual(first["pages"], 26)
        self.assertEqual((second["start"], second["end"]), (11, 20))
        self.assertFalse(set(first["rows"]["SessionID"]) & set(second["rows"]["SessionID"]))
        selected = second["rows"].iloc[0]
        detail = session_detail(self.predictions, selected["SessionID"])
        pd.testing.assert_series_equal(detail, selected)
        exported = pd.read_csv(BytesIO(export_sessions(rows)))
        self.assertEqual(len(exported), 258)
        self.assertEqual(exported["SessionID"].tolist(), rows["SessionID"].tolist())
        self.assertEqual(exported["OriginalRow"].tolist(), rows["OriginalRow"].tolist())
        self.assertEqual(exported["ResultCategory"].unique().tolist(), ["Missed buyers"])
        self.assertTrue(rows["PredictionScore"].is_monotonic_decreasing)

    def test_literal_case_insensitive_search_and_no_match_state(self):
        identifier = self.predictions["SessionID"].iloc[0]
        result = filter_sessions(self.predictions, search=identifier.lower())
        self.assertEqual(result["SessionID"].tolist(), [identifier])
        empty = filter_sessions(self.predictions, search="[not a regular expression")
        page = paginate_sessions(empty, page=10)
        self.assertTrue(page["rows"].empty)
        self.assertEqual((page["count"], page["pages"], page["page"], page["start"], page["end"]), (0, 0, 1, 0, 0))
        self.assertEqual(len(filter_sessions(self.predictions, search="")), len(self.frame))

    def test_page_is_clamped_after_filter_changes(self):
        result = paginate_sessions(self.predictions.head(5), page=20, page_size=2)
        self.assertEqual((result["page"], result["start"], result["end"]), (3, 5, 5))
        self.assertEqual(len(result["rows"]), 1)
        for args in [{"page": 0}, {"page": 1.5}, {"page_size": 0}, {"page_size": 501}, {"page_size": True}]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                paginate_sessions(self.predictions, **args)

    def test_duplicate_input_rows_have_distinct_repeatable_ids(self):
        duplicate = pd.concat([self.frame.iloc[[0]], self.frame.iloc[[0]]], ignore_index=True)
        normalized = read_dataset(duplicate.to_csv(index=False).encode())
        result = evaluate_model(self.model, normalized)["predictions"]
        again = evaluate_model(self.model, normalized)["predictions"]
        self.assertTrue(result["SessionID"].is_unique)
        self.assertEqual(result["OriginalRow"].tolist(), [1, 2])
        self.assertEqual(result["SessionID"].tolist(), again["SessionID"].tolist())
        different_model = evaluate_model(load_pipeline("Decision Tree", ROOT), normalized)["predictions"]
        self.assertEqual(result["SessionID"].tolist(), different_model["SessionID"].tolist())

    def test_unlabeled_rows_are_neither_correct_nor_incorrect(self):
        features = read_dataset(self.frame.head(8).drop(columns="Revenue").to_csv(index=False).encode(), require_target=False)
        predictions = evaluate_model(self.model, features)["predictions"]
        self.assertEqual(len(filter_sessions(predictions, "Unlabeled")), 8)
        self.assertTrue(filter_sessions(predictions, "Correct predictions").empty)
        self.assertTrue(filter_sessions(predictions, "Incorrect predictions").empty)
        exported = pd.read_csv(BytesIO(export_sessions(predictions)))
        self.assertNotIn("Revenue", exported)
        self.assertTrue(exported["Correct"].isna().all())

    def test_export_keeps_formula_like_text_literal_without_changing_predictions(self):
        formula_text = ["=1+1", "+1+1", "-1+1", "@SUM(1,1)", "  =1+1", "\t=1+1", "\r\n=1+1"]
        rows = self.predictions.head(len(formula_text)).copy()
        rows["Month"] = formula_text
        rows["VisitorType"] = pd.Series(["Returning_Visitor"] * len(rows), index=rows.index, dtype="string")
        rows.loc[rows.index[0], "VisitorType"] = "=1+1"
        rows["PredictionScore"] = -0.25
        original = rows.copy(deep=True)

        exported = pd.read_csv(BytesIO(export_sessions(rows)))

        self.assertEqual(exported["Month"].tolist(), ["'" + value for value in formula_text])
        self.assertEqual(exported["VisitorType"].iloc[0], "'=1+1")
        self.assertTrue(exported["VisitorType"].iloc[1:].eq("Returning_Visitor").all())
        self.assertTrue(exported["PredictionScore"].eq(-0.25).all())
        self.assertTrue(pd.api.types.is_numeric_dtype(exported["PredictionScore"]))
        self.assertEqual(exported["SessionID"].tolist(), original["SessionID"].tolist())
        self.assertEqual(len(exported), len(original))
        pd.testing.assert_frame_equal(rows, original)

    def test_invalid_filters_sort_columns_and_ids_are_recoverable(self):
        with self.assertRaises(ValueError):
            filter_sessions(self.predictions, "Unknown category")
        with self.assertRaises(ValueError):
            filter_sessions(self.predictions, sort_by="Unknown column")
        with self.assertRaises(KeyError):
            session_detail(self.predictions, "missing-id")


if __name__ == "__main__":
    unittest.main()

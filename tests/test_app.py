"""Exercise the saved-model workflow through the app's actual native controls."""

import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from shopper_app.evaluation import evaluate_model, load_pipeline, read_dataset
from shopper_app.explorer import export_sessions

ROOT = Path(__file__).resolve().parents[1]
DISPLAY_METRICS = {
    "Accuracy": "Accuracy", "ROC AUC": "AUC", "Precision": "Precision",
    "Recall": "Recall", "F1": "F1", "MCC": "MCC",
}


class RenderedTable(HTMLParser):
    """Read the same table content sent to the browser without a parser dependency."""

    def __init__(self, markup):
        super().__init__()
        self.caption = ""
        self.headers = []
        self.rows = []
        self.selected = []
        self._caption = False
        self._cell = None
        self._row = []
        self._header = False
        self._active = False
        self._skip_label = False
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "caption":
            self._caption = True
        elif tag == "tr":
            self._row = []
            self._header = False
            self._active = attrs.get("class") == "selected-row"
        elif tag in {"td", "th"}:
            self._cell = ""
            self._header = self._header or tag == "th"
        elif tag == "span" and attrs.get("class") == "selected-label":
            self._skip_label = True

    def handle_data(self, data):
        if self._caption:
            self.caption += data
        if self._cell is not None and not self._skip_label:
            self._cell += data

    def handle_endtag(self, tag):
        if tag == "caption":
            self._caption = False
        elif tag == "span":
            self._skip_label = False
        elif tag in {"td", "th"}:
            self._row.append(self._cell.strip())
            self._cell = None
        elif tag == "tr":
            if self._header:
                self.headers = [label.rstrip(" ↑↓") for label in self._row]
            elif self._row:
                self.rows.append(self._row)
                self.selected.append(self._active)

    @property
    def records(self):
        return [dict(zip(self.headers, row)) for row in self.rows]


class StreamlitAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = pd.read_csv(ROOT / "results/model_comparison.csv", index_col="Model")
        cls.counts = pd.read_csv(ROOT / "results/error_counts.csv", index_col="Model")
        cls.test_frame = pd.read_csv(ROOT / "test_data.csv")

    def assert_ok(self, app):
        self.assertEqual(len(app.exception), 0, str(app.exception))
        return app

    def make_app(self):
        return self.assert_ok(AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run())

    def navigate(self, app, view):
        app.segmented_control(key="view").set_value(view).run()
        return self.assert_ok(app)

    def upload_mode(self, app):
        app.radio(key="dataset_source").set_value("Upload CSV").run()
        return self.assert_ok(app)

    def upload(self, app, frame, filename="sessions.csv"):
        payload = frame.to_csv(index=False).encode("utf-8")
        app.file_uploader(key="dataset_upload").set_value((filename, payload, "text/csv")).run()
        self.assert_ok(app)
        return payload

    def metrics(self, app):
        return {metric.label: metric.value for metric in app.metric}

    def assert_metrics(self, app, values):
        displayed = self.metrics(app)
        self.assertEqual(set(displayed), set(DISPLAY_METRICS))
        for label, key in DISPLAY_METRICS.items():
            value = values[key]
            self.assertEqual(displayed[label], "—" if value is None else f"{value:.4f}", label)

    def assert_uploaded_metrics(self, app, payload):
        name = app.selectbox(key="model_name").value
        expected = evaluate_model(load_pipeline(name, ROOT), read_dataset(payload, require_target=False))
        self.assert_metrics(app, expected["metrics"])
        return expected

    def session_table(self, app):
        # AppTest can inspect v2 output but does not implement its browser events.
        # The native keyboard controls below exercise the same application paths.
        components = [element for element in app.get("bidi_component")
                      if element.proto.component_name == "shopper_session_table"]
        self.assertEqual(len(components), 1)
        return RenderedTable(json.loads(components[0].proto.json)["html"])

    def comparison_table(self, app):
        tables = [RenderedTable(element.value) for element in app.html if "<table" in element.value]
        comparisons = [table for table in tables if table.caption.startswith("All models ranked")]
        self.assertEqual(len(comparisons), 1)
        return comparisons[0]

    def matching_count(self, app):
        captions = [element.value for element in app.caption]
        matches = [re.match(r"([\d,]+) matching sessions of ([\d,]+)", caption) for caption in captions]
        matches = [match for match in matches if match]
        self.assertEqual(len(matches), 1)
        return int(matches[0][1].replace(",", ""))

    def test_default_screen_exposes_evaluation_context_metrics_and_navigation(self):
        app = self.make_app()
        self.assertEqual(app.selectbox(key="model_name").value, "Decision Tree")
        self.assertEqual(app.radio(key="dataset_source").value, "Bundled test set")
        self.assertEqual(app.segmented_control(key="view").options,
                         ["Overview", "Compare Models", "Session Explorer", "About"])
        self.assert_metrics(app, self.expected.loc["Decision Tree"])
        context = " ".join(element.value for element in app.html)
        self.assertIn("2,466", context)
        self.assertIn("382 actual buyers", context)
        self.assertIn("Positive class", context)
        self.assertTrue(all(metric.help for metric in app.metric))
        self.assertTrue(app.download_button(key="download_test").url.endswith(".csv"))

    def test_every_model_updates_all_metrics_and_outcome_counts(self):
        app = self.make_app()
        columns = {"tn": "True negatives", "fp": "False positives",
                   "fn": "False negatives", "tp": "True positives"}
        for name, expected in self.expected.iterrows():
            with self.subTest(model=name):
                app.selectbox(key="model_name").set_value(name).run()
                self.assert_ok(app)
                self.assert_metrics(app, expected)
                for code, column in columns.items():
                    expected_count = int(self.counts.loc[name, column])
                    self.assertTrue(app.button(key=f"cell_{code}").label.startswith(f"**{expected_count:,}**"))

    def test_matrix_drilldown_matches_all_four_knn_outcomes_and_preserves_context(self):
        app = self.make_app()
        app.selectbox(key="model_name").set_value("k-Nearest Neighbors").run()
        identity = app.session_state["active_id"]
        expected = {"tn": ("Correct non-buyers", 2029), "fp": ("False alerts", 55),
                    "fn": ("Missed buyers", 258), "tp": ("Identified buyers", 124)}
        for code, (category, count) in expected.items():
            with self.subTest(category=category):
                app.button(key=f"cell_{code}").click().run()
                self.assert_ok(app)
                self.assertEqual(app.segmented_control(key="view").value, "Session Explorer")
                self.assertEqual(app.selectbox(key="outcome_filter").value, category)
                self.assertEqual(self.matching_count(app), count)
                table = self.session_table(app)
                self.assertEqual(len(table.rows), min(count, 25))
                self.assertTrue(all(row["Outcome"] == category for row in table.records))
                self.assertEqual(app.session_state["active_id"], identity)
                self.assertEqual(app.selectbox(key="model_name").value, "k-Nearest Neighbors")
                app.button(key="back_to_overview").click().run()
                self.assert_ok(app)
                self.assertEqual(app.segmented_control(key="view").value, "Overview")
                self.assertEqual(app.session_state["focus_request"]["selector"], f".st-key-cell_{code} button")

    def test_percentage_matrix_keeps_counts_and_actual_class_denominators(self):
        app = self.make_app()
        app.selectbox(key="model_name").set_value("k-Nearest Neighbors").run()
        app.segmented_control(key="matrix_mode").set_value("% of actual class").run()
        self.assert_ok(app)
        self.assertIn("**67.5%**", app.button(key="cell_fn").label)
        self.assertIn("258 sessions", app.button(key="cell_fn").label)
        self.assertIn("**2.6%**", app.button(key="cell_fp").label)
        self.assertTrue(any("actual-class row total" in caption.value for caption in app.caption))

    def test_comparison_reproduces_all_metrics_and_sorts_each_selected_metric(self):
        app = self.make_app()
        self.navigate(app, "Compare Models")
        self.assertEqual(len(app.session_state["evaluations"]), 5)
        for key, label in ((key, label) for label, key in DISPLAY_METRICS.items()):
            with self.subTest(metric=key):
                app.selectbox(key="comparison_metric").set_value(key).run()
                self.assert_ok(app)
                table = self.comparison_table(app)
                self.assertEqual(len(table.rows), 5)
                self.assertEqual(set(table.headers), {"Model", "Rank", *DISPLAY_METRICS})
                expected_order = self.expected[key].sort_values(ascending=False, kind="stable").index.tolist()
                self.assertEqual([row["Model"] for row in table.records], expected_order)
                for row in table.records:
                    for display, column in DISPLAY_METRICS.items():
                        self.assertEqual(row[display], f"{self.expected.loc[row['Model'], column]:.4f}")
                self.assertEqual([row["Model"] for row, selected in zip(table.records, table.selected) if selected],
                                 ["Decision Tree"])
        app.selectbox(key="model_name").set_value("k-Nearest Neighbors").run()
        self.assert_ok(app)
        self.navigate(app, "Overview")
        self.assert_metrics(app, self.expected.loc["k-Nearest Neighbors"])

    def test_empty_upload_control_keeps_the_bundled_evaluation_available(self):
        app = self.make_app()
        identity = app.session_state["active_id"]
        self.upload_mode(app)
        self.assertEqual(app.session_state["active_id"], identity)
        self.assert_metrics(app, self.expected.loc["Decision Tree"])
        self.assertTrue(any("current valid dataset stays active" in message.value for message in app.info))

    def test_replacing_upload_resets_cached_comparisons_and_filter_context(self):
        app = self.make_app()
        self.upload_mode(app)
        first = self.test_frame.groupby("Revenue", group_keys=False).head(15).reset_index(drop=True)
        second = self.test_frame.groupby("Revenue", group_keys=False).tail(7).reset_index(drop=True)
        self.upload(app, first)
        first_identity = app.session_state["active_id"]
        self.navigate(app, "Compare Models")
        self.assertEqual(len(app.session_state["evaluations"]), 5)
        self.navigate(app, "Session Explorer")
        app.selectbox(key="outcome_filter").set_value("Missed buyers").run()
        self.assert_ok(app)
        second_payload = self.upload(app, second)
        self.assertNotEqual(app.session_state["active_id"], first_identity)
        self.assertEqual(app.selectbox(key="outcome_filter").value, "All sessions")
        self.assertEqual(len(app.session_state["evaluations"]), 1)
        self.assertEqual(self.matching_count(app), len(second))
        self.assertEqual(app.session_state["session_page"], 1)
        self.navigate(app, "Overview")
        self.assert_uploaded_metrics(app, second_payload)
        self.navigate(app, "Compare Models")
        self.assertEqual(len(app.session_state["evaluations"]), 5)
        table = self.comparison_table(app)
        normalized = read_dataset(second_payload)
        for row in table.records:
            metrics = evaluate_model(load_pipeline(row["Model"], ROOT), normalized)["metrics"]
            for label, key in DISPLAY_METRICS.items():
                self.assertEqual(row[label], "—" if metrics[key] is None else f"{metrics[key]:.4f}")

    def test_invalid_upload_retains_last_valid_results_and_recovers(self):
        app = self.make_app()
        self.upload_mode(app)
        payload = self.upload(app, self.test_frame.head(40), "accepted.csv")
        identity = app.session_state["active_id"]
        before = self.metrics(app)
        app.file_uploader(key="dataset_upload").set_value(("invalid.csv", b"wrong,headers\n1,2\n", "text/csv")).run()
        self.assert_ok(app)
        self.assertGreater(len(app.error), 0)
        self.assertTrue(any("last valid dataset" in error.value for error in app.error))
        self.assertEqual(app.session_state["active_id"], identity)
        self.assertEqual(app.session_state["dataset_name"], "accepted.csv")
        self.assertEqual(self.metrics(app), before)
        self.assert_uploaded_metrics(app, payload)
        self.upload(app, self.test_frame.tail(32), "corrected.csv")
        self.assertEqual(len(app.error), 0)
        self.assertEqual(app.session_state["dataset_name"], "corrected.csv")
        self.assertNotEqual(app.session_state["active_id"], identity)
        app.radio(key="dataset_source").set_value("Bundled test set").run()
        self.assert_ok(app)
        self.assert_metrics(app, self.expected.loc["Decision Tree"])

    def test_single_class_uploads_preserve_counts_and_unavailable_metrics(self):
        app = self.make_app()
        self.upload_mode(app)
        for outcome in (False, True):
            with self.subTest(outcome=outcome):
                self.navigate(app, "Overview")
                subset = self.test_frame.loc[self.test_frame["Revenue"] == outcome].head(20)
                payload = self.upload(app, subset)
                self.assert_uploaded_metrics(app, payload)
                self.assertEqual(self.metrics(app)["ROC AUC"], "—")
                self.assertEqual(self.metrics(app)["MCC"], "—")
                self.assertTrue(any("one outcome class" in message.value.lower() for message in app.info))
                self.navigate(app, "Compare Models")
                app.selectbox(key="comparison_metric").set_value("MCC").run()
                self.assert_ok(app)
                for row in self.comparison_table(app).records:
                    self.assertEqual(row["ROC AUC"], "—")
                    self.assertEqual(row["MCC"], "—")
                self.navigate(app, "Session Explorer")
                self.assertEqual(self.matching_count(app), len(subset))

    def test_feature_only_upload_has_predictions_without_fabricated_metrics(self):
        app = self.make_app()
        self.upload_mode(app)
        payload = self.upload(app, self.test_frame.head(12).drop(columns="Revenue"), "features.csv")
        self.assert_uploaded_metrics(app, payload)
        self.assertEqual(set(self.metrics(app).values()), {"—"})
        self.assertFalse(any(button.key == "cell_fn" for button in app.button))
        app.button(key="explore_unlabeled").click().run()
        self.assert_ok(app)
        self.assertEqual(self.matching_count(app), 12)
        rows = self.session_table(app).records
        self.assertTrue(all(row["Actual"] == "Unavailable" and row["Outcome"] == "Unlabeled" for row in rows))
        self.assertTrue(all(row["Purchase probability"] != "—" for row in rows))
        self.navigate(app, "Compare Models")
        self.assertTrue(any("Observed Revenue values are required" in message.value for message in app.info))

    def test_search_sort_pagination_reset_and_detail_keep_stable_session_ids(self):
        app = self.make_app()
        self.navigate(app, "Session Explorer")
        app.selectbox(key="page_size").set_value(10).run()
        first_ids = [row["Session"] for row in self.session_table(app).records]
        app.button(key="next_page").click().run()
        self.assert_ok(app)
        second_ids = [row["Session"] for row in self.session_table(app).records]
        self.assertFalse(set(first_ids) & set(second_ids))
        self.assertEqual(app.session_state["session_page"], 2)
        app.button(key="previous_page").click().run()
        self.assertEqual([row["Session"] for row in self.session_table(app).records], first_ids)
        app.selectbox(key="session_sort").set_value("PredictionScore").run()
        app.selectbox(key="sort_direction").set_value("Descending").run()
        self.assert_ok(app)
        rows = self.session_table(app).records
        scores = [float(row["Purchase probability"]) for row in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))
        identifier = rows[0]["Session"]
        app.text_input(key="session_search").set_value(identifier).run()
        self.assert_ok(app)
        self.assertEqual(self.matching_count(app), 1)
        self.assertEqual(self.session_table(app).records[0]["Session"], identifier)
        app.button(key="open_detail").click().run()
        self.assert_ok(app)
        self.assertEqual(app.session_state["detail_id"], identifier)
        details = [RenderedTable(element.value) for element in app.html if "Original features supplied" in element.value]
        self.assertEqual(len(details), 1)
        self.assertEqual(len(details[0].rows), 17)
        self.assertEqual(set(details[0].headers), {"Feature", "Column", "Value"})
        app.button(key="close_detail").click().run()
        self.assert_ok(app)
        self.assertFalse(any(button.key == "close_detail" for button in app.button))
        self.assertEqual(self.matching_count(app), 1)
        app.text_input(key="session_search").set_value("no-session-can-match-this-text").run()
        self.assert_ok(app)
        self.assertEqual(self.matching_count(app), 0)
        self.assertTrue(any("active dataset is valid" in message.value for message in app.info))
        app.button(key="reset_filters").click().run()
        self.assert_ok(app)
        self.assertEqual(self.matching_count(app), 2466)
        self.assertEqual(app.selectbox(key="outcome_filter").value, "All sessions")
        self.assertEqual(app.text_input(key="session_search").value, "")
        self.assertEqual(app.selectbox(key="session_sort").value, "OriginalRow")
        self.assertEqual(app.session_state["session_page"], 1)

    def test_model_switch_keeps_filter_and_exports_every_matching_row_not_just_page(self):
        app = self.make_app()
        app.selectbox(key="model_name").set_value("k-Nearest Neighbors").run()
        with patch("shopper_app.explorer.export_sessions", wraps=export_sessions) as export:
            app.button(key="cell_fn").click().run()
            self.assert_ok(app)
            self.assertEqual(self.matching_count(app), 258)
            exported = export.call_args.args[0]
            self.assertEqual(len(exported), 258)
            self.assertTrue(exported["ResultCategory"].eq("Missed buyers").all())
            self.assertTrue(exported["SessionID"].is_unique)
            self.assertEqual(len(self.session_table(app).rows), 25)
            self.assertEqual(app.download_button(key="download_predictions").label, "Export 258 filtered predictions")
            app.selectbox(key="model_name").set_value("Decision Tree").run()
            self.assert_ok(app)
            self.assertEqual(app.selectbox(key="outcome_filter").value, "Missed buyers")
            self.assertEqual(self.matching_count(app), 168)
            self.assertEqual(len(export.call_args.args[0]), 168)
            app.download_button(key="download_predictions").click().run()
            self.assert_ok(app)
            self.assertEqual(self.matching_count(app), 168)


if __name__ == "__main__":
    unittest.main()

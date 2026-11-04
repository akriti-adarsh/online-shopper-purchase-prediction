"""Exercise the app's model selection and upload states through Streamlit."""

from pathlib import Path
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from shopper_app.evaluation import evaluate_model, load_pipeline, read_dataset


ROOT = Path(__file__).resolve().parents[1]
DISPLAY_METRICS = {
    "Accuracy": "Accuracy",
    "ROC AUC": "AUC",
    "Precision": "Precision",
    "Recall": "Recall",
    "F1": "F1",
    "MCC": "MCC",
}


class StreamlitAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = pd.read_csv(ROOT / "results/model_comparison.csv", index_col="Model")
        cls.test_frame = pd.read_csv(ROOT / "test_data.csv")

    def make_app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        return app

    def upload(self, app, frame, filename="sessions.csv"):
        payload = frame.to_csv(index=False).encode("utf-8")
        # FileUploader.set_value is a supported AppTest API in Streamlit 1.66.
        app.file_uploader(key="dataset_upload").set_value((filename, payload, "text/csv")).run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        return payload

    def comparison_tables(self, app):
        tables = []
        for element in app.dataframe:
            table = element.value
            if not isinstance(table, pd.DataFrame):
                continue
            columns = {"AUC" if column == "ROC AUC" else column for column in table.columns}
            if len(table) == 5 and {"Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"} <= columns:
                tables.append(table)
        return tables

    def prediction_table(self, app):
        tables = [element.value for element in app.dataframe
                  if isinstance(element.value, pd.DataFrame)
                  and {"SessionRow", "PredictedRevenue", "PredictionScore", "Correct"} <= set(element.value.columns)]
        self.assertEqual(len(tables), 1, "The session explorer must display its prediction table.")
        return tables[0]

    def assert_uploaded_metrics(self, app, payload):
        selected = app.selectbox(key="model_name").value
        expected = evaluate_model(load_pipeline(selected, ROOT), read_dataset(payload))["metrics"]
        displayed = {metric.label: metric.value for metric in app.metric}
        for label, column in DISPLAY_METRICS.items():
            value = expected[column]
            self.assertEqual(displayed[label], "—" if value is None else f"{value:.4f}")

    def test_default_screen_contains_required_metrics_and_navigation(self):
        app = self.make_app()
        self.assertEqual(app.selectbox(key="model_name").value, "Decision Tree")
        self.assertEqual(app.radio(key="dataset_source").value, "Included test set")
        displayed = {metric.label: metric.value for metric in app.metric}
        self.assertTrue(set(DISPLAY_METRICS) <= set(displayed))
        primary_tabs = ["Overview", "Compare models", "Session explorer", "About"]
        self.assertEqual(
            [tab.label for tab in app.tabs if tab.label in primary_tabs], primary_tabs
        )

    def test_each_model_updates_all_six_metrics(self):
        app = self.make_app()
        for name, expected in self.expected.iterrows():
            with self.subTest(model=name):
                app.selectbox(key="model_name").set_value(name).run()
                self.assertEqual(len(app.exception), 0, str(app.exception))
                displayed = {metric.label: metric.value for metric in app.metric}
                for label, column in DISPLAY_METRICS.items():
                    self.assertEqual(displayed[label], f"{expected[column]:.4f}")

    def test_empty_upload_state_has_helpful_message_and_no_results(self):
        app = self.make_app()
        app.radio(key="dataset_source").set_value("Upload a CSV").run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertGreater(len(app.info), 0)
        self.assertEqual(len(app.metric), 0)

    def test_comparison_is_available_on_request(self):
        app = self.make_app()
        app.button(key="compare_models").click().run(timeout=60)
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertTrue(self.comparison_tables(app), "Requested comparison must include every model and required metric.")

    def test_replacing_upload_updates_results_and_resets_comparison(self):
        app = self.make_app()
        app.radio(key="dataset_source").set_value("Upload a CSV").run()
        first = self.test_frame.groupby("Revenue", group_keys=False).head(15).reset_index(drop=True)
        second = self.test_frame.groupby("Revenue", group_keys=False).tail(7).reset_index(drop=True)
        first_payload = self.upload(app, first)
        self.assert_uploaded_metrics(app, first_payload)
        app.button(key="compare_models").click().run(timeout=60)
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertTrue(app.session_state["comparison_ready"])
        self.assertTrue(self.comparison_tables(app))

        # Reusing the filename must still invalidate results when its bytes change.
        second_payload = self.upload(app, second)
        self.assertNotEqual(first_payload, second_payload)
        self.assertFalse(app.session_state["comparison_ready"])
        self.assertFalse(self.comparison_tables(app))
        self.assert_uploaded_metrics(app, second_payload)
        self.assertEqual(len(self.prediction_table(app)), len(second))

    def test_invalid_upload_hides_previous_results_and_can_recover(self):
        app = self.make_app()
        app.radio(key="dataset_source").set_value("Upload a CSV").run()
        self.upload(app, self.test_frame.head(40))
        self.assertEqual(len(app.metric), 6)
        app.file_uploader(key="dataset_upload").set_value(
            ("sessions.csv", b"wrong,headers\n1,2\n", "text/csv")
        ).run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertGreater(len(app.error), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.dataframe), 0)

        app.radio(key="dataset_source").set_value("Included test set").run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        displayed = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(displayed["F1"], f"{self.expected.loc['Decision Tree', 'F1']:.4f}")

    def test_single_class_uploads_show_unavailable_metrics_without_errors(self):
        app = self.make_app()
        app.radio(key="dataset_source").set_value("Upload a CSV").run()
        for outcome in (False, True):
            with self.subTest(outcome=outcome):
                subset = self.test_frame.loc[self.test_frame["Revenue"] == outcome].head(20)
                payload = self.upload(app, subset)
                self.assert_uploaded_metrics(app, payload)
                self.assertTrue(any("one outcome class" in message.value.lower() for message in app.info))
                self.assertEqual(len(self.prediction_table(app)), len(subset))
                app.button(key="compare_models").click().run(timeout=60)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                tables = self.comparison_tables(app)
                self.assertEqual(len(tables), 1)
                table = tables[0].rename(columns={"ROC AUC": "AUC"})
                self.assertTrue(table["AUC"].isna().all())
                self.assertTrue(table["MCC"].isna().all())

    def test_prediction_filters_match_each_models_actual_errors(self):
        app = self.make_app()
        expected_counts = pd.read_csv(ROOT / "results/error_counts.csv", index_col="Model")
        for name in ("Decision Tree", "Gaussian Naive Bayes"):
            app.selectbox(key="model_name").set_value(name).run()
            counts = expected_counts.loc[name]
            selections = {
                "All sessions": len(self.test_frame),
                "Correct predictions": counts["True negatives"] + counts["True positives"],
                "False purchase alerts": counts["False positives"],
                "Missed purchases": counts["False negatives"],
            }
            for selection, expected_count in selections.items():
                with self.subTest(model=name, selection=selection):
                    app.selectbox(key="prediction_filter").set_value(selection).run()
                    self.assertEqual(len(app.exception), 0, str(app.exception))
                    shown = self.prediction_table(app)
                    self.assertEqual(len(shown), expected_count)
                    self.assertTrue(shown["SessionRow"].is_unique)
                    if selection == "Correct predictions":
                        self.assertTrue(shown["Correct"].all())
                    elif selection == "False purchase alerts":
                        self.assertTrue((~shown["Revenue"] & shown["PredictedRevenue"]).all())
                    elif selection == "Missed purchases":
                        self.assertTrue((shown["Revenue"] & ~shown["PredictedRevenue"]).all())


if __name__ == "__main__":
    unittest.main()

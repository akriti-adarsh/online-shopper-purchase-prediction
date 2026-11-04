"""Protect the numerical meaning of curves and model-comparison figures."""

import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from shopper_app.charts import (
    confusion_chart,
    curve_chart,
    palette,
    performance_3d_chart,
    purchase_outcomes_chart,
    ranking_chart,
)


class PerformanceCurveTests(unittest.TestCase):
    actual = np.array([0, 0, 0, 1])
    scores = np.array([0.1, 0.4, 0.7, 0.9])
    matrix = np.array([[2, 1], [0, 1]])
    rule = {"threshold": 0.5, "comparison": ">", "fpr": 1 / 3,
            "recall": 1.0, "precision": 0.5}

    def chart(self, kind, **kwargs):
        return curve_chart(self.actual, self.scores, "Decision Tree", kind,
                           confusion=self.matrix, operating_point=self.rule, **kwargs)

    def test_saved_rule_coordinates_use_actual_confusion_counts(self):
        for kind, expected in (("roc", [1 / 3, 1.0]), ("pr", [1.0, 0.5])):
            with self.subTest(kind=kind):
                chart = self.chart(kind)
                point = next(trace for trace in chart.data if trace.name == "Saved decision rule")
                np.testing.assert_allclose([point.x[0], point.y[0]], expected)
                self.assertIn("&gt; 0.5", point.hovertemplate)

    def test_precision_recall_baseline_uses_observed_buyer_share(self):
        chart = self.chart("pr")
        np.testing.assert_allclose(chart.data[0].y, [0.25, 0.25])
        self.assertIn("25.00%", chart.data[0].name)
        precision, recall, _ = precision_recall_curve(self.actual, self.scores)
        np.testing.assert_array_equal(chart.data[1].x, recall)
        np.testing.assert_array_equal(chart.data[1].y, precision)

    def test_positive_label_mapping_does_not_depend_on_label_order(self):
        labels = np.array(["purchase", "no purchase", "no purchase", "purchase"])
        scores = np.array([0.8, 0.6, 0.2, 0.4])
        chart = curve_chart(labels, scores, "Decision Tree", positive_label="purchase")
        expected_x, expected_y, _ = roc_curve([1, 0, 0, 1], scores, drop_intermediate=False)
        np.testing.assert_array_equal(chart.data[1].x, expected_x)
        np.testing.assert_array_equal(chart.data[1].y, expected_y)

    def test_repeated_scores_keep_one_threshold_per_distinct_score(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        for kind in ("roc", "pr"):
            with self.subTest(kind=kind), warnings.catch_warnings():
                warnings.simplefilter("error")
                chart = curve_chart(self.actual, scores, "Decision Tree", kind)
                self.assertEqual(len(chart.data[1].x), 2)
                self.assertTrue(np.isfinite(chart.data[1].x).all())
                self.assertTrue(np.isfinite(chart.data[1].y).all())
                self.assertFalse(any(trace.name == "Saved decision rule" for trace in chart.data))

    def test_unverified_rule_never_receives_an_inferred_threshold_marker(self):
        chart = curve_chart(self.actual, self.scores, "Decision Tree", confusion=self.matrix)
        self.assertEqual(len(chart.data), 2)
        incorrect_counts = np.array([[1, 1], [1, 1]])
        chart = curve_chart(self.actual, self.scores, "Decision Tree",
                            confusion=incorrect_counts, operating_point=self.rule)
        self.assertEqual(len(chart.data), 2)

    def test_no_predicted_buyers_has_no_precision_operating_point(self):
        matrix = np.array([[3, 0], [1, 0]])
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        chart = curve_chart(self.actual, scores, "Decision Tree", "pr",
                            confusion=matrix, operating_point=self.rule)
        self.assertEqual(len(chart.data), 2)
        roc = curve_chart(self.actual, scores, "Decision Tree", "roc",
                          confusion=matrix, operating_point=self.rule)
        self.assertEqual(len(roc.data), 3)
        np.testing.assert_allclose([roc.data[2].x[0], roc.data[2].y[0]], [0, 0])

    def test_single_class_missing_scores_and_bad_inputs_show_explanations(self):
        cases = [([0, 0], [0.1, 0.2]), ([1, 1], [0.1, 0.2]),
                 (None, [0.1, 0.2]), ([0, 1], None), ([0, 1], [0.1]),
                 ([0, 1], [0.1, np.nan]), ([], [])]
        for actual, scores in cases:
            for kind in ("roc", "pr"):
                with self.subTest(actual=actual, scores=scores, kind=kind), warnings.catch_warnings():
                    warnings.simplefilter("error")
                    chart = curve_chart(actual, scores, "Decision Tree", kind)
                    self.assertEqual(len(chart.data), 0)
                    self.assertEqual(len(chart.layout.annotations), 1)


class ComparisonChartTests(unittest.TestCase):
    def setUp(self):
        self.comparison = pd.DataFrame({
            "Accuracy": [0.9, 0.9, 0.6], "AUC": [0.95, 0.8, 0.4],
            "Precision": [0.8, 0.7, 0.3], "Recall": [0.4, 0.7, 0.9],
            "F1": [0.5, 0.7, 0.4], "MCC": [-0.5, 0.7, None],
        }, index=["Logistic Regression", "Decision Tree", "Gaussian Naive Bayes"])

    def test_all_six_metrics_sort_the_selected_values_in_their_valid_range(self):
        for metric in self.comparison:
            with self.subTest(metric=metric):
                chart = ranking_chart(self.comparison, metric)
                expected = self.comparison[metric].dropna().sort_values(ascending=False, kind="stable")
                np.testing.assert_allclose(chart.data[0].x, expected.to_numpy())
                self.assertEqual(list(chart.data[0].y), list(expected.index))
                self.assertEqual(list(chart.layout.xaxis.range), [-1, 1] if metric == "MCC" else [0, 1])

    def test_tied_scores_share_rank_without_a_hidden_secondary_metric(self):
        chart = ranking_chart(self.comparison, "Accuracy")
        self.assertEqual([row[1] for row in chart.data[0].customdata], [1, 1, 3])
        self.assertEqual(list(chart.data[0].y), list(self.comparison.index))
        self.assertIn("Equal scores", chart.layout.annotations[0].text)

    def test_selection_has_label_and_color_and_does_not_change_values(self):
        chart = ranking_chart(self.comparison, "F1", "Decision Tree", theme="dark")
        self.assertIn("★", chart.layout.yaxis.ticktext[0])
        self.assertEqual(chart.data[0].marker.color[0], palette("dark")["primary"])
        self.assertEqual(chart.data[0].customdata[0][2], "Selected model")
        self.assertEqual(chart.data[0].x[0], 0.7)

    def test_unavailable_metric_is_not_imputed_to_zero(self):
        self.comparison["MCC"] = None
        chart = ranking_chart(self.comparison, "MCC")
        self.assertFalse(chart.data)
        self.assertIn("unavailable", chart.layout.annotations[0].text)

    def test_three_dimensions_keep_negative_mcc_and_omit_unavailable_results(self):
        chart = performance_3d_chart(self.comparison, "Decision Tree", theme="dark")
        self.assertEqual(len(chart.data), 2)
        self.assertEqual(chart.data[0].z[0], -0.5)
        self.assertEqual(list(chart.layout.scene.zaxis.range), [-1, 1])
        self.assertEqual(chart.data[1].marker.symbol, "diamond")
        self.assertEqual(chart.layout.scene.bgcolor, palette("dark")["surface"])

    def test_all_figures_theme_text_grid_and_hover_surfaces(self):
        matrix = np.array([[2, 1], [1, 2]])
        for theme in ("light", "dark"):
            colors = palette(theme)
            figures = [ranking_chart(self.comparison, theme=theme),
                       performance_3d_chart(self.comparison, theme=theme),
                       curve_chart([0, 1], [0.2, 0.8], "Decision Tree", theme=theme),
                       confusion_chart(matrix, theme=theme), purchase_outcomes_chart(matrix, theme=theme)]
            for chart in figures:
                with self.subTest(theme=theme, type=chart.data[0].type):
                    self.assertEqual(chart.layout.font.color, colors["text"])
                    self.assertEqual(chart.layout.paper_bgcolor, colors["surface"])
                    self.assertEqual(chart.layout.hoverlabel.bgcolor, colors["surface"])

    def test_percentage_matrix_exposes_missing_denominator(self):
        chart = confusion_chart([[3, 2], [0, 0]], mode="% of actual class")
        self.assertIn("60.0%", chart.data[0].text[0][0])
        self.assertIn("Unavailable", chart.data[0].text[1][0])


if __name__ == "__main__":
    unittest.main()

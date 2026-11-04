"""Responsive, themed figures derived from the current evaluation results."""

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

PALETTES = {
    "light": {
        "paper": "#F3F7FF", "surface": "#FFFFFF", "text": "#14213D",
        "muted": "#465873", "grid": "#D5E1F5", "primary": "#245BFF",
        "highlight": "#245BFF", "bar": "#A9C0FF", "on_color": "#FFFFFF",
        "identified": "#047857", "missed": "#A54700",
        "false_alerts": "#BB2450", "non_buyers": "#315CB2",
    },
    "dark": {
        "paper": "#0B1228", "surface": "#141F3B", "text": "#F4F8FF",
        "muted": "#B2C2DF", "grid": "#334A71", "primary": "#8AABFF",
        "highlight": "#315DF5", "bar": "#708DCE", "on_color": "#0B1734",
        "identified": "#47E5B1", "missed": "#FFD166",
        "false_alerts": "#FF7AA7", "non_buyers": "#9BB8FF",
    },
}
COLORS = {
    "Logistic Regression": "#245BFF",
    "Decision Tree": "#008A65",
    "k-Nearest Neighbors": "#DA5500",
    "Gaussian Naive Bayes": "#D52968",
    "Support Vector Machine": "#007FA3",
}
DARK_COLORS = {
    "Logistic Regression": "#8AABFF",
    "Decision Tree": "#45E5AF",
    "k-Nearest Neighbors": "#FFB547",
    "Gaussian Naive Bayes": "#FF7AA7",
    "Support Vector Machine": "#5AD5E8",
}
SHORT_NAMES = {
    "Logistic Regression": "Logistic<br>regression",
    "Decision Tree": "Decision tree",
    "k-Nearest Neighbors": "k-Nearest<br>neighbors",
    "Gaussian Naive Bayes": "Gaussian<br>naive Bayes",
    "Support Vector Machine": "Support vector<br>machine",
}
METRIC_LABELS = {
    "Accuracy": "Accuracy", "AUC": "ROC AUC", "Precision": "Precision",
    "Recall": "Recall", "F1": "F1", "MCC": "MCC",
}


def palette(theme="light"):
    """Return a copy of the resolved light or dark chart palette."""
    return PALETTES.get(str(theme).lower(), PALETTES["light"]).copy()


def model_color(model_name, theme="light"):
    colors = DARK_COLORS if str(theme).lower() == "dark" else COLORS
    return colors.get(model_name, palette(theme)["primary"])


def style(figure, height=400, *, theme="light"):
    """Apply legible axes and tooltips without fixing the container width."""
    colors = palette(theme)
    figure.update_layout(
        template="none", autosize=True, height=height,
        margin=dict(l=16, r=20, t=20, b=64),
        paper_bgcolor=colors["surface"], plot_bgcolor=colors["surface"],
        font=dict(family="Source Sans 3, Segoe UI, sans-serif", color=colors["text"], size=13),
        hoverlabel=dict(bgcolor=colors["surface"], bordercolor=colors["muted"],
                        font=dict(color=colors["text"], size=13)),
        legend=dict(orientation="v", y=-0.22, yanchor="top", x=0, xanchor="left",
                    font=dict(size=12, family="Source Sans 3, Segoe UI, sans-serif"), traceorder="normal"),
        hovermode="closest", dragmode=False,
    )
    figure.update_xaxes(gridcolor=colors["grid"], zeroline=False, automargin=True,
                       title_standoff=12, tickfont_color=colors["muted"],
                       tickfont_family="Source Sans 3, Segoe UI, sans-serif",
                       title_font_family="Source Sans 3, Segoe UI, sans-serif", fixedrange=True)
    figure.update_yaxes(gridcolor=colors["grid"], zeroline=False, automargin=True,
                       title_standoff=12, tickfont_color=colors["muted"],
                       tickfont_family="Source Sans 3, Segoe UI, sans-serif",
                       title_font_family="Source Sans 3, Segoe UI, sans-serif", fixedrange=True)
    return figure


def _notice(message, *, theme="light", height=350):
    figure = style(go.Figure(), height, theme=theme)
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, align="center", font_size=14)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def _counts(matrix):
    counts = np.asarray(matrix, dtype=float)
    if (counts.shape != (2, 2) or not np.isfinite(counts).all()
            or (counts < 0).any() or (counts % 1 != 0).any()):
        raise ValueError("Outcome counts must be a nonnegative, whole-number 2 by 2 matrix.")
    return counts.astype(np.int64)


def confusion_chart(matrix, *, mode="count", theme="light"):
    """Static chart alternative; application buttons provide outcome drill-down."""
    counts = _counts(matrix)
    colors = palette(theme)
    names = np.array([["Correct non-buyers", "False purchase alerts"],
                      ["Missed buyers", "Identified buyers"]])
    codes = np.array([["TN", "FP"], ["FN", "TP"]])
    totals = counts.sum(axis=1)
    percent = str(mode).lower() in {"percent", "percentage", "% of actual class"}
    text = []
    for row in range(2):
        values = []
        for column in range(2):
            value = f"{counts[row, column]:,}"
            if percent:
                value = f"{counts[row, column] / totals[row]:.1%}" if totals[row] else "Unavailable"
            values.append(f"<b>{value}</b><br>{names[row, column]}<br>{codes[row, column]}")
        text.append(values)
    shades = [colors[key] for key in ("non_buyers", "false_alerts", "missed", "identified")]
    scale = [[stop, color] for index, color in enumerate(shades)
             for stop in (index / 4, (index + 1) / 4)]
    figure = go.Figure(go.Heatmap(
        z=[[0, 1], [2, 3]], zmin=-0.5, zmax=3.5,
        x=["No purchase", "Purchase"], y=["No purchase", "Purchase"],
        colorscale=scale, showscale=False, text=text, texttemplate="%{text}",
        textfont=dict(size=15, color=colors["on_color"]), customdata=counts,
        xgap=6, ygap=6,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Sessions: %{customdata:,}<extra></extra>",
    ))
    style(figure, 360, theme=theme)
    figure.update_yaxes(autorange="reversed", title="Actual outcome", showgrid=False)
    figure.update_xaxes(title="Predicted outcome", showgrid=False)
    return figure


def purchase_outcomes_chart(matrix, *, theme="light"):
    """Show the different denominators behind buyer recall and purchase precision."""
    _, false_alerts, missed, identified = _counts(matrix).ravel()
    colors = palette(theme)
    totals = np.array([identified + missed, identified + false_alerts])
    figure = go.Figure()
    for name, values, key in (
        ("Identified buyers", [identified, identified], "identified"),
        ("Missed buyers", [missed, 0], "missed"),
        ("False purchase alerts", [0, false_alerts], "false_alerts"),
    ):
        values = np.asarray(values)
        fractions = np.divide(values, totals, out=np.zeros(2, dtype=float), where=totals != 0)
        figure.add_trace(go.Bar(
            name=name, x=fractions, y=["Actual buyers", "Predicted purchases"],
            orientation="h", marker_color=colors[key],
            text=[f"{value:,}" if fraction >= 0.14 else "" for value, fraction in zip(values, fractions)],
            textposition="inside", insidetextfont_color=colors["on_color"],
            customdata=np.column_stack((values, totals)),
            hovertemplate=name + ": %{customdata[0]:,}<br>Of %{customdata[1]:,} sessions (%{x:.1%})<extra></extra>",
        ))
    style(figure, 390, theme=theme)
    figure.update_layout(barmode="stack", bargap=0.48, margin=dict(b=120))
    figure.update_xaxes(title="Share of the stated group", range=[0, 1], tickformat=".0%")
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return figure


def _operating_coordinates(confusion, binary, kind):
    if confusion is None:
        return None
    try:
        matrix = _counts(confusion)
    except (ValueError, TypeError, OverflowError):
        return None
    tn, fp, fn, tp = matrix.ravel()
    if matrix.sum() != len(binary) or tp + fn != binary.sum():
        return None
    if not tp + fn:
        return None
    recall = float(tp / (tp + fn))
    if kind == "roc" and tn + fp:
        return float(fp / (tn + fp)), recall
    if kind == "pr" and tp + fp:
        return recall, float(tp / (tp + fp))
    return None


def curve_chart(y_true, scores, model_name, kind="roc", *, confusion=None,
                positive_label=True, operating_point=None, theme="light"):
    """Plot scores oriented toward positive_label, preserving the saved decisions.

    The inference layer supplies operating_point only after checking that a score
    rule reproduces every saved prediction. Coordinates still come from the actual
    confusion counts, including the original classifier's behavior at tied scores.
    """
    if kind not in {"roc", "pr"}:
        raise ValueError("Choose 'roc' or 'pr' for the performance curve.")
    if y_true is None:
        return _notice("Observed purchase outcomes are needed<br>to draw this performance curve.", theme=theme)
    if scores is None:
        return _notice("This model has no available purchase scores.<br>Its saved predictions are unchanged.", theme=theme)
    try:
        actual = np.asarray(y_true)
        values = np.asarray(scores, dtype=float)
        if actual.ndim != 1 or values.ndim != 1 or len(actual) != len(values) or not len(actual):
            return _notice("A score and observed outcome are needed<br>for every evaluated session.", theme=theme)
        if not np.isfinite(values).all():
            return _notice("Performance curves are unavailable<br>because some model scores are not finite.", theme=theme)
        binary = (actual == positive_label).astype(int)
        classes = np.unique(actual)
    except (ValueError, TypeError):
        return _notice("Performance curves are unavailable<br>for these model scores and outcomes.", theme=theme)
    if len(classes) > 2:
        return _notice("These purchase curves require<br>exactly two outcome classes.", theme=theme)
    if len(np.unique(binary)) < 2:
        return _notice("Include both buyers and non-buyers<br>to compare performance across thresholds.", theme=theme)

    colors = palette(theme)
    color = model_color(model_name, theme)
    if kind == "roc":
        x, y, thresholds = roc_curve(binary, values, pos_label=1, drop_intermediate=False)
        summary = f"Model AUC · {auc(x, y):.4f}"
        baseline_x, baseline_y = [0, 1], [0, 1]
        baseline_name = "Chance AUC · 0.50"
        x_label, y_label = "False positive rate", "Buyer recall"
        hover = "False positive rate: %{x:.2%}<br>Buyer recall: %{y:.2%}"
        shape = "linear"
    else:
        y, x, thresholds = precision_recall_curve(binary, values, pos_label=1)
        thresholds = np.append(thresholds, np.nan)
        summary = f"Model AP · {average_precision_score(binary, values):.4f}"
        prevalence = float(binary.mean())
        baseline_x, baseline_y = [0, 1], [prevalence, prevalence]
        baseline_name = f"Buyer share · {prevalence:.2%}"
        x_label, y_label = "Buyer recall", "Purchase precision"
        hover = "Buyer recall: %{x:.2%}<br>Purchase precision: %{y:.2%}"
        shape = "hv"
    threshold_labels = [f"{value:.6g}" if np.isfinite(value) else "Endpoint"
                        for value in thresholds]
    figure = go.Figure(go.Scatter(
        x=baseline_x, y=baseline_y, mode="lines", name=baseline_name,
        line=dict(color=colors["muted"], dash="dash", width=1.5),
        hovertemplate=baseline_name + "<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=summary,
        line=dict(color=color, width=3, shape=shape),
        customdata=threshold_labels,
        hovertemplate=hover + "<br>Score threshold: %{customdata}<extra></extra>",
    ))
    point = _operating_coordinates(confusion, binary, kind) if isinstance(operating_point, dict) else None
    if point is not None:
        threshold = operating_point.get("threshold")
        comparator = escape(str(operating_point.get("comparison", "")))
        rule = f"<br>Purchase when score {comparator} {threshold:g}" if isinstance(threshold, (int, float)) else ""
        figure.add_trace(go.Scatter(
            x=[point[0]], y=[point[1]], mode="markers", name="Saved decision rule",
            marker=dict(symbol="diamond", size=12, color=colors["primary"],
                        line=dict(color=colors["surface"], width=2)),
            hovertemplate="Saved decision rule<br>" + hover + rule + "<extra></extra>", cliponaxis=False,
        ))
    style(figure, 440, theme=theme)
    figure.update_layout(margin=dict(l=16, r=22, t=16, b=126),
                        meta=dict(kind=kind, positive_label=str(positive_label), model=model_name))
    axis = dict(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1], tickformat=".0%")
    figure.update_xaxes(title=x_label, **axis)
    figure.update_yaxes(title=y_label, **axis)
    return figure


def ranking_chart(comparison, metric="F1", selected_model=None, *, theme="light"):
    """Rank one metric with competition ranks: exact ties share the same rank."""
    metric = "AUC" if metric == "ROC AUC" else metric
    if metric not in METRIC_LABELS:
        raise ValueError("Choose one of the six evaluation metrics.")
    values = pd.to_numeric(comparison[metric], errors="coerce")
    valid = values.notna() & np.isfinite(values)
    ranked = values[valid].sort_values(ascending=False, kind="stable")
    if ranked.empty:
        return _notice(f"{METRIC_LABELS[metric]} is unavailable<br>for the selected dataset.", theme=theme)
    colors = palette(theme)
    ranks = ranked.rank(method="min", ascending=False).astype(int)
    labels = [f"{ranks[name]}. {SHORT_NAMES.get(name, escape(str(name)))}"
              + (" ★" if name == selected_model else "") for name in ranked.index]
    custom = [[name, int(ranks[name]), "Selected model" if name == selected_model else "Model"]
              for name in ranked.index]
    figure = go.Figure(go.Bar(
        x=ranked.to_numpy(), y=list(ranked.index), orientation="h",
        marker=dict(color=[colors["primary"] if name == selected_model else colors["bar"]
                           for name in ranked.index],
                    line=dict(color=colors["muted"], width=0.7)),
        text=[f"{value:.4f}" for value in ranked],
        textposition=["inside" if abs(value) >= 0.88 else "outside" for value in ranked],
        insidetextfont_color=[
            colors["text"] if str(theme).lower() != "dark" and name != selected_model else colors["on_color"]
            for name in ranked.index
        ], outsidetextfont_color=colors["text"],
        customdata=custom, cliponaxis=False,
        hovertemplate="%{customdata[0]}<br>" + METRIC_LABELS[metric]
                      + ": %{x:.4f}<br>Rank %{customdata[1]} · %{customdata[2]}<extra></extra>",
    ))
    style(figure, max(340, 68 * len(ranked) + 80), theme=theme)
    figure.update_layout(showlegend=False, bargap=0.38, margin=dict(l=135, r=32, t=12, b=76),
                        meta=dict(metric=metric, ranks={str(name): int(ranks[name]) for name in ranked.index}))
    figure.update_xaxes(title=f"{METRIC_LABELS[metric]} · higher is better",
                        range=[-1, 1] if metric == "MCC" else [0, 1],
                        dtick=0.5 if metric == "MCC" else 0.25,
                        zeroline=metric == "MCC", zerolinecolor=colors["muted"], zerolinewidth=1.5)
    figure.update_yaxes(autorange="reversed", showgrid=False, tickvals=list(ranked.index),
                        ticktext=labels, tickfont=dict(size=12), title=None)
    notes = []
    if ranked.duplicated().any():
        notes.append("Equal scores share a rank.")
    if not valid.all():
        notes.append(f"Unavailable results omitted: {int((~valid).sum())}.")
    if notes:
        figure.update_layout(margin=dict(b=76 + 14 * (len(notes) - 1)))
        figure.add_annotation(text="<br>".join(notes), x=0, y=0, xref="paper", yref="paper",
                              yshift=-58, yanchor="top", align="left", showarrow=False,
                              xanchor="left", font=dict(size=11, color=colors["muted"]))
    return figure


def performance_3d_chart(comparison, selected_model=None, *, theme="light"):
    """Optional spatial view; the comparison table remains the exact-value view."""
    colors = palette(theme)
    figure = go.Figure()
    for number, (name, row) in enumerate(comparison.iterrows(), 1):
        coordinates = pd.to_numeric(row[["Precision", "Recall", "MCC"]], errors="coerce")
        if not np.isfinite(coordinates.to_numpy(dtype=float)).all():
            continue
        selected = name == selected_model
        figure.add_trace(go.Scatter3d(
            x=[coordinates["Precision"]], y=[coordinates["Recall"]], z=[coordinates["MCC"]],
            mode="markers+text", name=f"{number}. {name}",
            marker=dict(size=11 if selected else 8, color=model_color(name, theme),
                        symbol="diamond" if selected else "circle",
                        line=dict(width=1, color=colors["surface"])),
            text=[str(number)], textposition="top center",
            hovertemplate=escape(str(name)) + "<br>Precision: %{x:.4f}<br>Recall: %{y:.4f}<br>MCC: %{z:.4f}<extra></extra>",
        ))
    if not figure.data:
        return _notice("The 3D comparison needs available<br>precision, recall, and MCC values.", theme=theme)
    style(figure, 590, theme=theme)
    scene_axis = dict(backgroundcolor=colors["surface"], gridcolor=colors["grid"],
                      zerolinecolor=colors["muted"], color=colors["muted"],
                      tickfont=dict(size=11), showbackground=True)
    figure.update_layout(
        scene=dict(
            xaxis=dict(title="Precision", range=[0, 1], **scene_axis),
            yaxis=dict(title="Recall", range=[0, 1], **scene_axis),
            zaxis=dict(title="MCC", range=[-1, 1], **scene_axis),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.95)),
            bgcolor=colors["surface"], aspectmode="cube",
        ),
        legend=dict(font_size=11, orientation="v", y=-0.03, yanchor="top"),
        margin=dict(l=0, r=0, t=8, b=150),
    )
    return figure

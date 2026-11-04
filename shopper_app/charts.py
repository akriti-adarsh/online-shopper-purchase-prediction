"""Interactive figures for the currently selected evaluation dataset."""

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve


COLORS = {
    "Logistic Regression": "#3979E8",
    "Decision Tree": "#0B9D91",
    "k-Nearest Neighbors": "#D99114",
    "Gaussian Naive Bayes": "#DC4C93",
    "Support Vector Machine": "#8254D8",
}
INK = "#172B49"
MUTED = "#61738B"


def style(figure, height=360):
    figure.update_layout(
        height=height, margin=dict(l=15, r=20, t=20, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color=INK, size=12),
        hoverlabel=dict(bgcolor="white", font_color=INK),
        legend=dict(orientation="h", y=-0.17, x=0),
    )
    figure.update_xaxes(gridcolor="#E4EAF2", zeroline=False)
    figure.update_yaxes(gridcolor="#E4EAF2", zeroline=False)
    return figure


def confusion_chart(matrix):
    labels = ["No purchase", "Purchase"]
    codes = np.array([["TN", "FP"], ["FN", "TP"]])
    text = np.array([[f"{int(matrix[r, c]):,}<br>{codes[r, c]}" for c in range(2)]
                     for r in range(2)])
    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels, text=text, texttemplate="%{text}",
        textfont=dict(size=22), colorscale=[[0, "#EDF7F7"], [0.5, "#75CEC4"], [1, "#153D62"]],
        zmin=0, zmax=max(int(matrix.max()), 1), showscale=False, xgap=8, ygap=8,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Sessions: %{z:,}<extra></extra>",
    ))
    style(fig, 330)
    fig.update_yaxes(autorange="reversed", title="Actual outcome", showgrid=False)
    fig.update_xaxes(title="Predicted outcome", showgrid=False)
    return fig


def purchase_outcomes_chart(matrix):
    _, false_alerts, missed, caught = matrix.ravel()
    fig = go.Figure()
    for label, counts, color in [
        ("Buyers caught", [caught, 0], "#0B9D91"),
        ("Buyers missed", [missed, 0], "#EBA52C"),
        ("False purchase alerts", [0, false_alerts], "#DC4C93"),
    ]:
        fig.add_trace(go.Bar(
            name=label, x=counts, y=["Actual purchases", "False alerts"], orientation="h",
            marker_color=color, text=[str(int(value)) if value else "" for value in counts],
            textposition="auto", hovertemplate=label + ": %{x:,}<extra></extra>",
        ))
    style(fig, 330)
    fig.update_layout(barmode="stack", bargap=0.48, legend=dict(font_size=11))
    fig.update_xaxes(title="Number of sessions", rangemode="tozero")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def curve_chart(y_true, scores, model_name, kind="roc"):
    color = COLORS[model_name]
    if kind == "roc":
        x, y, _ = roc_curve(y_true, scores)
        trace = go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=3),
                           name=model_name, hovertemplate="False positive rate: %{x:.3f}<br>Recall: %{y:.3f}<extra></extra>")
        fig = go.Figure(trace)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance ranking",
                                 line=dict(color="#A1ADBD", dash="dash")))
        x_label, y_label = "False positive rate", "True positive rate (recall)"
    else:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        average = average_precision_score(y_true, scores)
        fig = go.Figure(go.Scatter(
            x=recall, y=precision, mode="lines", line=dict(color=color, width=2.5, shape="hv"),
            name=f"Average precision: {average:.4f}",
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        ))
        prevalence = float(np.mean(y_true))
        fig.add_trace(go.Scatter(x=[0, 1], y=[prevalence, prevalence], mode="lines",
                                 name=f"Purchase prevalence: {prevalence:.2%}",
                                 line=dict(color="#A1ADBD", dash="dash")))
        x_label, y_label = "Recall", "Precision"
    style(fig, 360)
    fig.update_xaxes(range=[0, 1], title=x_label)
    fig.update_yaxes(range=[0, 1.02], title=y_label)
    return fig


def ranking_chart(comparison):
    ranked = comparison.sort_values(["F1", "MCC"], ascending=True, na_position="first")
    fig = go.Figure(go.Bar(
        x=ranked["F1"], y=ranked.index, orientation="h",
        marker_color=[COLORS[name] for name in ranked.index],
        text=[f"{value:.4f}" for value in ranked["F1"]], textposition="outside",
        hovertemplate="%{y}<br>Purchase F1: %{x:.4f}<extra></extra>", cliponaxis=False,
    ))
    style(fig, 400)
    fig.update_layout(margin=dict(l=0, r=30, t=20, b=35))
    fig.update_xaxes(title="Purchase F1", range=[0, 1.08])
    fig.update_yaxes(showgrid=False, automargin=True)
    return fig


def performance_3d_chart(comparison):
    fig = go.Figure()
    for number, (name, row) in enumerate(comparison.iterrows(), 1):
        fig.add_trace(go.Scatter3d(
            x=[row["Precision"]], y=[row["Recall"]], z=[row["MCC"]], mode="markers+text",
            marker=dict(size=9, color=COLORS[name], line=dict(width=1, color="white")),
            text=[str(number)], textposition="top center", name=name,
            hovertemplate=name + "<br>Precision: %{x:.4f}<br>Recall: %{y:.4f}<br>MCC: %{z:.4f}<extra></extra>",
        ))
    style(fig, 460)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Precision", range=[0, 1], backgroundcolor="#F1F5FC"),
            yaxis=dict(title="Recall", range=[0, 1], backgroundcolor="#F1F5FC"),
            zaxis=dict(title="MCC", range=[-1, 1], backgroundcolor="#F1F5FC"),
            camera=dict(eye=dict(x=1.5, y=1.4, z=0.9)),
        ),
        legend=dict(font_size=10, orientation="h", y=-0.05),
        margin=dict(l=0, r=0, t=0, b=60),
    )
    return fig

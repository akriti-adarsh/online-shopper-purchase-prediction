"""Consistent, publication-friendly figures for the purchase prediction analysis.

The three-dimensional views supplement exact tables and conventional plots.
All figure functions return a Figure; the caller controls display and export.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve


MODEL_NAMES = [
    "Logistic Regression", "Decision Tree", "k-Nearest Neighbors",
    "Gaussian Naive Bayes", "Support Vector Machine",
]
COLORS = {
    "Logistic Regression": "#3979E8",
    "Decision Tree": "#0B9D91",
    "k-Nearest Neighbors": "#EBA52C",
    "Gaussian Naive Bayes": "#DC4C93",
    "Support Vector Machine": "#8254D8",
}
SHORT_NAMES = {
    "Logistic Regression": "Logistic Regression", "Decision Tree": "Decision Tree",
    "k-Nearest Neighbors": "k-Nearest Neighbors", "Gaussian Naive Bayes": "Gaussian Naive Bayes",
    "Support Vector Machine": "Support Vector Machine",
}
INK = "#142B49"
MUTED = "#61738B"
CANVAS = "#F5F7FC"
GRID = "#DEE5EF"
OUTCOME_COLORS = ["#3979E8", "#DC4C93", "#EBA52C", "#0B9D91"]


def _canvas(title, subtitle, number, size=(13, 7.2)):
    fig = plt.figure(figsize=size, facecolor=CANVAS)
    fig.text(0.045, 0.955, f"MODEL STUDY  /  {number}", color=COLORS["Decision Tree"],
             size=9, weight="bold", va="top", family="DejaVu Sans")
    fig.text(0.045, 0.911, title, color=INK, size=22, weight="bold", va="top",
             family="DejaVu Sans")
    fig.text(0.045, 0.853, subtitle, color=MUTED, size=10.5, va="top",
             family="DejaVu Sans")
    return fig


def _footer(fig, text):
    fig.text(0.045, 0.024, text, color=MUTED, size=8.5, family="DejaVu Sans")


def _card(fig, bounds, color="white"):
    patch = FancyBboxPatch((bounds[0], bounds[1]), bounds[2], bounds[3],
                          boxstyle="round,pad=0.012,rounding_size=0.016",
                          transform=fig.transFigure, facecolor=color,
                          edgecolor="none", zorder=-1)
    fig.add_artist(patch)


def _axes_style(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(alpha=1, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _score_rail(ax, values, metric, best_name=None):
    ax.axis("off")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(0.03, 0.96, metric.upper(), color=MUTED, size=9, weight="bold",
            transform=ax.transAxes)
    for i, (name, value) in enumerate(values):
        y = 0.86 - i * 0.17
        if name == best_name:
            ax.add_patch(FancyBboxPatch((0, y - 0.083), 1, 0.145,
                         boxstyle="round,pad=0.012,rounding_size=0.025",
                         transform=ax.transAxes, facecolor="#E2F5F0", edgecolor="none"))
        ax.scatter([0.04], [y], s=65, color=COLORS[name], transform=ax.transAxes)
        ax.text(0.10, y + 0.006, SHORT_NAMES[name], color=INK, size=10,
                va="center", transform=ax.transAxes)
        ax.text(0.10, y - 0.047, f"{value:.4f}", color=COLORS[name], size=17,
                weight="bold", va="center", transform=ax.transAxes)


def class_balance_figure(counts):
    """Show class imbalance with a labeled ring and exact counts."""
    no_purchase, purchase = int(counts.loc[0]), int(counts.loc[1])
    total = no_purchase + purchase
    share = purchase / total
    fig = _canvas("The purchase class is the minority",
                  "Source data distribution  |  A high accuracy score can still miss most buyers.",
                  "01", (13, 6.6))
    _card(fig, (0.04, 0.12, 0.41, 0.66))
    _card(fig, (0.48, 0.12, 0.47, 0.66))
    ax = fig.add_axes([0.065, 0.20, 0.36, 0.49])
    ax.pie([purchase, no_purchase], startangle=90, counterclock=False,
           colors=[COLORS["Decision Tree"], "#DCE6F7"],
           wedgeprops={"width": 0.20, "edgecolor": "white", "linewidth": 5})
    ax.text(0, 0.06, f"{share:.2%}", ha="center", va="center", size=33,
            color=INK, weight="bold")
    ax.text(0, -0.20, "sessions end in a purchase", ha="center", color=MUTED, size=10)
    ax.set_aspect("equal")
    ax2 = fig.add_axes([0.52, 0.26, 0.37, 0.35])
    _axes_style(ax2)
    bars = ax2.barh([1, 0], [no_purchase, purchase], height=0.40,
                    color=[COLORS["Logistic Regression"], COLORS["Decision Tree"]])
    for bar, value, label in zip(bars, [no_purchase, purchase], ["No purchase", "Purchase"]):
        ax2.text(0, bar.get_y() + 0.55, label,
                 color=INK, size=11, weight="bold")
        ax2.text(value + total * 0.018, bar.get_y() + bar.get_height() / 2,
                 f"{value:,}", color=INK, va="center", size=12, weight="bold")
    ax2.set_xlim(0, total * 1.08)
    ax2.set_ylim(-0.4, 1.85)
    ax2.set_yticks([])
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("Number of sessions", labelpad=10)
    fig.text(0.52, 0.715, f"{total:,}", size=28, weight="bold", color=INK)
    fig.text(0.69, 0.728, "recorded sessions", size=12, color=MUTED)
    _footer(fig, "Positive class: Revenue = True. Counts use the complete source dataset.")
    return fig


def comparison_figure(comparison):
    """Pair an exact F1 ranking with the precision-recall operating points."""
    ranked = comparison.sort_values(["F1", "MCC"], ascending=False)
    fig = _canvas("One comparison, two useful perspectives",
                  "Purchase F1 identifies the strongest balance; precision and recall expose the tradeoff.", "02")
    _card(fig, (0.04, 0.13, 0.49, 0.65))
    _card(fig, (0.56, 0.13, 0.39, 0.65))
    left = fig.add_axes([0.205, 0.235, 0.285, 0.45])
    _axes_style(left)
    values = ranked["F1"].to_numpy()
    ypos = np.arange(len(ranked))
    left.barh(ypos, np.ones(len(ranked)), color="#EEF2F8", height=0.50)
    left.barh(ypos, values, color=[COLORS[name] for name in ranked.index], height=0.50)
    for y, value in zip(ypos, values):
        left.text(value + 0.025, y, f"{value:.4f}", va="center", color=INK, weight="bold", size=10)
    left.set(yticks=ypos, yticklabels=ranked.index, xlim=(0, 1), xlabel="Purchase F1")
    left.invert_yaxis()
    left.tick_params(axis="y", labelsize=9)
    left.grid(axis="y", visible=False)
    left.xaxis.set_major_locator(MultipleLocator(0.25))
    fig.text(0.065, 0.727, "Ranked by purchase F1", color=INK, weight="bold", size=13)
    right = fig.add_axes([0.61, 0.235, 0.30, 0.45])
    _axes_style(right)
    for number, name in enumerate(MODEL_NAMES, start=1):
        row = comparison.loc[name]
        right.scatter(row["Recall"], row["Precision"], color=COLORS[name], s=160,
                       edgecolors="white", linewidths=2, zorder=3)
        right.annotate(str(number), (row["Recall"], row["Precision"]),
                        xytext=(7, 8 if name != "k-Nearest Neighbors" else -17),
                        textcoords="offset points", color=COLORS[name], weight="bold", size=10)
    right.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    right.xaxis.set_major_locator(MultipleLocator(0.25))
    right.yaxis.set_major_locator(MultipleLocator(0.25))
    fig.text(0.585, 0.727, "Default decision rules", color=INK, weight="bold", size=13)
    names = "   ".join(f"{i} {name}" for i, name in enumerate(["LR", "Tree", "kNN", "NB", "SVM"], 1))
    fig.text(0.60, 0.165, names, color=MUTED, size=8.5)
    _footer(fig, "All five models use the same 2,466 test sessions. Every axis starts at zero; higher scores are better.")
    return fig


def confusion_figure(matrices):
    """Five exact confusion matrices with a single shared count color scale."""
    fig = _canvas("Where each model gets the decision right or wrong",
                  "Actual outcomes are rows; predicted outcomes are columns. Each panel contains all 2,466 test sessions.",
                  "04", (14, 9.3))
    maximum = max(int(matrix.max()) for matrix in matrices.values())
    cmap = LinearSegmentedColormap.from_list("count_scale", ["#F0F6FA", "#80D5CD", "#123C62"])
    positions = [(0.07 + col * 0.31, 0.48 if row == 0 else 0.11)
                 for row in range(2) for col in range(3)]
    for (name, matrix), (x, y) in zip(matrices.items(), positions):
        _card(fig, (x - 0.025, y - 0.035, 0.275, 0.335))
        ax = fig.add_axes([x + 0.035, y + 0.015, 0.17, 0.235])
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=maximum)
        for (row, col), count in np.ndenumerate(matrix):
            text_color = "white" if count > maximum * 0.60 else INK
            label = [["TN", "FP"], ["FN", "TP"]][row][col]
            ax.text(col, row - 0.08, f"{count:,}", ha="center", va="center",
                    color=text_color, size=17, weight="bold")
            ax.text(col, row + 0.20, label, ha="center", va="center", color=text_color, size=8)
        ax.set_xticks([0, 1], ["No purchase", "Purchase"], fontsize=8, color=MUTED)
        ax.set_yticks([0, 1], ["No purchase", "Purchase"], fontsize=8, color=MUTED)
        ax.tick_params(length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Predicted", fontsize=8, color=MUTED, labelpad=7)
        ax.set_ylabel("Actual", fontsize=8, color=MUTED, labelpad=7)
        fig.text(x - 0.006, y + 0.275, name, color=COLORS[name], size=12, weight="bold")
    legend = fig.add_axes([0.70, 0.14, 0.24, 0.25])
    legend.axis("off")
    legend.text(0, 0.92, "Read the four outcomes", color=INK, size=13, weight="bold")
    for i, text in enumerate(["TN  Correctly rejected nonpurchase", "FP  False purchase alert",
                             "FN  Missed buyer", "TP  Correctly identified buyer"]):
        legend.text(0, 0.70 - i * 0.16, text, color=MUTED, size=9.5)
    color_ax = fig.add_axes([0.70, 0.102, 0.22, 0.014])
    fig.colorbar(image, cax=color_ax, orientation="horizontal")
    color_ax.tick_params(labelsize=8, colors=MUTED, length=0)
    color_ax.set_title("Shared color scale: number of sessions", fontsize=8, color=MUTED, loc="left", pad=9)
    _footer(fig, "The same color represents the same count across every model. Numeric labels retain the exact results.")
    return fig


def _curve_canvas(title, subtitle, number):
    fig = _canvas(title, subtitle, number)
    _card(fig, (0.04, 0.12, 0.60, 0.66))
    _card(fig, (0.67, 0.12, 0.28, 0.66))
    ax = fig.add_axes([0.095, 0.225, 0.50, 0.49])
    _axes_style(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1.02))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    rail = fig.add_axes([0.685, 0.17, 0.25, 0.55])
    return fig, ax, rail


def roc_figure(y_test, scores, comparison):
    """ROC curves without smoothing and a separate numerical score rail."""
    fig, ax, rail = _curve_canvas("How well do the models rank purchase sessions?",
        "ROC curves compare true-positive and false-positive rates across score thresholds.", "06")
    winner = comparison["AUC"].idxmax()
    for name, score in scores.items():
        fpr, tpr, _ = roc_curve(y_test, score)
        ax.plot(fpr, tpr, color=COLORS[name], lw=3 if name == winner else 2.2, alpha=0.96)
        if name == winner:
            ax.fill_between(fpr, tpr, color=COLORS[name], alpha=0.06)
    ax.plot([0, 1], [0, 1], color="#96A2B4", linestyle=(0, (4, 4)), lw=1.4)
    ax.text(0.57, 0.47, "Chance ranking", rotation=35, color=MUTED, fontsize=9)
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)")
    _score_rail(rail, [(name, comparison.loc[name, "AUC"]) for name in MODEL_NAMES],
                "ROC AUC · higher is better", winner)
    _footer(fig, "Purchase probabilities are used where available; the SVM uses decision scores. Test set: 2,466 sessions.")
    return fig


def _style_3d(ax):
    ax.set_facecolor("white")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.95, 0.97, 1, 0.6))
        axis.pane.set_edgecolor("white")
        axis._axinfo["grid"].update({"color": GRID, "linewidth": 0.6})
        axis.line.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, pad=1)


def landscape_3d_figure(comparison):
    """Plot precision, recall, and MCC as three real coordinates for each model."""
    fig = _canvas("Five models in a three-dimensional performance space",
        "Each point is one fitted model: precision on x, recall on y, and MCC on z. Dashed stems locate points above the floor.", "03", (14, 8))
    _card(fig, (0.04, 0.11, 0.58, 0.67))
    _card(fig, (0.65, 0.11, 0.30, 0.67))
    ax = fig.add_axes([0.04, 0.12, 0.57, 0.64], projection="3d", computed_zorder=False)
    _style_3d(ax)
    ax.view_init(elev=24, azim=-56)
    ax.set_box_aspect((1.05, 1.05, 0.9))
    for i, name in enumerate(MODEL_NAMES, 1):
        row = comparison.loc[name]
        x, y, z = row["Precision"], row["Recall"], row["MCC"]
        color = COLORS[name]
        ax.plot([x, x], [y, y], [0, z], color=color, linestyle=(0, (2, 3)), alpha=0.7, lw=1.5, zorder=2)
        ax.scatter([x], [y], [0], s=65, facecolors="white", edgecolors=color, linewidth=1.5, zorder=3)
        ax.scatter([x], [y], [z], s=220, color=color, edgecolors="white", linewidth=1.8, depthshade=False, zorder=4)
        ax.text(x, y, z, str(i), size=9, weight="bold", color="white",
                ha="center", va="center", zorder=5)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xlabel="Precision", ylabel="Recall", zlabel="MCC")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_zticks([0, 0.5, 1])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color(INK)
        axis.label.set_size(10)
    rail = fig.add_axes([0.67, 0.14, 0.26, 0.60])
    rail.axis("off")
    rail.text(0, 0.98, "METRIC COORDINATES", color=MUTED, size=9, weight="bold")
    for i, name in enumerate(MODEL_NAMES, 1):
        y = 0.87 - (i - 1) * 0.17
        rail.text(0, y, str(i), color="white", size=9, weight="bold",
                  bbox={"boxstyle": "circle,pad=0.35", "fc": COLORS[name], "ec": "none"})
        rail.text(0.10, y + 0.005, name, size=10, color=INK, weight="bold")
        row = comparison.loc[name]
        rail.text(0.10, y - 0.065, f"P {row['Precision']:.4f}    R {row['Recall']:.4f}    MCC {row['MCC']:.4f}",
                  size=8.5, color=MUTED)
    _footer(fig, "Perspective helps explore tradeoffs; the labeled coordinates and six-metric table support numerical comparisons. All observed MCC values are positive.")
    return fig


def confusion_3d_figure(matrices):
    """Add a true 3D view of the four outcomes, keeping one common height scale."""
    fig = _canvas("The same prediction outcomes, viewed in three dimensions",
        "Column height is the number of sessions. Fixed outcome colors and exact labels are shared across every model.", "05", (14, 10))
    maximum = max(int(matrix.max()) for matrix in matrices.values())
    ztop = int(np.ceil(maximum / 500) * 500)
    for index, (name, matrix) in enumerate(matrices.items()):
        col, row = index % 3, index // 3
        x, y = 0.045 + col * 0.315, 0.465 if row == 0 else 0.10
        _card(fig, (x, y, 0.29, 0.335))
        ax = fig.add_axes([x + 0.005, y + 0.022, 0.28, 0.285], projection="3d")
        _style_3d(ax)
        ax.view_init(elev=25, azim=-48)
        ax.set_box_aspect((1, 1, 0.85))
        xpos = np.array([0, 1, 0, 1], dtype=float)
        ypos = np.array([0, 0, 1, 1], dtype=float)
        heights = matrix.ravel()
        ax.bar3d(xpos, ypos, np.zeros(4), 0.44, 0.44, heights,
                 color=OUTCOME_COLORS, shade=True, edgecolor="white", linewidth=0.6, alpha=0.96)
        for xx, yy, count in zip(xpos, ypos, heights):
            ax.text(xx + 0.22, yy + 0.22, int(count) + ztop * 0.035, f"{int(count):,}",
                    ha="center", color=INK, size=9, weight="bold", zorder=10)
        ax.set(xlim=(-0.15, 1.7), ylim=(-0.15, 1.7), zlim=(0, ztop))
        ax.set_xticks([0.22, 1.22], ["No purchase", "Purchase"])
        ax.set_yticks([0.22, 1.22], ["No purchase", "Purchase"])
        ax.set_zticks([0, 1000, 2000])
        ax.tick_params(axis="x", labelsize=6.5, pad=0)
        ax.tick_params(axis="y", labelsize=6.5, pad=0)
        ax.tick_params(axis="z", labelsize=7, pad=0)
        ax.set_xlabel("Predicted", labelpad=1, fontsize=7, color=MUTED)
        ax.set_ylabel("Actual", labelpad=1, fontsize=7, color=MUTED)
        fig.text(x + 0.012, y + 0.315, name, color=COLORS[name], size=11, weight="bold")
    _card(fig, (0.675, 0.10, 0.29, 0.335))
    legend = fig.add_axes([0.70, 0.135, 0.24, 0.26])
    legend.axis("off")
    legend.set(xlim=(0, 1), ylim=(0, 1))
    legend.text(0, 0.94, "Every column has a role", size=13, color=INK, weight="bold",
                transform=legend.transAxes)
    for index, (label, color) in enumerate(zip(
            ["TN  Correct nonpurchase", "FP  False purchase alert", "FN  Missed buyer", "TP  Correct purchase"], OUTCOME_COLORS)):
        y = 0.72 - index * 0.15
        legend.scatter([0.035], [y], s=95, color=color, marker="s", transform=legend.transAxes)
        legend.text(0.11, y, label, va="center", size=10, color=INK, transform=legend.transAxes)
    legend.text(0, 0.02, "Height scale: identical in all five panels\nThe 2D matrices retain the clearest exact view.",
                size=9, color=MUTED, linespacing=1.7, transform=legend.transAxes)
    _footer(fig, "Shared vertical axis: session count, starting at zero. The 3D view supplements the labeled 2D matrices.")
    return fig


def precision_recall_figure(y_test, scores):
    """Add purchase-class precision-recall curves and average precision."""
    fig, ax, rail = _curve_canvas("A closer look at the minority purchase class",
        "Precision-recall curves show how purchase-alert quality changes as more buyers are identified.", "07")
    averages = {name: average_precision_score(y_test, score) for name, score in scores.items()}
    winner = max(averages, key=averages.get)
    for name, score in scores.items():
        precision, recall, _ = precision_recall_curve(y_test, score)
        ax.step(recall, precision, where="post", color=COLORS[name],
                 lw=2.6 if name == winner else 1.8, alpha=0.95)
    prevalence = float(np.mean(y_test))
    ax.axhline(prevalence, color="#96A2B4", linestyle=(0, (4, 4)), lw=1.4)
    ax.text(0.03, prevalence + 0.025, f"Purchase prevalence: {prevalence:.2%}", color=MUTED, size=9)
    ax.set(xlabel="Recall", ylabel="Precision")
    _score_rail(rail, [(name, averages[name]) for name in MODEL_NAMES],
                "Average precision (AP)", winner)
    _footer(fig, "Average precision is a supplemental ranking summary; it does not replace any of the six required evaluation metrics.")
    return fig

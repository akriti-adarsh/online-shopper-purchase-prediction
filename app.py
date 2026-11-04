"""Explore saved purchase classifiers on a shared, labeled evaluation dataset."""

from hashlib import sha256
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from shopper_app.charts import (
    COLORS, confusion_chart, curve_chart, performance_3d_chart,
    purchase_outcomes_chart, ranking_chart,
)
from shopper_app.evaluation import (
    DataValidationError, FEATURE_COLUMNS, MODEL_DESCRIPTIONS, MODEL_FILES,
    dataset_warnings, evaluate_model, load_pipeline, read_dataset,
)


ROOT = Path(__file__).resolve().parent
REPOSITORY = "https://github.com/akriti-adarsh/online-shopper-purchase-prediction"
CHART_CONFIG = {"displaylogo": False, "scrollZoom": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

st.set_page_config(page_title="Shopper Signals | Purchase Prediction", page_icon="🛍️", layout="wide")
st.markdown("""
<style>
.block-container {max-width:1480px;padding-top:4.5rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {border-right:1px solid #e2e9f2;}
[data-testid="stSidebar"] .block-container {padding-top:1.5rem;}
.brand {display:flex;align-items:center;gap:12px;margin-bottom:26px;}
.brand-mark {background:#087f8c;color:#fff;border-radius:12px;font-weight:800;
             width:42px;height:42px;display:grid;place-items:center;font-size:20px;}
.brand-name {font-size:20px;font-weight:750;color:#172b49;line-height:1.15;}
.brand-sub {font-size:11px;color:#6a7c93;letter-spacing:1.6px;margin-top:4px;}
.hero {background:linear-gradient(118deg,#132b49 0%,#1a4561 70%,#126d75 100%);
       border-radius:22px;padding:30px 34px;color:#fff;position:relative;margin-bottom:22px;}
.hero .eyebrow {font-size:11px;letter-spacing:2px;font-weight:750;color:#79e1d0;margin-bottom:10px;}
.hero h1 {color:#fff;font-size:34px;line-height:1.15;letter-spacing:-.8px;margin:0 0 12px;padding:0;}
.hero p {color:#d9e6f0;font-size:15px;line-height:1.6;max-width:760px;margin:0;}
.hero-pills {display:flex;gap:9px;margin-top:22px;flex-wrap:wrap;}
.hero-pills span {padding:5px 11px;background:#ffffff12;border:1px solid #ffffff22;
                  border-radius:20px;font-size:11px;color:#e3f2f7;letter-spacing:.3px;}
.dataset-strip {display:flex;gap:8px 24px;align-items:center;flex-wrap:wrap;background:white;
                border:1px solid #e4eaf2;border-radius:12px;padding:13px 18px;margin-bottom:18px;
                color:#65788e;font-size:13px;}
.dataset-strip strong {color:#172b49;font-weight:700;}
.model-heading {display:flex;align-items:center;gap:10px;font-size:21px;
                font-weight:750;letter-spacing:-.4px;margin:8px 0;}
.model-dot {width:10px;height:10px;border-radius:50%;display:inline-block;}
[data-testid="stMetric"] {background:white;border:1px solid #e3eaf2;border-radius:13px;
                         padding:15px 17px 14px;box-shadow:0 3px 10px #172b4903;}
[data-testid="stMetricValue"] {font-size:28px;font-weight:700;letter-spacing:-.6px;}
[data-testid="stMetricLabel"] {color:#62758c;font-size:12px;}
.st-key-metric_grid [data-testid="stHorizontalBlock"] {display:grid;
    grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;}
.st-key-metric_grid [data-testid="stColumn"] {width:100%!important;min-width:0!important;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:22px;border-bottom:1px solid #dfe6f0;}
[data-testid="stTabs"] [data-baseweb="tab"] {font-size:14px;padding-top:14px;padding-bottom:14px;}
.section-note {font-size:13px;color:#64778f;line-height:1.6;margin-bottom:10px;}
.app-footer {color:#7b8ba0;font-size:12px;margin-top:26px;border-top:1px solid #e1e8f1;padding-top:14px;}
@media(max-width:1100px) {
 .st-key-metric_grid [data-testid="stHorizontalBlock"] {grid-template-columns:repeat(3,minmax(0,1fr));}
}
@media(max-width:800px) {
 .block-container {padding:4.5rem 1rem 2rem;}
 .hero {padding:24px;}.hero h1 {font-size:27px;}
 [data-testid="stMetricValue"] {font-size:23px;}
}
@media(max-width:480px) {
 .st-key-metric_grid [data-testid="stHorizontalBlock"] {grid-template-columns:repeat(2,minmax(0,1fr));}
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False, max_entries=5)
def model_resource(model_name):
    """Cache only the packaged, read-only classifiers, never uploaded datasets."""
    return load_pipeline(model_name, ROOT)


def plot(figure, key):
    st.plotly_chart(figure, width="stretch", theme=None, key=key, config=CHART_CONFIG)


def table_config(columns):
    return {name: st.column_config.NumberColumn(name, format="%.4f") for name in columns}


def result_for(model_name, frame):
    if model_name not in st.session_state.evaluations:
        st.session_state.evaluations[model_name] = evaluate_model(model_resource(model_name), frame)
    return st.session_state.evaluations[model_name]


with st.sidebar:
    st.markdown('''<div class="brand"><div class="brand-mark">S</div><div>
        <div class="brand-name">Shopper Signals</div><div class="brand-sub">PURCHASE PREDICTION</div>
        </div></div>''', unsafe_allow_html=True)
    st.markdown("**01 · Choose your data**")
    source = st.radio("Evaluation dataset", ["Included test set", "Upload a CSV"],
                      key="dataset_source", label_visibility="collapsed")
    sample_bytes = (ROOT / "test_data.csv").read_bytes()
    if source == "Upload a CSV":
        uploaded = st.file_uploader("Upload labeled test data", type=["csv"], key="dataset_upload",
                                    help="Include all 17 features and Revenue. Up to 5 MB and 5,000 rows.")
        payload = uploaded.getvalue() if uploaded is not None else None
        dataset_name = uploaded.name if uploaded is not None else "Your CSV"
    else:
        payload = sample_bytes
        dataset_name = "Included test set"
        st.caption("2,466 held-out sessions from the notebook experiment.")
    st.download_button("Download test CSV", sample_bytes, "test_data.csv", "text/csv",
                       width="stretch", key="download_test")
    st.divider()
    st.markdown("**02 · Select a classifier**")
    selected_model = st.selectbox("Prediction model", list(MODEL_FILES), index=1, key="model_name")
    with st.expander("About this model"):
        st.write(MODEL_DESCRIPTIONS[selected_model])
    st.caption("Each model uses its saved decision rule and preprocessing.")
    st.divider()
    with st.expander("CSV requirements"):
        st.write("Include these 17 features plus **Revenue**:")
        st.code(", ".join([*FEATURE_COLUMNS, "Revenue"]), language=None, wrap_lines=True)
        st.caption("Revenue and Weekend accept True/False or 1/0. Missing values are not accepted.")
    st.link_button("View project on GitHub", REPOSITORY, width="stretch")

st.markdown('''<div class="hero"><div class="eyebrow">MACHINE LEARNING · MODEL EXPLORER</div>
    <h1>Understand the sessions<br>that turn into purchases.</h1>
    <p>Explore five classifiers on the same shopping sessions. See which buyers they identify,
    where they make mistakes, and how the tradeoffs change.</p>
    <div class="hero-pills"><span>5 CLASSIFIERS</span><span>6 EVALUATION METRICS</span>
    <span>ONE SHARED DATASET</span></div></div>''', unsafe_allow_html=True)

if payload is None:
    st.info("Upload a labeled CSV from the sidebar to begin, or choose the included test set.")
    st.stop()

try:
    frame = read_dataset(payload)
except DataValidationError as error:
    st.error(f"We could not evaluate this CSV. {error}")
    st.info("Download the test CSV in the sidebar for a valid example.")
    st.stop()

data_token = sha256(payload).hexdigest()
if st.session_state.get("data_token") != data_token:
    st.session_state.data_token = data_token
    st.session_state.evaluations = {}
    st.session_state.comparison_ready = False

actual_purchases = int(frame["Revenue"].sum())
st.markdown(f'''<div class="dataset-strip"><strong>{escape(dataset_name)}</strong>
    <span><strong>{len(frame):,}</strong> sessions</span>
    <span><strong>{actual_purchases:,}</strong> actual purchases</span>
    <span><strong>{actual_purchases / len(frame):.2%}</strong> purchase rate</span>
    <span><strong>17</strong> features</span></div>''', unsafe_allow_html=True)

warnings = dataset_warnings(frame)
if warnings:
    with st.expander(f"Dataset notes ({len(warnings)})"):
        for message in warnings:
            st.write(f"• {message}")

try:
    with st.spinner("Evaluating the selected model..."):
        result = result_for(selected_model, frame)
except (ValueError, OSError, KeyError) as error:
    st.error("The selected saved model could not be evaluated. Check the project files and pinned dependencies.")
    st.caption(str(error))
    st.stop()

st.markdown(f'<div class="model-heading"><span class="model-dot" style="background:{COLORS[selected_model]}"></span>'
            f'{escape(selected_model)}</div>', unsafe_allow_html=True)
st.caption("Current dataset · Purchase is the positive class · Original model decision rule")
metric_help = {
    "Accuracy": "Share of all sessions classified correctly.",
    "AUC": "ROC area under the curve measures ranking across thresholds. Requires both outcome classes.",
    "Precision": "Share of purchase alerts that are actual purchases.",
    "Recall": "Share of actual buyers identified.",
    "F1": "Harmonic mean of purchase precision and recall.",
    "MCC": "Correlation using all four confusion counts, from -1 to 1. Requires both outcome classes here.",
}
with st.container(key="metric_grid"):
    for column, (name, value) in zip(st.columns(6), result["metrics"].items()):
        with column:
            st.metric("ROC AUC" if name == "AUC" else name,
                      "—" if value is None else f"{value:.4f}", help=metric_help[name])
if result["metrics"]["AUC"] is None:
    st.info("This dataset contains one outcome class. ROC AUC and MCC are unavailable; the other metrics and counts remain visible.")

overview, comparison_tab, explorer, about = st.tabs(["Overview", "Compare models", "Session explorer", "About"])
with overview:
    left, right = st.columns([1, 1])
    with left, st.container(border=True):
        st.subheader("Every prediction, accounted for", divider=False)
        st.caption("Actual outcomes on rows. Predicted outcomes on columns.")
        plot(confusion_chart(result["confusion"]), "confusion")
    with right, st.container(border=True):
        st.subheader("Buyer coverage & false alerts", divider=False)
        tn, fp, fn, tp = result["confusion"].ravel()
        st.caption(f"{tp:,} buyers identified · {fn:,} buyers missed · {fp:,} false alerts")
        plot(purchase_outcomes_chart(result["confusion"]), "purchase_outcomes")
    with st.container(border=True):
        st.subheader("Performance across score thresholds")
        st.caption("These curves describe ranking. They do not change the predictions or tune a threshold.")
        roc_tab, pr_tab, report_tab = st.tabs(["ROC curve", "Precision–recall curve", "Classification report"])
        with roc_tab:
            if result["metrics"]["AUC"] is not None:
                plot(curve_chart(result["y_true"], result["scores"], selected_model, "roc"), "roc")
            else:
                st.info("A ROC curve needs both purchase and nonpurchase outcomes.")
        with pr_tab:
            if result["metrics"]["AUC"] is not None:
                plot(curve_chart(result["y_true"], result["scores"], selected_model, "pr"), "pr")
                st.caption("Average precision is a supplemental summary; the six required metrics remain above.")
            else:
                st.info("Upload both outcome classes for a meaningful precision-recall comparison.")
        with report_tab:
            st.dataframe(result["report"], width="stretch",
                         column_config={**table_config(["precision", "recall", "f1-score"]),
                                        "support": st.column_config.NumberColumn("Sessions", format="%d")})
            st.caption("Support is the number of actual sessions in each class. Undefined precision or recall is reported as zero.")

with comparison_tab:
    st.subheader("Find the tradeoff that fits")
    st.write("Compare all five saved classifiers on the currently selected dataset. Purchase F1 is the primary ranking measure, with MCC as a tiebreaker.")
    if st.button("Compare all five models", type="primary", key="compare_models"):
        try:
            with st.spinner("Evaluating all five classifiers..."):
                for name in MODEL_FILES:
                    result_for(name, frame)
            st.session_state.comparison_ready = True
        except (ValueError, OSError, KeyError) as error:
            st.error(f"Comparison could not be completed. {error}")
    if st.session_state.comparison_ready:
        comparison = pd.DataFrame({name: st.session_state.evaluations[name]["metrics"] for name in MODEL_FILES}).T
        comparison.index.name = "Model"
        ranked = comparison.sort_values(["F1", "MCC"], ascending=False, na_position="last")
        if frame["Revenue"].nunique() == 2:
            best = ranked.index[0]
            st.success(f"Highest purchase F1 on this dataset: {best} · {ranked.iloc[0]['F1']:.4f}")
        else:
            st.info("Both outcome classes are needed for a balanced model comparison.")
        st.dataframe(ranked, width="stretch", column_config=table_config(comparison.columns))
        c1, c2 = st.columns([1, 1])
        with c1, st.container(border=True):
            st.markdown("**Purchase F1 by model**")
            plot(ranking_chart(comparison), "ranking")
        with c2, st.container(border=True):
            st.markdown("**Explore the performance space**")
            st.caption("Drag to rotate · Use the chart controls to zoom · Hover for exact values")
            if frame["Revenue"].nunique() == 2:
                plot(performance_3d_chart(comparison), "performance_3d")
            else:
                st.info("The 3D view needs defined MCC values from both outcome classes.")
        st.download_button("Download comparison CSV", ranked.to_csv().encode("utf-8"),
                           "model_comparison.csv", "text/csv", key="download_comparison")
        st.caption("A winner on one dataset does not establish performance on future traffic. Selection and tuning require a separate validation process.")
    else:
        st.info("Run the comparison to see all six metrics, an F1 ranking, and an interactive 3D view.")

with explorer:
    st.subheader("Look beyond the averages")
    st.caption("Inspect individual outcomes and download every prediction from the selected model.")
    predictions = result["predictions"].copy()
    predictions.insert(0, "SessionRow", range(1, len(predictions) + 1))
    prediction_filter = st.selectbox("Show sessions", ["All sessions", "Correct predictions", "False purchase alerts", "Missed purchases"], key="prediction_filter")
    filters = {
        "All sessions": pd.Series(True, index=predictions.index),
        "Correct predictions": predictions["Correct"],
        "False purchase alerts": ~predictions["Revenue"] & predictions["PredictedRevenue"],
        "Missed purchases": predictions["Revenue"] & ~predictions["PredictedRevenue"],
    }
    visible = predictions.loc[filters[prediction_filter]]
    st.caption(f"Showing {len(visible):,} of {len(predictions):,} sessions · Score: {result['score_label']}")
    if result["score_label"] == "Decision score":
        st.info("SVM decision scores are uncalibrated ranking values, not purchase probabilities. Positive scores lie on the purchase side of its decision boundary.")
    shown_columns = ["SessionRow", "Revenue", "PredictedRevenue", "Correct", "PredictionScore", "PageValues", "ProductRelated", "Month", "VisitorType"]
    st.dataframe(visible[shown_columns], hide_index=True, width="stretch",
                 column_config={
                     "SessionRow": st.column_config.NumberColumn("Row", format="%d"),
                     "Revenue": st.column_config.CheckboxColumn("Actual purchase"),
                     "PredictedRevenue": st.column_config.CheckboxColumn("Predicted purchase"),
                     "Correct": st.column_config.CheckboxColumn("Correct"),
                     "PredictionScore": st.column_config.NumberColumn(result["score_label"], format="%.4f"),
                 })
    st.download_button("Download all predictions", predictions.to_csv(index=False).encode("utf-8"),
                       selected_model.lower().replace(" ", "_") + "_predictions.csv", "text/csv", key="download_predictions")
    with st.expander("Preview all input features"):
        st.dataframe(frame.head(100), hide_index=True, width="stretch")
        st.caption("Preview limited to the first 100 rows. Evaluation uses every validated row.")

with about:
    st.subheader("A transparent model comparison")
    st.markdown("""
This project predicts whether a recorded online shopping session ends in a purchase.
All five classifiers share the same original stratified 80/20 split and use saved
preprocessing fitted only on training rows. The app evaluates existing pipelines;
it does not fit them to an uploaded file.

**Data:** UCI Online Shoppers Purchasing Intention Dataset — 12,330 sessions and
17 predictors. Revenue is the purchase outcome. The included test set contains
2,466 sessions, including 382 purchases.

**Interpretation:** purchase F1 balances precision and recall. The Decision Tree
ranked highest in the original notebook. A different upload may produce a different
ranking. Use truly held-out labeled data when assessing generalization.

**Limitations:** one split does not measure uncertainty or future performance.
The source contains repeated rows; 29 test rows share feature values with training
rows. PageValues and accumulated session activity require a prediction-time review
before early-session use.

**Uploaded data:** files are processed for the current app session and are not
written to the repository. Only the packaged model objects are shared in the cache.
""")
    st.link_button("Read the methods, notebook, and results", REPOSITORY)
    st.markdown("[Dataset attribution and license](https://doi.org/10.24432/C5F88Q) · "
                "Sakar, C. & Kastro, Y. (2018), CC BY 4.0")

st.markdown('<div class="app-footer">SHOPPER SIGNALS · Recorded sessions, measurable outcomes · '
            'Machine Learning Assignment 2</div>', unsafe_allow_html=True)

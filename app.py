"""Shopper Signals: inspect saved purchase decisions through a retail ledger."""

from hashlib import sha256
from uuid import uuid4

import pandas as pd
import streamlit as st

from shopper_app.appearance import render_appearance
from shopper_app.charts import curve_chart, performance_3d_chart, ranking_chart
from shopper_app.evaluation import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    MODEL_DESCRIPTIONS,
    MODEL_FILES,
    DataValidationError,
    dataset_warnings,
    evaluate_model,
    load_pipeline,
    model_metadata,
    read_dataset,
    schema_guide,
)
from shopper_app.explorer import (
    FILTER_OPTIONS,
    export_sessions,
    filter_sessions,
    paginate_sessions,
    session_detail,
)
from shopper_app.presentation import (
    ROOT,
    brand,
    coverage_bar,
    finding,
    formatted,
    heading,
    install_styles,
    ledger_table,
    text,
)
from shopper_app.session_table import session_table

REPOSITORY = "https://github.com/akriti-adarsh/online-shopper-purchase-prediction"
VIEWS = ["Overview", "Compare Models", "Session Explorer", "About"]
METRIC_LABELS = {"Accuracy": "Accuracy", "AUC": "ROC AUC", "Precision": "Precision",
                 "Recall": "Recall", "F1": "F1", "MCC": "MCC"}
METRIC_HELP = {
    "Accuracy": "All correctly classified sessions divided by all evaluated sessions. High accuracy can hide missed buyers.",
    "AUC": "Area under the ROC curve, calculated from purchase scores. Requires scores and both observed classes.",
    "Precision": "Of sessions flagged as purchases, the share that actually bought: TP / (TP + FP).",
    "Recall": "Of actual buyers, the share identified: TP / (TP + FN).",
    "F1": "Harmonic mean of purchase precision and recall. It balances these two measures equally.",
    "MCC": "Correlation using all four outcomes. Range −1 to +1; higher is better. Unavailable here for one observed class.",
}
CHART_CONFIG = {"displaylogo": False, "scrollZoom": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
MODEL_VERSION = sha256((ROOT / "results/experiment_manifest.json").read_bytes()).hexdigest()
SAMPLE_BYTES = (ROOT / "test_data.csv").read_bytes()

st.set_page_config(page_title="Shopper Signals · Purchase decisions", page_icon=str(ROOT / "assets/brand.svg"), layout="wide")
install_styles()


def set_focus(selector):
    st.session_state.focus_request = {"token": uuid4().hex, "selector": selector}


def reset_page():
    st.session_state.session_page = 1


def reset_filters():
    st.session_state.outcome_filter = "All sessions"
    st.session_state.session_search = ""
    st.session_state.session_sort = "OriginalRow"
    st.session_state.sort_direction = "Ascending"
    reset_page()


def open_explorer(category, cell=None):
    st.session_state.view = "Session Explorer"
    st.session_state.outcome_filter = category
    st.session_state.session_search = ""
    st.session_state.return_cell = cell
    reset_page()
    set_focus("#explorer-heading")


def back_to_overview():
    st.session_state.view = "Overview"
    cell = st.session_state.get("return_cell")
    set_focus(f".st-key-cell_{cell} button" if cell else "#overview-heading")


def change_view():
    set_focus({"Overview": "#overview-heading", "Compare Models": "#comparison-heading",
               "Session Explorer": "#explorer-heading", "About": "#about-heading"}.get(st.session_state.view, "#overview-heading"))


def activate_dataset(frame, name, kind):
    identity = frame.attrs["dataset_fingerprint"]
    if st.session_state.get("active_id") != identity:
        st.session_state.active_id = identity
        st.session_state.active_frame = frame
        st.session_state.evaluations = {}
        reset_filters()
    st.session_state.dataset_name = name
    st.session_state.dataset_kind = kind


@st.cache_resource(show_spinner=False, max_entries=5)
def model_resource(name, artifact_version):
    pipeline = load_pipeline(name, ROOT)
    return pipeline, model_metadata(pipeline, name)


def result_for(name):
    pipeline, metadata = model_resource(name, MODEL_VERSION)
    identity = st.session_state.active_id
    cache_key = (identity, metadata["identity"])
    if cache_key not in st.session_state.evaluations:
        evaluated = evaluate_model(pipeline, st.session_state.active_frame)
        if st.session_state.active_id != identity:
            st.stop()
        st.session_state.evaluations[cache_key] = evaluated
    return st.session_state.evaluations[cache_key], metadata


def plot(figure, key):
    st.plotly_chart(figure, theme=None, width="stretch", config=CHART_CONFIG, key=key)


if "active_id" not in st.session_state:
    activate_dataset(read_dataset(SAMPLE_BYTES), "Bundled test set", "bundled")
for key, default in {"view": "Overview", "model_name": "Decision Tree", "dataset_source": "Bundled test set",
                     "session_page": 1, "page_size": 25, "comparison_metric": "F1",
                     "session_sort": "OriginalRow", "sort_direction": "Ascending",
                     "outcome_filter": "All sessions", "session_search": "", "matrix_mode": "Count"}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Keep conditional widgets' values when their destination is not being rendered.
for key in ["outcome_filter", "session_search", "session_sort", "sort_direction",
            "page_size", "comparison_metric", "matrix_mode"]:
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]

with st.container(key="brand_row"):
    brand_column, appearance_column = st.columns([4, 1])
    with brand_column:
        brand()
    with appearance_column:
        theme = render_appearance(st.session_state.get("focus_request"))

with st.container(key="navigation"):
    st.segmented_control("Navigation", VIEWS, key="view", label_visibility="collapsed",
                         width="stretch", on_change=change_view)

with st.container(key="evaluation_header"):
    model_column, settings_column = st.columns([3, 1], vertical_alignment="bottom")
    with model_column:
        selected_model = st.selectbox("Prediction model", list(MODEL_FILES), key="model_name")
    with settings_column:
        with st.popover("Evaluation settings", width="stretch", icon=":material/tune:"):
            with st.container(key="dataset_settings"):
                st.subheader("Evaluation settings")
                source = st.radio("Dataset source", ["Bundled test set", "Upload CSV"], key="dataset_source")
                if source == "Bundled test set":
                    st.session_state.upload_error = None
                    st.session_state.upload_attempt = None
                    if st.session_state.dataset_kind != "bundled":
                        activate_dataset(read_dataset(SAMPLE_BYTES), "Bundled test set", "bundled")
                    st.caption("2,466 sessions held out from model training. Revenue contains the observed purchase outcome.")
                else:
                    uploaded = st.file_uploader("Upload session CSV", type=["csv"], key="dataset_upload",
                                                help="Up to 5 MB and 5,000 rows. Revenue is optional for predictions.")
                    if uploaded is None:
                        st.session_state.upload_error = None
                        st.session_state.upload_attempt = None
                        st.info("Choose a CSV. Your current valid dataset stays active until a new file passes validation.")
                    else:
                        payload = uploaded.getvalue()
                        attempt = sha256(payload).hexdigest()
                        if st.session_state.get("upload_attempt") != attempt:
                            st.session_state.upload_attempt = attempt
                            try:
                                candidate = read_dataset(payload, require_target=False)
                                st.session_state.valid_upload = (attempt, candidate, uploaded.name)
                                st.session_state.upload_error = None
                            except DataValidationError as error:
                                st.session_state.upload_error = str(error)
                        accepted = st.session_state.get("valid_upload")
                        if accepted and accepted[0] == attempt:
                            activate_dataset(accepted[1], uploaded.name, "uploaded")
                            st.success(f"Active: {uploaded.name} · {len(accepted[1]):,} validated sessions")
                        elif st.session_state.get("upload_error"):
                            st.error(st.session_state.upload_error)
                            st.caption(f"Still using {st.session_state.dataset_name}. Correct the file and upload it again.")
                st.download_button("Download test CSV", SAMPLE_BYTES, "test_data.csv", "text/csv", width="stretch", key="download_test")
                with st.expander("CSV requirements"):
                    st.write("All 17 features are required. Include Revenue to evaluate accuracy and buyer outcomes, or omit it for predictions only. Column order may differ.")
                    ledger_table(schema_guide(), "Accepted fields and validation rules")
                    st.download_button("Download schema guide", schema_guide().to_csv(index=False).encode("utf-8"),
                                       "shopper_schema.csv", "text/csv", key="download_schema")
                with st.expander("Preview active data"):
                    ledger_table(st.session_state.active_frame.head(5), "First five validated input rows")
                with st.expander("About the selected model"):
                    st.write(MODEL_DESCRIPTIONS[selected_model])

frame = st.session_state.active_frame
dataset_name = st.session_state.dataset_name
labeled = "Revenue" in frame
buyer_count = int(frame["Revenue"].sum()) if labeled else None
class_text = f"{buyer_count:,} actual buyers · {buyer_count/len(frame):.2%} prevalence" if labeled else "Observed outcomes not provided"
st.html(f'<div class="context-line"><span>DATA <strong>{text(dataset_name)}</strong></span>'
        f'<span><strong>{len(frame):,}</strong> sessions</span><span>{class_text}</span>'
        '<span>Positive class <strong>Purchase</strong></span></div>')
if source == "Upload CSV" and st.session_state.get("upload_error"):
    st.error(f"The new upload was rejected. {st.session_state.upload_error} Your last valid dataset ({dataset_name}) remains active.")

try:
    with st.spinner("Evaluating the selected model…"):
        result, metadata = result_for(selected_model)
except (ValueError, OSError, KeyError) as error:
    st.error("Evaluation could not be completed. The valid dataset is retained. Restore the packaged model files and pinned dependencies, then retry.")
    st.caption(str(error))
    st.stop()


def render_metrics():
    with st.container(key="metric_grid"):
        for column, (metric, value) in zip(st.columns(6), result["metrics"].items()):
            with column:
                st.metric(METRIC_LABELS[metric], "—" if value is None else f"{value:.4f}", help=METRIC_HELP[metric])
    with st.popover("How to read these metrics"):
        for metric, explanation in METRIC_HELP.items():
            st.markdown(f"**{METRIC_LABELS[metric]}** — {explanation}")
        st.caption("Higher is better for all six metrics. Accuracy, AUC, precision, recall, and F1 range from 0 to 1. MCC ranges from −1 to 1. A dash means unavailable.")
        st.caption("Headline precision, recall, and F1 retain the notebook's zero-for-undefined convention. The detailed report marks undefined values unavailable.")


def render_matrix():
    heading("Every decision, accounted for", "01 / OUTCOMES")
    st.caption("Actual outcomes on rows. Predicted outcomes on columns. Select a cell to inspect its sessions.")
    mode = st.segmented_control("Matrix values", ["Count", "% of actual class"], key="matrix_mode")
    matrix = result["confusion"]
    with st.container(key="matrix_grid"):
        st.html('<div class="matrix-axis"><span>Predicted: no purchase</span><span>Predicted: purchase</span></div>')
        for row, actual in enumerate(["No purchase", "Purchase"]):
            st.html(f'<div class="matrix-row-label">Actual: {text(actual)}</div>')
            cells = st.columns(2)
            for col, container in enumerate(cells):
                code, category = [[("tn", "Correct non-buyers"), ("fp", "False alerts")],
                                  [("fn", "Missed buyers"), ("tp", "Identified buyers")]][row][col]
                count = int(matrix[row, col])
                denominator = int(matrix[row].sum())
                value = f"{count:,}" if mode != "% of actual class" else (f"{count/denominator:.1%}" if denominator else "Unavailable")
                suffix = f"{code.upper()} · {count:,} sessions" if mode == "% of actual class" else code.upper()
                with container:
                    st.button(f"**{value}**\n\n{category}\n\n{suffix}", width="stretch", key=f"cell_{code}",
                              on_click=open_explorer, args=(category, code))
    st.caption("Each percentage divides the cell count by its actual-class row total. A row with no observations is unavailable."
               if mode == "% of actual class" else f"All four outcomes add up to {int(matrix.sum()):,} evaluated sessions.")


def render_coverage():
    heading("Finding buyers. Earning trust.", "02 / COVERAGE")
    st.caption("Two denominators reveal the difference between recall and precision.")
    tn, fp, fn, tp = map(int, result["confusion"].ravel())
    coverage_bar("Actual buyers · recall", "Of the people who bought, how many did we find?", tp, fn, "Missed", "missed")
    coverage_bar("Purchase alerts · precision", "Of the sessions flagged as purchases, how many were correct?", tp, fp, "False alerts", "false")
    a, b = st.columns(2)
    with a:
        st.button(f"Inspect {fn:,} missed buyers", on_click=open_explorer, args=("Missed buyers", "fn"), key="inspect_missed", width="stretch")
    with b:
        st.button(f"Inspect {fp:,} false alerts", on_click=open_explorer, args=("False alerts", "fp"), key="inspect_false", width="stretch")


def overview():
    st.html('<h1 id="overview-heading" tabindex="-1">The purchase ledger</h1>')
    finding(result, len(frame))
    render_metrics()
    notes = dataset_warnings(frame)
    if notes:
        with st.expander(f"Dataset notes · {len(notes)}"):
            for note in notes:
                st.write(note)
    if not labeled:
        st.info("This feature-only dataset supports predictions and scores. Add observed Revenue values for evaluation metrics, confusion counts, and ranking curves.")
        st.button("Explore predictions", on_click=open_explorer, args=("All sessions",), type="primary", key="explore_unlabeled")
        return
    if frame["Revenue"].nunique() < 2:
        st.info("Only one outcome class is present. ROC AUC and MCC are unavailable. Counts and predictions remain available.")
    with st.container(key="outcome_row"):
        left, right = st.columns(2)
        with left, st.container(key="panel_matrix"):
            render_matrix()
        with right, st.container(key="panel_coverage"):
            render_coverage()
    with st.container(key="panel_curves"):
        heading("Beyond one decision rule", "03 / RANKING")
        st.caption("The saved model's predictions stay fixed. Curves show how its scores rank sessions across possible thresholds.")
        roc_tab, pr_tab, report_tab = st.tabs(["ROC curve", "Precision–recall", "Classification report"])
        for tab, kind in [(roc_tab, "roc"), (pr_tab, "pr")]:
            with tab:
                st.write("ROC compares buyer recall with the false-positive rate among actual non-buyers. The diagonal is chance ranking."
                         if kind == "roc" else "Precision–recall focuses on buyers. Its baseline is observed buyer prevalence; AP means average precision.")
                if result["metrics"]["AUC"] is None:
                    st.info("A meaningful ranking curve needs both observed classes and available model scores.")
                else:
                    plot(curve_chart(result["y_true"], result["scores"], selected_model, kind,
                                     confusion=result["confusion"], operating_point=result["operating_point"],
                                     positive_label=result["positive_class"], theme=theme), f"curve_{kind}")
                    point = result["operating_point"]
                    if point:
                        precision_text = f"{point['precision']:.2%}" if point["precision"] is not None else "unavailable"
                        st.caption(f"Saved rule: recall {point['recall']:.2%} · false-positive rate {point['fpr']:.2%} · precision {precision_text}")
                        with st.expander("Operating point values · text equivalent"):
                            ledger_table(pd.DataFrame([{"Recall": point["recall"], "False-positive rate": point["fpr"],
                                                       "Precision": point["precision"]}]), "Actual saved prediction rule",
                                         numeric_columns=["Recall", "False-positive rate", "Precision"])
        with report_tab:
            report = result["report"].rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Support"})
            report.index.name = "Class / average"
            ledger_table(report, "Class-level performance and averages", index=True, numeric_columns=["Precision", "Recall", "F1", "Support"])
            st.caption("Support is the number of observed sessions in each class. A dash marks an undefined value; averages exclude undefined terms where applicable.")


def comparison_view():
    st.html('<h1 id="comparison-heading" tabindex="-1">Choose the tradeoff</h1>')
    st.write("Five saved classifiers. The same sessions. Their original decision rules.")
    if not labeled:
        st.info("Observed Revenue values are required to compare evaluation metrics. You can still switch models and inspect their predictions in Session Explorer.")
        return
    all_results = {}
    try:
        with st.spinner("Comparing all five models on the active dataset…"):
            for name in MODEL_FILES:
                all_results[name] = result_for(name)[0]
    except (ValueError, OSError, KeyError) as error:
        st.error(f"Comparison could not finish. The current evaluation is retained. {error}")
        return
    comparison = pd.DataFrame({name: item["metrics"] for name, item in all_results.items()}).T
    comparison.index.name = "Model"
    metric = st.selectbox("Rank by metric", list(METRIC_LABELS), format_func=METRIC_LABELS.get, key="comparison_metric")
    st.caption(METRIC_HELP[metric] + " Higher is better. Equal values share the same rank.")
    ranked = comparison.sort_values(metric, ascending=False, kind="stable", na_position="last")
    display = ranked.copy()
    display.insert(0, "Rank", ranked[metric].rank(method="min", ascending=False).astype("Int64"))
    display = display.rename(columns={"AUC": "ROC AUC"})
    with st.container(key="panel_comparison"):
        ledger_table(display, f"All models ranked by {METRIC_LABELS[metric]} · selected model highlighted", index=True,
                     selected=selected_model, numeric_columns=list(display.columns))
        plot(ranking_chart(comparison, metric, selected_model, theme=theme), "metric_ranking")
    coverage_name = comparison["Recall"].idxmax()
    alert_counts = {name: int(item["confusion"][0, 1]) for name, item in all_results.items()}
    cautious_name = min(alert_counts, key=alert_counts.get)
    high = all_results[coverage_name]["confusion"]
    low = all_results[cautious_name]["confusion"]
    if buyer_count:
        st.info(f"{coverage_name} identifies {int(high[1,1]):,} of {buyer_count:,} buyers with {int(high[0,1]):,} false alerts. "
                f"{cautious_name} produces {int(low[0,1]):,} false alerts and identifies {int(low[1,1]):,} buyers on these same rows.")
    else:
        st.info("This dataset contains no observed buyers. Buyer coverage cannot establish a model preference.")
    with st.expander("Explore the 3D performance space"):
        if comparison[["Precision", "Recall", "MCC"]].notna().all().all():
            st.caption("Drag to rotate. Use the table above as the keyboard-accessible equivalent for exact values.")
            plot(performance_3d_chart(comparison, selected_model, theme=theme), "comparison_3d")
        else:
            st.info("The 3D comparison needs defined precision, recall, and MCC values.")
    st.download_button("Download comparison CSV", display.to_csv().encode("utf-8"), "model_comparison.csv", "text/csv", key="download_comparison")
    st.caption("These results describe this dataset. Selecting a model and tuning settings requires a separate validation process before a final untouched evaluation.")


def close_detail():
    identifier = st.session_state.get("detail_id")
    if identifier:
        set_focus(f'button[data-session="{identifier}"]')


def outcome_word(value):
    return "Unavailable" if value is None else ("Purchase" if bool(value) else "No purchase")


@st.dialog("Session detail", width="large", on_dismiss=close_detail)
def show_detail(identifier):
    try:
        row = session_detail(result["predictions"], identifier)
    except KeyError:
        st.info("That session is no longer in the active dataset. Close this panel and choose another row.")
        return
    st.session_state.detail_id = identifier
    st.html(f'<p class="mono">{text(identifier)} · original data row {int(row["OriginalRow"]):,}</p>')
    st.html(f'<div class="detail-summary"><div><small>Observed</small><strong>{outcome_word(row.get("Revenue"))}</strong></div>'
            f'<div><small>Predicted</small><strong>{outcome_word(row["PredictedRevenue"])}</strong></div>'
            f'<div><small>{text(result["score_label"])}</small><strong>{formatted(row["PredictionScore"], True)}</strong></div></div>')
    st.write(f"**Outcome:** {row['ResultCategory']}")
    if result["score_label"] == "Decision score":
        st.caption("This SVM score is an uncalibrated ranking value, not a purchase probability. Positive scores lie on the purchase side of the boundary.")
    details = pd.DataFrame([{"Feature": FEATURE_LABELS[name], "Column": name, "Value": str(row[name])} for name in FEATURE_COLUMNS])
    ledger_table(details, "Original features supplied to the saved preprocessing pipeline")
    st.caption("These are the recorded input features. No per-session feature contributions are claimed.")
    if st.button("Back to sessions", key="close_detail", type="primary"):
        close_detail()
        st.rerun()


def explorer_view():
    requested_detail = None
    st.html('<h1 id="explorer-heading" tabindex="-1">Inspect the sessions</h1>')
    st.button("Back to overview", on_click=back_to_overview, key="back_to_overview", icon=":material/arrow_back:")
    st.caption("Search recorded features or a session ID. Select an ID to open its original inputs and prediction.")
    filters = st.columns([2, 3, 1])
    with filters[0]:
        category = st.selectbox("Outcome filter", FILTER_OPTIONS, key="outcome_filter", on_change=reset_page)
    with filters[1]:
        query = st.text_input("Search sessions", key="session_search", placeholder="Session ID, month, visitor type…", on_change=reset_page)
    with filters[2]:
        st.button("Reset filters", on_click=reset_filters, key="reset_filters", width="stretch")
    sort_labels = {"OriginalRow": "Original row", "SessionID": "Session ID", "PredictionScore": result["score_label"],
                   "PredictedRevenue": "Predicted outcome", "ResultCategory": "Outcome category"}
    if labeled:
        sort_labels["Revenue"] = "Observed outcome"
    if st.session_state.session_sort not in sort_labels:
        st.session_state.session_sort = "OriginalRow"
    controls = st.columns([2, 1, 1])
    with controls[0]:
        sort = st.selectbox("Sort by", list(sort_labels), format_func=sort_labels.get, key="session_sort", on_change=reset_page)
    with controls[1]:
        direction = st.selectbox("Direction", ["Ascending", "Descending"], key="sort_direction", on_change=reset_page)
    with controls[2]:
        size = st.selectbox("Rows per page", [10, 25, 50, 100], key="page_size", on_change=reset_page)
    filtered = filter_sessions(result["predictions"], category, query, sort, direction == "Ascending")
    page = paginate_sessions(filtered, int(st.session_state.session_page), int(size))
    st.session_state.session_page = page["page"]
    st.caption(f"{len(filtered):,} matching sessions of {len(frame):,} · {category} · Score: {result['score_label']}")
    if result["score_label"] == "Decision score":
        st.info("SVM decision scores are uncalibrated ranking values, not purchase probabilities.")
    if filtered.empty:
        st.info("No sessions match these filters. The active dataset is valid; reset the filters or change your search.")
        return
    with st.container(key="panel_explorer"):
        interaction = session_table(page["rows"], result["score_label"],
                                    f"Rows {page['start']:,}–{page['end']:,} of {page['count']:,} matching sessions",
                                    sort, direction == "Ascending")
        if interaction.sort and interaction.sort in sort_labels:
            st.session_state.pending_sort = interaction.sort
            st.rerun()
        if interaction.selected:
            requested_detail = interaction.selected
    previous, page_label, next_page = st.columns([1, 2, 1])
    with previous:
        st.button("Previous page", disabled=page["page"] <= 1, on_click=lambda: st.session_state.update(session_page=page["page"] - 1), key="previous_page", width="stretch")
    with page_label:
        st.caption(f"Page {page['page']} of {page['pages']} · {size} rows per page")
    with next_page:
        st.button("Next page", disabled=page["page"] >= page["pages"], on_click=lambda: st.session_state.update(session_page=page["page"] + 1), key="next_page", width="stretch")
    st.download_button(f"Export {len(filtered):,} filtered predictions", export_sessions(filtered),
                       "shopper_filtered_predictions.csv", "text/csv", key="download_predictions", type="primary")
    with st.expander("Choose a session using the keyboard"):
        identifier = st.selectbox("Session on this page", page["rows"]["SessionID"].tolist(), key="detail_picker")
        if st.button("Open session detail", key="open_detail"):
            requested_detail = identifier
    if requested_detail:
        show_detail(requested_detail)


def about_view():
    st.html('<h1 id="about-heading" tabindex="-1">Behind the ledger</h1>')
    with st.container(key="panel_about"):
        st.markdown("""
### What a session means
A session is one recorded visit to an online shop. A purchase means the visit ended in a transaction: **Revenue = True**. The other class is a visit without a purchase.

### One experiment, five perspectives
Logistic Regression, Decision Tree, k-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine share a stratified 80/20 split of the UCI Online Shoppers Purchasing Intention Dataset. The source contains **12,330 sessions and 17 features**. The bundled holdout contains **2,466 sessions, including 382 buyers**.

Each saved pipeline encodes categories and scales numerical features using transformations fitted only on training rows. The application calls the saved model's prediction rule; it never trains on an upload. Purchase probabilities or SVM decision scores are used for ranking curves and ROC AUC. SVM scores are not calibrated probabilities.

### Read the tradeoffs
The Decision Tree has the highest purchase F1 in the original experiment. F1 balances precision and recall; MCC uses all four outcomes. Accuracy alone can hide missed buyers because most sessions do not end in a purchase. Model comparisons always use the same active dataset.

### Where the evidence ends
One split does not establish uncertainty or future performance. The source includes 125 exact duplicate rows; 29 test rows share feature values with training rows. PageValues and accumulated activity need a prediction-time review before early-session use. No causal claims or individual feature contributions are inferred here.

### Your uploaded data
Uploads and evaluation results belong to the current app session and are not written to the repository or shared in a data cache. Only trusted packaged model objects are cached across sessions. Invalid uploads preserve the last valid dataset. Session identifiers retain each original row through filtering, paging, and export.
""")
        ledger_table(pd.DataFrame([{"Stage": "Input", "What happens": "17 recorded session features; Revenue is reserved for evaluation"},
                                  {"Stage": "Preprocessing", "What happens": f"Saved scaling and encoding produce {metadata['transformed_feature_count']} features"},
                                  {"Stage": "Prediction", "What happens": f"{selected_model} applies its original fitted decision rule"},
                                  {"Stage": "Evaluation", "What happens": "Predictions are compared with observed outcomes; scores produce ranking metrics"}]),
                     "Trace one session through the pipeline")
        st.html(f'<div class="receipt">ACTIVE MODEL <strong>{text(selected_model)}</strong><br>'
                f'<span class="mono">Model {metadata["model_fingerprint"][:12]} · Preprocessing {metadata["preprocessing_fingerprint"][:12]}<br>'
                f'Dataset {st.session_state.active_id[:12]}</span></div>')
        st.link_button("Project, notebook & verified results", REPOSITORY, width="stretch")
        st.markdown("Dataset: Sakar, C. & Kastro, Y. (2018). [UCI Online Shoppers Purchasing Intention](https://doi.org/10.24432/C5F88Q), CC BY 4.0.")


if st.session_state.get("pending_sort"):
    pending = st.session_state.pop("pending_sort")
    st.session_state.sort_direction = ("Descending" if st.session_state.session_sort == pending and
                                       st.session_state.sort_direction == "Ascending" else "Ascending")
    st.session_state.session_sort = pending
    reset_page()

view = st.session_state.view or "Overview"
with st.container(key="results"):
    {"Overview": overview, "Compare Models": comparison_view,
     "Session Explorer": explorer_view, "About": about_view}[view]()
st.html('<footer class="app-footer"><span>SHOPPER SIGNALS · Machine Learning Assignment 2</span>'
        '<span>Recorded sessions. Inspectable decisions.</span></footer>')

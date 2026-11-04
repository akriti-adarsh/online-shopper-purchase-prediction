"""Validate session data and evaluate the five published decision rules."""

import csv
import json
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "k-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
MODEL_DESCRIPTIONS = {
    "Logistic Regression": (
        "The most precise purchase alerts on the original test set, with fewer buyers detected. "
        "A useful linear reference when false alerts matter."
    ),
    "Decision Tree": (
        "The highest purchase F1, MCC, and ROC AUC on the original test set. "
        "Its depth-limited rules provided the strongest balance in that comparison."
    ),
    "k-Nearest Neighbors": (
        "Uses the vote of 11 nearby training sessions. On the original test set it produced "
        "relatively few false alerts, but detected the fewest buyers."
    ),
    "Gaussian Naive Bayes": (
        "Detected the most buyers on the original test set, with the most false alerts. "
        "Its broad coverage comes with lower purchase precision."
    ),
    "Support Vector Machine": (
        "A nonlinear classifier with the second-highest purchase F1 on the original test set. "
        "It detected fewer buyers than the tree while producing fewer false alerts."
    ),
}
NUMERIC_COLUMNS = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CATEGORY_CODE_COLUMNS = ["OperatingSystems", "Browser", "Region", "TrafficType"]
FEATURE_COLUMNS = [
    *NUMERIC_COLUMNS,
    "Month", *CATEGORY_CODE_COLUMNS, "VisitorType", "Weekend",
]
TARGET = "Revenue"
MAX_ROWS = 5000
MAX_BYTES = 5 * 1024 * 1024
COUNT_COLUMNS = ["Administrative", "Informational", "ProductRelated"]
RATE_COLUMNS = ["BounceRates", "ExitRates", "SpecialDay"]
# Categories observed by the encoders fitted on the documented seed-42 split.
KNOWN_CATEGORIES = {
    "Month": {"Aug", "Dec", "Feb", "Jul", "June", "Mar", "May", "Nov", "Oct", "Sep"},
    "VisitorType": {"New_Visitor", "Other", "Returning_Visitor"},
    "OperatingSystems": set(range(1, 9)),
    "Browser": set(range(1, 14)),
    "Region": set(range(1, 10)),
    "TrafficType": set(range(1, 21)),
}
RESULT_CATEGORIES = [
    "Correct non-buyers", "False alerts", "Missed buyers", "Identified buyers", "Unlabeled",
]
FEATURE_LABELS = {
    "Administrative": "Administrative pages",
    "Administrative_Duration": "Administrative duration",
    "Informational": "Informational pages",
    "Informational_Duration": "Informational duration",
    "ProductRelated": "Product pages",
    "ProductRelated_Duration": "Product page duration",
    "BounceRates": "Bounce rate", "ExitRates": "Exit rate", "PageValues": "Page value",
    "SpecialDay": "Special-day proximity", "Month": "Session month",
    "OperatingSystems": "Operating system code", "Browser": "Browser code",
    "Region": "Region code", "TrafficType": "Traffic source code",
    "VisitorType": "Visitor type", "Weekend": "Weekend session", "Revenue": "Observed purchase",
}


class DataValidationError(ValueError):
    """The uploaded CSV cannot be evaluated with the published pipelines."""


def _rows(mask: pd.Series) -> str:
    """Identify CSV records without including uploaded values in an error."""
    positions = np.flatnonzero(mask.to_numpy()) + 2
    shown = ", ".join(map(str, positions[:5]))
    return f"CSV row{'s' if len(positions) != 1 else ''} {shown}" + (" and more" if len(positions) > 5 else "")


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Hash normalized ordered data, independent of CSV formatting or headers."""
    columns = [*FEATURE_COLUMNS, *([TARGET] if TARGET in frame.columns else [])]
    canonical = frame.loc[:, columns].copy()
    for column in NUMERIC_COLUMNS:
        canonical[column] = canonical[column].astype(float)
    canonical_text = canonical.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return sha256(canonical_text.encode("utf-8")).hexdigest()


def read_dataset(payload: bytes, require_target: bool = True) -> pd.DataFrame:
    """Read a UTF-8 CSV and return validated features and Boolean outcomes.

    Column order may vary in an upload. The returned frame always follows the
    original feature order and retains Revenue solely for evaluation. Set
    require_target=False to accept feature-only files for prediction.
    """
    if not isinstance(payload, bytes):
        raise DataValidationError("Provide the CSV file as bytes.")
    if not payload or not payload.strip():
        raise DataValidationError("The CSV is empty. Upload a dataset with session rows.")
    if len(payload) > MAX_BYTES:
        raise DataValidationError("The CSV exceeds the 5 MB limit. Upload a smaller test subset.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise DataValidationError("Save the CSV with UTF-8 encoding, then upload it again.") from error
    if "\x00" in text:
        raise DataValidationError("The file contains invalid text. Upload a UTF-8 CSV.")

    reader = csv.reader(StringIO(text, newline=""), strict=True)
    try:
        header = next(reader, [])
        header = [name.strip() for name in header]
        if not header:
            raise DataValidationError("The CSV has no column header.")
        if len(header) != len(set(header)):
            raise DataValidationError("The CSV contains duplicate column names. Keep each column once.")
        expected = [*FEATURE_COLUMNS, *([TARGET] if require_target or TARGET in header else [])]
        missing = [name for name in expected if name not in header]
        extra = [name for name in header if name not in [*FEATURE_COLUMNS, TARGET]]
        if missing or extra:
            details = []
            if missing:
                details.append("Missing columns: " + ", ".join(missing))
            if extra:
                details.append("Unexpected columns: " + ", ".join(extra))
            raise DataValidationError(". ".join(details) + ". Use the downloadable test CSV as a template.")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise DataValidationError(
                    f"CSV row {line_number} has {len(row)} values; expected {len(header)}. "
                    "Check separators and missing fields."
                )
            rows.append([value.strip() for value in row])
            if len(rows) > MAX_ROWS:
                raise DataValidationError("The CSV exceeds the 5,000-row limit. Upload only test data.")
    except csv.Error as error:
        raise DataValidationError("The CSV is malformed. Check its quotes and comma separators.") from error
    if not rows:
        raise DataValidationError("The CSV has a header but no session rows.")

    frame = pd.DataFrame(rows, columns=header).loc[:, expected]
    missing_columns = frame.columns[frame.eq("").any()].tolist()
    if missing_columns:
        bad_rows = frame[missing_columns].eq("").any(axis=1)
        target_help = " Remove Revenue entirely for prediction-only data." if TARGET in missing_columns and not require_target else ""
        raise DataValidationError("Missing values in " + ", ".join(missing_columns) + f" ({_rows(bad_rows)}). Fill every required field." + target_help)
    # Reject textual missing-value markers in category fields as well as numbers.
    for column in ["Month", "VisitorType"]:
        missing = frame[column].str.lower().isin({"nan", "null", "none", "na", "n/a"})
        if missing.any():
            raise DataValidationError(f"{column} contains missing values ({_rows(missing)}). Fill every required field.")

    for column in [*NUMERIC_COLUMNS, *CATEGORY_CODE_COLUMNS]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise DataValidationError(f"{column} must contain numeric values ({_rows(values.isna())}). Replace missing or nonnumeric entries.")
        nonfinite = pd.Series(~np.isfinite(values.to_numpy(dtype=float)), index=values.index)
        if nonfinite.any():
            raise DataValidationError(f"{column} must contain finite numbers ({_rows(nonfinite)}); replace infinity.")
        if (values < 0).any():
            raise DataValidationError(f"{column} cannot contain negative values ({_rows(values < 0)}). Use zero or a positive value.")
        if column in COUNT_COLUMNS and (values % 1 != 0).any():
            raise DataValidationError(f"{column} is a page count; use whole numbers ({_rows(values % 1 != 0)}).")
        if column in RATE_COLUMNS and (values > 1).any():
            raise DataValidationError(f"{column} must be between 0 and 1, inclusive ({_rows(values > 1)}).")
        if column in CATEGORY_CODE_COLUMNS:
            invalid = (values < 1) | (values % 1 != 0) | (values > np.iinfo(np.int32).max)
            if invalid.any():
                raise DataValidationError(f"{column} must contain positive integer category codes ({_rows(invalid)}).")
            frame[column] = values.astype(np.int64)
        else:
            if (values > np.finfo(np.float32).max).any():
                raise DataValidationError(f"{column} contains values too large for the saved classifiers.")
            frame[column] = values.astype(float)

    for column in ["Weekend", *([TARGET] if TARGET in frame else [])]:
        values = frame[column].str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if values.isna().any():
            raise DataValidationError(f"{column} must contain True/False or 1/0 ({_rows(values.isna())}).")
        frame[column] = values.astype(bool)
    fingerprint = dataset_fingerprint(frame)
    frame.attrs["dataset_fingerprint"] = fingerprint
    frame.attrs["original_rows"] = {int(index): int(index) + 1 for index in frame.index}
    return frame


def dataset_warnings(frame: pd.DataFrame) -> list[str]:
    """Explain valid inputs that need care when interpreting evaluation results."""
    warnings = []
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        warnings.append(f"This dataset contains {duplicates:,} repeated rows beyond their first occurrence; they are retained in the evaluation.")
    if TARGET not in frame:
        warnings.append("Revenue is absent. Predictions and available scores are shown; observed purchase outcomes are needed for evaluation metrics.")
    elif frame[TARGET].nunique() < 2:
        warnings.append("Only one Revenue class is present. ROC AUC and MCC are unavailable; include purchases and nonpurchases for a complete comparison.")
    for column, known in KNOWN_CATEGORIES.items():
        unknown = set(frame[column].unique()) - known
        if unknown:
            warnings.append(
                f"{column} contains {len(unknown)} category value(s) not seen during training. "
                "The saved encoder ignores those category indicators; interpret their predictions cautiously."
            )
    return warnings


def load_pipeline(model_name: str, root: Path):
    """Load only a registered local model after checking its published checksum."""
    if model_name not in MODEL_FILES:
        raise ValueError("Choose one of the five supported models.")
    root = Path(root)
    relative = "model/" + MODEL_FILES[model_name]
    path = root / relative
    manifest = json.loads((root / "results/experiment_manifest.json").read_text(encoding="utf-8"))
    expected = manifest["artifacts"][relative]
    serialized = path.read_bytes()
    if sha256(serialized).hexdigest() != expected:
        raise ValueError("The saved model does not match the published artifact checksum.")
    return joblib.load(BytesIO(serialized))


def _classes(pipeline) -> list:
    classes = list(pipeline.classes_)
    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError("This application requires binary classes 0 and 1, with purchase as class 1.")
    return classes


def purchase_score(pipeline, features: pd.DataFrame) -> np.ndarray | None:
    """Return scores oriented toward class 1, or None when scores are unavailable."""
    classes = _classes(pipeline)
    if hasattr(pipeline, "predict_proba"):
        probabilities = np.asarray(pipeline.predict_proba(features))
        if probabilities.shape != (len(features), 2):
            raise ValueError("The model returned a probability array with an unexpected shape.")
        score = probabilities[:, classes.index(1)]
    elif hasattr(pipeline, "decision_function"):
        decision = np.asarray(pipeline.decision_function(features))
        if decision.ndim == 1:
            score = decision if classes[1] == 1 else -decision
        elif decision.shape == (len(features), 2):
            score = decision[:, classes.index(1)]
        else:
            raise ValueError("The model returned a decision-score array with an unexpected shape.")
    else:
        return None
    if score.shape != (len(features),) or not np.isfinite(score).all():
        return None
    return np.asarray(score, dtype=float)


def score_label(pipeline) -> str:
    if hasattr(pipeline, "predict_proba"):
        return "Purchase probability"
    return "Decision score" if hasattr(pipeline, "decision_function") else "Unavailable"


def _fitted_fingerprint(estimator) -> str:
    """Hash learned state while excluding unused structured-array padding."""
    state = estimator.__getstate__().copy()
    if "tree_" in state:
        tree_state = state["tree_"].__getstate__().copy()
        nodes = tree_state["nodes"]
        # Tree node records contain unused padding bytes that can change during
        # deserialization. Only named node fields affect the decision rule.
        tree_state["nodes"] = {
            field: np.ascontiguousarray(nodes[field]) for field in nodes.dtype.names
        }
        state["tree_"] = tree_state
    identity = (type(estimator).__module__, type(estimator).__qualname__, state)
    return sha256(joblib.hash(identity).encode("ascii")).hexdigest()


def model_metadata(pipeline, model_name: str | None = None) -> dict:
    """Identify both the fitted classifier and its fitted preprocessing state."""
    classifier = pipeline.named_steps["classifier"]
    prepare = pipeline.named_steps["prepare"]
    classifier_hash = _fitted_fingerprint(classifier)
    preprocessing_hash = _fitted_fingerprint(prepare)
    return {
        "name": model_name or type(classifier).__name__,
        "classifier": type(classifier).__name__,
        "parameters": classifier.get_params(deep=False),
        "positive_class": 1,
        "classes": [int(value) for value in _classes(pipeline)],
        "score_label": score_label(pipeline),
        "model_fingerprint": classifier_hash,
        "preprocessing_fingerprint": preprocessing_hash,
        "identity": sha256((classifier_hash + preprocessing_hash).encode("ascii")).hexdigest(),
        "feature_count": len(FEATURE_COLUMNS),
        "transformed_feature_count": int(classifier.n_features_in_),
    }


def schema_guide(require_target: bool = False) -> pd.DataFrame:
    """Generate field requirements from the same rules used by CSV validation."""
    rows = []
    for column in [*FEATURE_COLUMNS, TARGET]:
        required = "Yes" if column != TARGET or require_target else "For evaluation; optional for predictions"
        if column in COUNT_COLUMNS:
            kind, accepted = "Whole number", "Zero or greater; no missing values"
        elif column in RATE_COLUMNS:
            kind, accepted = "Number", "Between 0 and 1, inclusive"
        elif column in NUMERIC_COLUMNS:
            kind, accepted = "Number", "Finite, zero or greater; no missing values"
        elif column in KNOWN_CATEGORIES:
            kind = "Positive integer code" if column in CATEGORY_CODE_COLUMNS else "Text category"
            known = ", ".join(map(str, sorted(KNOWN_CATEGORIES[column])))
            accepted = "Known: " + known + ". Other valid values are accepted with an unseen-category warning."
        else:
            kind, accepted = "Boolean", "True / False or 1 / 0; case and surrounding spaces are ignored"
        rows.append({
            "Field": column, "Label": FEATURE_LABELS[column], "Required": required,
            "Type": kind, "Accepted values": accepted,
        })
    return pd.DataFrame(rows)


def evaluate_model(pipeline, frame: pd.DataFrame) -> dict:
    """Evaluate fixed model decisions without fitting or changing the pipeline."""
    if frame.empty:
        raise DataValidationError("The dataset contains no usable session rows.")
    _classes(pipeline)
    features = frame.loc[:, FEATURE_COLUMNS]
    labeled = TARGET in frame
    actual = frame[TARGET].astype(int).to_numpy() if labeled else None
    predicted = np.asarray(pipeline.predict(features))
    if predicted.shape != (len(frame),) or not np.isin(predicted, [0, 1]).all():
        raise ValueError("The model returned predictions outside the binary purchase classes.")
    predicted = predicted.astype(int)
    scores = purchase_score(pipeline, features)
    both_classes = labeled and len(np.unique(actual)) == 2
    metrics = {
        "Accuracy": float(accuracy_score(actual, predicted)) if labeled else None,
        "AUC": float(roc_auc_score(actual, scores)) if both_classes and scores is not None else None,
        "Precision": float(precision_score(actual, predicted, zero_division=0)) if labeled else None,
        "Recall": float(recall_score(actual, predicted, zero_division=0)) if labeled else None,
        "F1": float(f1_score(actual, predicted, zero_division=0)) if labeled else None,
        "MCC": float(matthews_corrcoef(actual, predicted)) if both_classes else None,
    }
    report = pd.DataFrame(columns=["precision", "recall", "f1-score", "support"])
    if labeled:
        report_values = classification_report(
            actual, predicted, labels=[0, 1], target_names=["No purchase", "Purchase"],
            output_dict=True, zero_division=np.nan,
        )
        # Overall accuracy is displayed separately; support is a session count.
        report = pd.DataFrame({
            name: values for name, values in report_values.items() if isinstance(values, dict)
        }).T
        report["support"] = report["support"].astype(int)
    predictions = frame.copy()
    fingerprint = dataset_fingerprint(frame)
    source_rows = frame.attrs.get("original_rows", {})
    original_rows = [int(source_rows.get(index, position + 1)) for position, index in enumerate(frame.index)]
    predictions["OriginalRow"] = original_rows
    predictions["SessionID"] = [f"SS-{fingerprint[:12].upper()}-{position:05d}" for position in original_rows]
    predictions["PredictedRevenue"] = predicted.astype(bool)
    predictions["PredictionScore"] = scores if scores is not None else np.nan
    predictions["Correct"] = actual == predicted if labeled else pd.array([pd.NA] * len(frame), dtype="boolean")
    if labeled:
        predictions["ResultCategory"] = np.select(
            [(actual == 0) & (predicted == 0), (actual == 0) & (predicted == 1),
             (actual == 1) & (predicted == 0), (actual == 1) & (predicted == 1)],
            RESULT_CATEGORIES[:4], default="Unlabeled",
        )
    else:
        predictions["ResultCategory"] = "Unlabeled"
    matrix = confusion_matrix(actual, predicted, labels=[0, 1]) if labeled else None
    operating_point = None
    if both_classes and scores is not None:
        threshold = 0.5 if hasattr(pipeline, "predict_proba") else 0.0
        for comparison, at_threshold in [(">", scores > threshold), (">=", scores >= threshold)]:
            if np.array_equal(at_threshold.astype(int), predicted):
                tn, fp, fn, tp = matrix.ravel()
                operating_point = {
                    "fpr": float(fp / (fp + tn)), "recall": float(tp / (tp + fn)),
                    "precision": float(tp / (tp + fp)) if tp + fp else None,
                    "threshold": threshold, "comparison": comparison,
                }
                break
    return {
        "metrics": metrics,
        "confusion": matrix,
        "report": report,
        "predictions": predictions,
        "scores": scores,
        "y_true": actual,
        "score_label": score_label(pipeline) if scores is not None else "Unavailable",
        "score_available": scores is not None,
        "positive_class": 1,
        "labeled": labeled,
        "dataset_fingerprint": fingerprint,
        "operating_point": operating_point,
    }

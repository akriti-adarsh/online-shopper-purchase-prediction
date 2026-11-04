"""Validate labeled session data and evaluate the five published pipelines."""

import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
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


class DataValidationError(ValueError):
    """The uploaded CSV cannot be evaluated with the published pipelines."""


def read_dataset(payload: bytes) -> pd.DataFrame:
    """Read a UTF-8 CSV and return validated features and Boolean outcomes.

    Column order may vary in an upload. The returned frame always follows the
    original feature order and retains Revenue solely for evaluation.
    """
    if not isinstance(payload, bytes):
        raise DataValidationError("Provide the CSV file as bytes.")
    if not payload or not payload.strip():
        raise DataValidationError("The CSV is empty. Upload a labeled test dataset.")
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
        expected = [*FEATURE_COLUMNS, TARGET]
        missing = [name for name in expected if name not in header]
        extra = [name for name in header if name not in expected]
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
        raise DataValidationError("Missing values in: " + ", ".join(missing_columns) + ". Fill every required field.")
    # Reject textual missing-value markers in category fields as well as numbers.
    for column in ["Month", "VisitorType"]:
        if frame[column].str.lower().isin({"nan", "null", "none", "na", "n/a"}).any():
            raise DataValidationError(f"{column} contains missing values. Fill every required field.")

    for column in [*NUMERIC_COLUMNS, *CATEGORY_CODE_COLUMNS]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise DataValidationError(f"{column} must contain numeric values without missing entries.")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise DataValidationError(f"{column} must contain finite numbers; infinity is not supported.")
        if (values < 0).any():
            raise DataValidationError(f"{column} cannot contain negative values.")
        if column in COUNT_COLUMNS and (values % 1 != 0).any():
            raise DataValidationError(f"{column} is a page count and must contain whole numbers.")
        if column in RATE_COLUMNS and (values > 1).any():
            raise DataValidationError(f"{column} must be between 0 and 1, inclusive.")
        if column in CATEGORY_CODE_COLUMNS:
            if (values < 1).any() or (values % 1 != 0).any() or (values > np.iinfo(np.int32).max).any():
                raise DataValidationError(f"{column} must contain positive integer category codes.")
            frame[column] = values.astype(np.int64)
        else:
            if (values > np.finfo(np.float32).max).any():
                raise DataValidationError(f"{column} contains values too large for the saved classifiers.")
            frame[column] = values.astype(float)

    for column in ["Weekend", TARGET]:
        values = frame[column].str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if values.isna().any():
            raise DataValidationError(f"{column} must contain True/False or 1/0 for every session.")
        frame[column] = values.astype(bool)
    return frame


def dataset_warnings(frame: pd.DataFrame) -> list[str]:
    """Explain valid inputs that need care when interpreting evaluation results."""
    warnings = []
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        warnings.append(f"This dataset contains {duplicates:,} repeated rows beyond their first occurrence; they are retained in the evaluation.")
    if frame[TARGET].nunique() < 2:
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


def purchase_score(pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return purchase probabilities or the SVM's uncalibrated ranking scores."""
    if hasattr(pipeline, "predict_proba"):
        positive_column = list(pipeline.classes_).index(1)
        return np.asarray(pipeline.predict_proba(features)[:, positive_column])
    return np.asarray(pipeline.decision_function(features))


def score_label(pipeline) -> str:
    return "Purchase probability" if hasattr(pipeline, "predict_proba") else "Decision score"


def evaluate_model(pipeline, frame: pd.DataFrame) -> dict:
    """Evaluate fixed model decisions without fitting or changing the pipeline."""
    features = frame.loc[:, FEATURE_COLUMNS]
    actual = frame[TARGET].astype(int).to_numpy()
    predicted = np.asarray(pipeline.predict(features), dtype=int)
    scores = purchase_score(pipeline, features)
    both_classes = len(np.unique(actual)) == 2
    metrics = {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "AUC": float(roc_auc_score(actual, scores)) if both_classes else None,
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        "MCC": float(matthews_corrcoef(actual, predicted)) if both_classes else None,
    }
    report_values = classification_report(
        actual, predicted, labels=[0, 1], target_names=["No purchase", "Purchase"],
        output_dict=True, zero_division=0,
    )
    # Overall accuracy is displayed separately; keep support as a session count.
    report = pd.DataFrame({
        name: values for name, values in report_values.items() if isinstance(values, dict)
    }).T
    report["support"] = report["support"].astype(int)
    predictions = frame.copy()
    predictions["PredictedRevenue"] = predicted.astype(bool)
    predictions["PredictionScore"] = scores
    predictions["Correct"] = actual == predicted
    return {
        "metrics": metrics,
        "confusion": confusion_matrix(actual, predicted, labels=[0, 1]),
        "report": report,
        "predictions": predictions,
        "scores": scores,
        "y_true": actual,
        "score_label": score_label(pipeline),
    }

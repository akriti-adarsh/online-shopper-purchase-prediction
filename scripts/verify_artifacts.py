"""Verify the published experiment without fitting any new classifiers."""

from hashlib import sha256
from html import unescape
from importlib.metadata import version
import json
from pathlib import Path
import re
import struct

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"
MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "k-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
COUNTS = ["True negatives", "False positives", "False negatives", "True positives"]
NUMERIC = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CATEGORICAL = [
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]
FIGURES = [
    "class_balance.png", "model_comparison.png", "confusion_matrices.png",
    "roc_curves.png", "model_landscape_3d.png", "confusion_matrices_3d.png",
    "precision_recall_curves.png",
]


def require(condition, message):
    """Use explicit exceptions so verification also runs under python -O."""
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def labels(values):
    encoded = values.astype(str).str.strip().str.lower().map({"false": 0, "true": 1})
    require(encoded.notna().all(), "Revenue contains values other than True/False.")
    return encoded.astype(int)


def check_manifest():
    manifest_path = ROOT / "results/experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    required = {
        "data/online_shoppers_intention.csv", "test_data.csv",
        "results/model_comparison.csv", "results/error_counts.csv",
        "results/training_versions.txt", "results/data_quality.json",
        *(f"model/{filename}" for filename in MODELS.values()),
        *(f"results/{filename}" for filename in FIGURES),
    }
    require(required <= set(artifacts), "Experiment manifest omits required artifacts.")
    for relative, expected_hash in artifacts.items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT), f"Manifest path leaves repository: {relative}")
        require(path.is_file(), f"Manifest artifact is missing: {relative}")
        require(digest(path) == expected_hash, f"Artifact checksum differs: {relative}")
    print(f"Verified checksums for {len(artifacts)} published artifacts.")


def check_versions():
    recorded = (ROOT / "results/training_versions.txt").read_text(encoding="utf-8")
    pins = {}
    for line in recorded.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            package, expected = line.strip().split("==", 1)
            pins[package] = expected
            require(version(package) == expected, f"Install {package}=={expected} before verification.")
    require(
        {"numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib"} <= pins.keys(),
        "Training version record is incomplete.",
    )
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package, expected in pins.items():
        require(f"{package}=={expected}" in requirements, f"Missing runtime pin for {package}.")


def check_data():
    source_path = ROOT / "data/online_shoppers_intention.csv"
    require(digest(source_path) == SOURCE_SHA256, "Source dataset checksum differs from UCI snapshot.")
    source = pd.read_csv(source_path)
    require(source.shape == (12330, 18), "Expected 12,330 source rows and 18 columns.")
    require(set(source.columns) == set(NUMERIC + CATEGORICAL + ["Revenue"]), "Unexpected source schema.")
    require(not source.isna().any().any(), "Source dataset contains missing values.")
    require(int(source.duplicated().sum()) == 125, "Source duplicate count differs.")
    train, expected = train_test_split(
        source, test_size=0.20, stratify=source["Revenue"], random_state=42
    )
    require(train.index.intersection(expected.index).empty, "Training and test row indices overlap.")
    require(len(train) == 9864 and len(expected) == 2466, "Unexpected split sizes.")
    test = pd.read_csv(ROOT / "test_data.csv")
    pd.testing.assert_frame_equal(
        test, expected.reset_index(drop=True), check_dtype=False,
        check_exact=False, rtol=1e-12, atol=1e-12,
        obj="published test data versus seed-42 holdout",
    )
    y = labels(test["Revenue"])
    require(y.value_counts().to_dict() == {0: 2084, 1: 382}, "Unexpected held-out class counts.")
    return train, test.drop(columns="Revenue"), y


def check_models(train, features, actual):
    comparison = pd.read_csv(ROOT / "results/model_comparison.csv", index_col="Model")
    errors = pd.read_csv(ROOT / "results/error_counts.csv", index_col="Model")
    require(comparison.index.is_unique and set(comparison.index) == set(MODELS), "Metric table must list each model once.")
    require(list(comparison.columns) == METRICS, "Metric table columns differ from the six required metrics.")
    require(errors.index.is_unique and set(errors.index) == set(MODELS), "Error table must list each model once.")
    require(set(COUNTS) <= set(errors.columns), "Error table is missing confusion-matrix counts.")
    expected_model_files = {ROOT / "model" / name for name in MODELS.values()}
    require(set((ROOT / "model").glob("*.joblib")) == expected_model_files, "Expected exactly five saved model files.")

    for name, filename in MODELS.items():
        pipeline = joblib.load(ROOT / "model" / filename)
        require(isinstance(pipeline, Pipeline), f"{name}: preprocessing must be saved in a Pipeline.")
        require(list(pipeline.feature_names_in_) == list(features.columns), f"{name}: model input columns differ.")
        require("Revenue" not in pipeline.feature_names_in_, f"{name}: target appears in model inputs.")
        require(list(pipeline.classes_) == [0, 1], f"{name}: unexpected outcome classes.")
        prepare = pipeline.named_steps["prepare"]
        scaler = prepare.named_transformers_["numeric"]
        np.testing.assert_allclose(scaler.mean_, train[NUMERIC].mean().to_numpy(), rtol=1e-12, atol=1e-12)
        require(int(scaler.n_samples_seen_) == len(train), f"{name}: scaler was not fitted on the training subset.")
        encoder = prepare.named_transformers_["category"]
        require(encoder.handle_unknown == "ignore", f"{name}: unseen categories are not handled.")
        for column, observed in zip(CATEGORICAL, encoder.categories_):
            require(set(observed) == set(train[column].unique()), f"{name}: {column} categories differ from training data.")
        transformed = prepare.transform(features.iloc[:5])
        require(np.isfinite(transformed).all(), f"{name}: preprocessing produces non-finite values.")
        require(transformed.shape[1] == pipeline.named_steps["classifier"].n_features_in_, f"{name}: transformed feature count differs.")

        predicted = pipeline.predict(features)
        if hasattr(pipeline, "predict_proba"):
            score = pipeline.predict_proba(features)[:, list(pipeline.classes_).index(1)]
        else:
            score = pipeline.decision_function(features)
        require(np.isfinite(score).all(), f"{name}: model scores contain non-finite values.")
        observed = [
            accuracy_score(actual, predicted), roc_auc_score(actual, score),
            precision_score(actual, predicted, zero_division=0),
            recall_score(actual, predicted, zero_division=0),
            f1_score(actual, predicted, zero_division=0), matthews_corrcoef(actual, predicted),
        ]
        np.testing.assert_allclose(
            observed, comparison.loc[name, METRICS].to_numpy(dtype=float),
            rtol=1e-7, atol=1e-9, err_msg=f"{name}: saved metrics do not match predictions",
        )
        np.testing.assert_array_equal(
            confusion_matrix(actual, predicted, labels=[0, 1]).ravel(),
            errors.loc[name, COUNTS].to_numpy(dtype=int),
            err_msg=f"{name}: confusion counts differ",
        )
        np.testing.assert_array_equal(predicted[:20], pipeline.predict(features.iloc[:20]))
        unknown = features.iloc[[0]].copy()
        unknown["Month"] = "UnseenMonth"
        require(pipeline.predict(unknown)[0] in (0, 1), f"{name}: unknown-category prediction is invalid.")
        print(f"Verified {name}: all six metrics and confusion counts match.")


def check_notebook_and_figures():
    path = ROOT / "model/Online_Shoppers_Model_Analysis.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
    require(bool(code_cells), "Notebook contains no executable analysis.")
    for cell in code_cells:
        require(cell.execution_count is not None, "Notebook contains an unexecuted code cell.")
        require(not any(output.output_type == "error" for output in cell.outputs), "Notebook contains an execution error.")
    outputs = [output for cell in code_cells for output in cell.outputs]
    for filename in FIGURES:
        require(
            any(
                filename in cell.source
                and any("image/png" in output.get("data", {}) for output in cell.outputs)
                for cell in code_cells
            ),
            f"Notebook is missing the rendered chart output for {filename}.",
        )
    tables = [
        output.get("data", {}).get("text/html", "")
        for output in outputs
        if "<table" in output.get("data", {}).get("text/html", "").lower()
    ]
    require(len(tables) >= 4, "Notebook is missing formatted data or evaluation tables.")
    table_texts = [" ".join(unescape(re.sub(r"<[^>]+>", " ", table)).split()) for table in tables]
    require(
        any(all(label in table for label in [*MODELS, *METRICS]) for table in table_texts),
        "Notebook must display all five model names and all six required metrics in one comparison table.",
    )
    for filename in FIGURES:
        contents = (ROOT / "results" / filename).read_bytes()
        require(contents[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG: {filename}")
        require(len(contents) > 1000, f"Figure is unexpectedly small: {filename}")
        width, height = struct.unpack(">II", contents[16:24])
        require(width >= 600 and height >= 300, f"Figure resolution is too low: {filename}")


def check_document_links():
    for path in (ROOT / "README.md", ROOT / "data/README.md"):
        content = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if re.match(r"[a-z][a-z0-9+.-]*:", target, re.IGNORECASE) or target.startswith("#"):
                continue
            local = target.split("#", 1)[0].split("?", 1)[0]
            if local:
                require((path.parent / local).exists(), f"Broken local link in {path.name}: {target}")


def main():
    check_manifest()
    check_versions()
    train, features, actual = check_data()
    check_models(train, features, actual)
    check_notebook_and_figures()
    check_document_links()
    print(
        "Verification passed: 5 models, 6 metrics each, exact holdout, "
        f"executed notebook, formatted tables, and {len(FIGURES)} figures."
    )


if __name__ == "__main__":
    main()

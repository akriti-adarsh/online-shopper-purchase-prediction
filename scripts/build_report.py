"""Build supporting figures and an auditable manifest from the fitted experiment."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from notebook_visuals import class_balance_figure, comparison_figure


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = [
    "Logistic Regression", "Decision Tree", "k-Nearest Neighbors",
    "Gaussian Naive Bayes", "Support Vector Machine",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    data_path = ROOT / "data" / "online_shoppers_intention.csv"
    data = pd.read_csv(data_path)
    test = pd.read_csv(ROOT / "test_data.csv")
    training, original_test = train_test_split(
        data, test_size=0.2, stratify=data["Revenue"], random_state=42
    )
    X = test.drop(columns="Revenue")
    y = test["Revenue"].astype(int)
    comparison = pd.read_csv(results / "model_comparison.csv", index_col="Model")

    error_rows = []
    model_metadata = {}
    model_paths = []
    for name in MODEL_NAMES:
        slug = name.lower().replace("-", "_").replace(" ", "_")
        path = ROOT / "model" / f"{slug}.joblib"
        pipeline = joblib.load(path)
        tn, fp, fn, tp = confusion_matrix(y, pipeline.predict(X), labels=[0, 1]).ravel()
        error_rows.append({
            "Model": name, "True negatives": int(tn), "False positives": int(fp),
            "False negatives": int(fn), "True positives": int(tp),
        })
        classifier = pipeline.named_steps["classifier"]
        model_metadata[name] = {
            "file": path.relative_to(ROOT).as_posix(),
            "estimator": type(classifier).__name__,
            "parameters": classifier.get_params(deep=False),
            "ranking_score": "purchase probability" if hasattr(pipeline, "predict_proba")
                             else "decision function",
        }
        model_paths.append(path)
    pd.DataFrame(error_rows).to_csv(results / "error_counts.csv", index=False)

    features = data.columns.drop("Revenue").tolist()
    training_features = set(training[features].itertuples(index=False, name=None))
    test_features = list(original_test[features].itertuples(index=False, name=None))
    quality = {
        "source_rows": len(data),
        "input_features": len(features),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows_beyond_first": int(data.duplicated().sum()),
        "duplicate_feature_rows_beyond_first": int(data[features].duplicated().sum()),
        "test_rows_with_features_also_in_training": sum(
            row in training_features for row in test_features
        ),
        "distinct_feature_vectors_shared_across_split": len(
            training_features.intersection(test_features)
        ),
        "overlapping_source_row_indices": len(training.index.intersection(original_test.index)),
        "handling": "Original rows retained to reproduce the documented experiment.",
    }
    write_json(results / "data_quality.json", quality)

    counts = data["Revenue"].astype(int).value_counts().reindex([0, 1])
    for figure, filename in [
        (class_balance_figure(counts), "class_balance.png"),
        (comparison_figure(comparison), "model_comparison.png"),
    ]:
        figure.savefig(results / filename, dpi=170, bbox_inches="tight",
                       facecolor=figure.get_facecolor())
        plt.close(figure)
    labels = comparison.sort_values(["F1", "MCC"], ascending=False).index.to_list()

    packages = ["scikit-learn", "numpy", "pandas", "scipy", "joblib", "matplotlib", "threadpoolctl"]
    artifact_paths = [
        data_path, ROOT / "test_data.csv", *model_paths,
        ROOT / "model" / "Online_Shoppers_Model_Analysis.ipynb",
        ROOT / "scripts" / "notebook_visuals.py",
        ROOT / "scripts" / "notebook_tables.py",
        *(results / name for name in [
            "model_comparison.csv", "error_counts.csv", "training_versions.txt",
            "data_quality.json", "class_balance.png", "model_comparison.png",
            "confusion_matrices.png", "roc_curves.png", "model_landscape_3d.png",
            "confusion_matrices_3d.png", "precision_recall_curves.png",
        ]),
    ]
    manifest = {
        "dataset": {"doi": "10.24432/C5F88Q", "source_sha256": sha256(data_path)},
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in packages},
        "target": "Revenue",
        "positive_class": 1,
        "input_columns": features,
        "split": {
            "method": "stratified random split", "random_state": 42, "test_fraction": 0.2,
            "training_rows": len(training), "test_rows": len(test),
            "training_purchases": int(training["Revenue"].sum()),
            "test_purchases": int(y.sum()),
            "test_row_indices": original_test.index.to_list(),
        },
        "selection": {
            "primary_metric": "F1", "tiebreaker": "MCC", "highest_ranked_model": labels[0],
            "scope": "Comparison on one test split; no independent final selection evaluation.",
        },
        "models": model_metadata,
        "artifacts": {path.relative_to(ROOT).as_posix(): sha256(path) for path in artifact_paths},
    }
    write_json(results / "experiment_manifest.json", manifest)
    print("Built two supporting figures, error counts, data-quality report, and experiment manifest.")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()

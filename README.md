# Online Shopper Purchase Prediction

[![Validate experiment artifacts](https://github.com/akriti-adarsh/online-shopper-purchase-prediction/actions/workflows/validate.yml/badge.svg)](https://github.com/akriti-adarsh/online-shopper-purchase-prediction/actions/workflows/validate.yml)

A reproducible comparison of five classifiers for identifying shopping sessions that end in a purchase. The project includes an executed notebook, fitted preprocessing and model pipelines, labeled test data, evaluation tables, and visual results.

**Main finding:** the Decision Tree achieves the highest purchase F1 (**0.6167**) on the shared test split, with **0.8921 accuracy**, **0.9129 ROC AUC**, and **0.5585 MCC**. These results describe this experiment; they do not establish performance on future traffic.

## 1. Problem statement

The task is binary classification: predict whether an online shopping session produces a transaction from its recorded browsing activity and visitor context. `Revenue = True` is the positive class, representing a purchase; `Revenue = False` represents a session without a purchase.

Purchase sessions are a minority of the data. A model can therefore achieve high accuracy while identifying few buyers. This comparison uses **purchase F1 as the primary ranking measure**, with MCC as a tiebreaker, and reports accuracy, ROC AUC, precision, recall, and MCC to make the tradeoffs visible.

Logistic Regression, Decision Tree, k-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine use identical training and test rows. The experiment assesses their relative behavior on recorded sessions. An early-session intervention would require a separate evaluation of which features are available when the prediction is made.

## 2. Dataset description

The project uses the **Online Shoppers Purchasing Intention Dataset**, published by C. Sakar and Y. Kastro through the [UCI Machine Learning Repository](https://doi.org/10.24432/C5F88Q). Each record describes one shopping session. The source data is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribution and the original file checksum are recorded in [data/README.md](data/README.md).

| Property | Value |
|---|---:|
| Sessions | 12,330 |
| Input features | 17 |
| Numerical inputs | 10 |
| Categorical inputs | 7 |
| Target | `Revenue` |
| Purchase sessions | 1,908 (15.47%) |
| Nonpurchase sessions | 10,422 (84.53%) |
| Missing values | 0 |
| Exact duplicate rows beyond their first occurrence | 125 |

The original records, including duplicate rows, are retained to preserve the experiment's split. **29 test rows share feature values with training rows**, representing 22 distinct feature patterns. Source row indices remain disjoint. Duplicate counts and overlap between the partitions are reported in [results/data_quality.json](results/data_quality.json); matching feature patterns are a limitation of this benchmark.

### Input features

The feature groups below cover all 17 predictors. Integer category identifiers are treated as categories, rather than as quantities with meaningful distances.

| Features | Meaning | Processing |
|---|---|---|
| `Administrative`, `Administrative_Duration` | Administrative page count and duration | Standard scaling |
| `Informational`, `Informational_Duration` | Informational page count and duration | Standard scaling |
| `ProductRelated`, `ProductRelated_Duration` | Product page count and duration | Standard scaling |
| `BounceRates`, `ExitRates` | Page-level bounce and exit measures | Standard scaling |
| `PageValues` | Page value associated with transaction activity | Standard scaling |
| `SpecialDay` | Proximity to a special shopping day | Standard scaling |
| `Month` | Session month | One-hot encoding |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | Coded session context | One-hot encoding |
| `VisitorType` | Visitor category | One-hot encoding |
| `Weekend` | Weekend indicator | One-hot encoding |

### Shared evaluation split

A stratified 80/20 split with `random_state=42` preserves the approximate purchase proportion. `Revenue` is retained in the exported CSV for scoring and excluded from model inputs.

| Partition | Sessions | Purchases | Nonpurchases | Purchase rate |
|---|---:|---:|---:|---:|
| Training | 9,864 | 1,526 | 8,338 | 15.47% |
| Held-out test | 2,466 | 382 | 2,084 | 15.49% |

On this test set, always predicting “no purchase” produces **84.51% accuracy and 0% purchase recall**. This reference explains why accuracy alone is insufficient.

## 3. GitHub repository link

[akriti-adarsh/online-shopper-purchase-prediction](https://github.com/akriti-adarsh/online-shopper-purchase-prediction)

The analysis is available in [model/Online_Shoppers_Model_Analysis.ipynb](model/Online_Shoppers_Model_Analysis.ipynb). The repository currently covers modeling and evaluation; the Streamlit application is the next development stage.

## 4. Models used

### Preprocessing and configuration

Each classifier is stored in a scikit-learn `Pipeline` with its own fitted `ColumnTransformer`. Numerical inputs use `StandardScaler`; categorical inputs use `OneHotEncoder(handle_unknown="ignore")`. The combined transformed output is dense for compatibility with Gaussian Naive Bayes.

The scaler and encoder learn only from training rows. Test rows use those fitted transformations. Keeping preprocessing inside each saved pipeline ensures the same transformations apply when the model is reloaded.

| Model | Explicit settings | Role in the comparison |
|---|---|---|
| Logistic Regression | `max_iter=1000`, `random_state=42` | Linear reference model |
| Decision Tree | `max_depth=7`, `min_samples_leaf=15`, `random_state=42` | Nonlinear rules with constrained tree growth |
| k-Nearest Neighbors | `n_neighbors=11` | Classification from nearby training sessions |
| Gaussian Naive Bayes | `var_smoothing=0.001` | Probabilistic classifier with variance smoothing |
| Support Vector Machine | `kernel="rbf"`, `C=1.0` | Nonlinear margin-based classifier |

Other settings use the defaults of the pinned library version. The experiment does not perform hyperparameter search, resampling, class weighting, or threshold optimization.

### Metric definitions

Precision, recall, and F1 refer to the **purchase class**. Predicted labels come from each estimator's `predict` method at its default decision rule.

| Metric | Interpretation |
|---|---|
| Accuracy | Share of all sessions classified correctly |
| ROC AUC | Ability to rank purchases above nonpurchases across thresholds |
| Precision | Share of predicted purchases that are actual purchases |
| Recall | Share of actual purchases identified |
| F1 | Harmonic mean of purchase precision and recall |
| MCC | Correlation between predicted and observed classes, using all four confusion-matrix counts |

ROC AUC uses purchase probabilities for the first four models and `decision_function` scores for the SVM. SVM scores are ranking values, not calibrated probabilities. MCC ranges from −1 to 1; the other metrics range from 0 to 1. Higher values are better for all six metrics.

### Performance comparison

All models are evaluated on the same **2,466 test sessions**. Values are rounded to four decimal places; [model_comparison.csv](results/model_comparison.csv) retains the underlying numerical results.

| Model | Accuracy | ROC AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8808 | 0.8828 | **0.7366** | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | **0.8921** | **0.9129** | 0.6859 | 0.5602 | **0.6167** | **0.5585** |
| k-Nearest Neighbors | 0.8731 | 0.8164 | 0.6927 | 0.3246 | 0.4421 | 0.4159 |
| Gaussian Naive Bayes | 0.6403 | 0.7681 | 0.2723 | **0.7906** | 0.4051 | 0.2934 |
| Support Vector Machine | 0.8873 | 0.8824 | 0.7047 | 0.4686 | 0.5629 | 0.5149 |

**Bold values identify the highest score in each column.**

## 5. Model observations

| Model | Observation on this test split |
|---|---|
| Logistic Regression | Highest precision: 137 of its 186 purchase alerts are correct. However, it misses 245 buyers, giving 35.86% recall. It produces relatively few false purchase alerts. |
| Decision Tree | Identifies 214 buyers with 98 false alerts. Compared with Logistic Regression, it catches 77 more buyers at the cost of 49 more false alerts. It leads on accuracy, ROC AUC, F1, and MCC. |
| k-Nearest Neighbors | Identifies 124 buyers and misses 258. Precision is 69.27%, but recall is 32.46%, the lowest in the comparison. Its high overall accuracy masks weak purchase detection. |
| Gaussian Naive Bayes | Identifies the most buyers: 302 of 382. It also produces 807 false alerts; just 27.23% of purchase alerts are correct. Its recall advantage comes with a substantial precision cost. |
| Support Vector Machine | Identifies 179 buyers with 75 false alerts. Compared with the tree, it produces 23 fewer false alerts but misses 35 additional buyers. Its F1 is second highest at 0.5629. |
| **Overall winner** | **Decision Tree**, under the F1-first rule. Its F1 of **0.6167** and MCC of **0.5585** provide the strongest balance in this experiment. This is a selection from one shared test split, not an independent estimate after model selection. |

### Errors in session counts

| Model | True negatives | False positives | False negatives | True positives |
|---|---:|---:|---:|---:|
| Logistic Regression | 2,035 | 49 | 245 | 137 |
| Decision Tree | 1,986 | 98 | 168 | 214 |
| k-Nearest Neighbors | 2,029 | 55 | 258 | 124 |
| Gaussian Naive Bayes | 1,277 | 807 | 80 | 302 |
| Support Vector Machine | 2,009 | 75 | 203 | 179 |

A false positive is a nonpurchase session labeled as a purchase. A false negative is a purchase labeled as a nonpurchase. Every row sums to 2,466; true positives plus false negatives sum to 382 buyers. These counts are also available in [error_counts.csv](results/error_counts.csv).

## 6. Visual results

### Class distribution

![Class distribution showing purchase and nonpurchase sessions](results/class_balance.png)

The minority purchase class makes precision and recall essential companions to accuracy.

### Comparison across models

![Model comparison chart for the five classifiers](results/model_comparison.png)

The comparison highlights the Decision Tree's balance and Gaussian Naive Bayes' recall tradeoff.

### Confusion matrices

![Five confusion matrices with actual classes on rows and predicted classes on columns](results/confusion_matrices.png)

All panels share one color scale. The upper-right cell counts false purchase alerts; the lower-left cell counts missed buyers.

### ROC curves

![ROC curves comparing purchase ranking across decision thresholds](results/roc_curves.png)

The diagonal is the chance-ranking reference. ROC AUC evaluates ranking across thresholds; it does not describe precision at the default operating point.

## 7. Reproduce the experiment

### Environment setup

Use **Python 3.12**. From a terminal:

```bash
git clone https://github.com/akriti-adarsh/online-shopper-purchase-prediction.git
cd online-shopper-purchase-prediction
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the analysis and notebook dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

[requirements.txt](requirements.txt) pins modeling and reporting dependencies. [requirements-dev.txt](requirements-dev.txt) adds notebook execution and editing tools. The versions used to generate the committed models are recorded in [training_versions.txt](results/training_versions.txt).

### Regenerate the outputs

```bash
python scripts/run_analysis.py
python scripts/verify_artifacts.py
```

The analysis command executes the notebook and rebuilds the reports from the repository's source dataset. It regenerates the split CSV files, fits all five pipelines, exports their models, and refreshes the results. This overwrites generated experiment outputs; inspect the Git diff after rerunning it.

To inspect or execute the notebook interactively:

```bash
python -m jupyterlab model/Online_Shoppers_Model_Analysis.ipynb
```

The source dataset is included, so a fresh clone does not depend on a separate download. `train_data.csv` is reproducible and ignored by Git; the labeled `test_data.csv` is tracked for evaluation and the planned application.

### Validate existing artifacts

```bash
python scripts/verify_artifacts.py
```

Verification reloads the saved pipelines and checks their results against the recorded experiment. The [validation workflow](.github/workflows/validate.yml) runs this check in GitHub Actions. The experiment manifest and data-quality report retain provenance and split information for inspection.

Serialized pipelines depend on their library versions. Use the pinned environment when loading the committed `.joblib` files. If dependencies change, regenerate the complete set of models and reports together; see scikit-learn's [model persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

## 8. Repository contents

| Path | Purpose |
|---|---|
| `README.md` | Problem, dataset, methods, findings, and reproduction instructions |
| `requirements.txt` | Pinned modeling and reporting dependencies |
| `requirements-dev.txt` | Notebook execution and editing dependencies |
| `data/online_shoppers_intention.csv` | Original dataset for repeatable execution |
| `data/README.md` | Dataset attribution, license, and checksum |
| `model/Online_Shoppers_Model_Analysis.ipynb` | Executed analysis with all five model implementations |
| `model/logistic_regression.joblib` | Fitted Logistic Regression pipeline |
| `model/decision_tree.joblib` | Fitted Decision Tree pipeline |
| `model/k_nearest_neighbors.joblib` | Fitted k-Nearest Neighbors pipeline |
| `model/gaussian_naive_bayes.joblib` | Fitted Gaussian Naive Bayes pipeline |
| `model/support_vector_machine.joblib` | Fitted SVM pipeline |
| `test_data.csv` | Shared labeled test set, including `Revenue` |
| `results/model_comparison.csv` | Six evaluation metrics for every model |
| `results/error_counts.csv` | Confusion-matrix counts for every model |
| `results/*.png` | Four visual summaries embedded above |
| `results/training_versions.txt` | Package versions recorded during execution |
| `results/experiment_manifest.json` | Experiment provenance and artifact metadata |
| `results/data_quality.json` | Data quality and partition-overlap information |
| `scripts/run_analysis.py` | Entry point for rerunning the notebook and reports |
| `scripts/build_report.py` | Builds supplementary tables and visual reports |
| `scripts/verify_artifacts.py` | Checks saved models and reported results |
| `.github/workflows/validate.yml` | Automated repository validation |
| `.gitignore`, `.gitattributes` | Exclude local files and keep file handling consistent |

## 9. Limitations and interpretation

- **One evaluation split.** There are no confidence intervals or repeated-split estimates. Ranking models on this test set also uses it for selection. Further work should select models and tune settings within training data, then evaluate once on an untouched final set.
- **Duplicate records.** The source contains 125 exact duplicate rows beyond their first occurrence. Retaining these rows preserves comparability, but matching feature patterns can cross the split. A separate duplicate-aware evaluation would test sensitivity to that choice.
- **Feature availability.** `PageValues`, durations, and related measures reflect recorded session activity. Their availability and construction must be checked at the intended prediction time before interpreting these results as early purchase prediction.
- **Threshold and cost assumptions.** F1 weights precision and recall equally. It does not encode the business cost of a false alert or missed buyer. Different costs could favor another model or threshold.
- **Historical scope.** A random split does not measure performance on a later period, another retailer, or changed shopping behavior.
- **Model assumptions.** Gaussian Naive Bayes applies conditional independence and Gaussian assumptions to transformed features. Correlated activity measures and encoded categories may depart from those assumptions; probability calibration is not evaluated here.

## 10. Next development stage

The next stage is a Streamlit interface with labeled test CSV upload, selection among the five saved pipelines, the six metrics, and a confusion matrix or classification report. The entry point and deployment link will be added when that stage is implemented and verified.

The repository does not yet include a live application or evidence of execution in the BITS Virtual Lab. Those deliverables, followed by the final submission PDF, remain separate steps.

## References

1. Sakar, C., & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C5F88Q). [Dataset license](https://creativecommons.org/licenses/by/4.0/).
2. Scikit-learn documentation: [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html).
3. Scikit-learn documentation: [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
4. Scikit-learn documentation: [Model persistence](https://scikit-learn.org/stable/model_persistence.html).

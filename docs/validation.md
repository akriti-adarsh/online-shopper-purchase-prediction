# Application validation record

Recorded on **4 November 2026** for Shopper Signals, covering local validation with Python 3.12 and Streamlit 1.66.0 and the subsequent [Streamlit Community Cloud deployment](https://shopper-signals.streamlit.app/). The local and hosted checks are identified separately below.

## Automated checks

Run these commands from the repository root in the activated Python environment:

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q app.py shopper_app
python -m ruff check app.py shopper_app tests
python -m unittest discover -s tests -v
python scripts/verify_artifacts.py
```

| Check | Recorded result |
|---|---|
| Python unit tests | **64 passed**, covering dataset validation, saved-model evaluation, chart calculations, explorer behavior, and Streamlit interactions |
| Browser-component tests | **4 passed**, covering native tab order, hidden targets, delayed rendering, and successful focus confirmation |
| Ruff 0.17.3 | Clean for `app.py`, `shopper_app`, and `tests` |
| Python compilation | Passed |
| Browser-component JavaScript syntax | Passed |
| Experiment integrity | All **21 artifact checksums** matched |
| Saved-model regression | All five classifiers reproduced all six published metrics and their confusion-matrix counts |

The tests cover positive-class score mapping, SVM score semantics, supported and unavailable scores, repeated scores, verified operating points, single-class inputs, optional labels, upload limits, validation errors, category handling, stable session identifiers, all four outcome filters, pagination, sorting, filtered exports, and recovery after invalid input. Comparison tests cover all six metrics, ties, unavailable values, and negative MCC ranges.

The notebook was not rerun and the fitted scientific artifacts were not regenerated for this interface update. Artifact verification reloads the saved models and checks the recorded experiment. No separate JavaScript production build is required: Streamlit serves the small native browser components directly, and production does not require Node.js.

Run the dependency-free component tests with Node.js 22 or newer: `node --test tests/test_components.mjs`. The hosted validation workflow runs these alongside the Python checks.

## Follow-up audit

The audit reproduced and corrected the following issues. Regression tests failed against the previous behavior before the fixes were applied.

| Issue | Correction and verification |
|---|---|
| Clicking the active navigation option cleared its selection and could jump to Overview | Preserve the last valid destination, including after matrix drill-down and return; checked across all four destinations |
| Clicking the active percentage option silently switched the matrix to counts | Retain the selected matrix display mode; checked for both counts and percentages |
| Restoring focus removed a native button from sequential keyboard navigation | Preserve the button's native tab index; live browser return focused the missed-buyer button and Tab moved to the next outcome |
| A hidden target could consume the focus request before results became visible | Wait for visible content, observe visibility changes, and confirm focus actually landed; checked with component regressions and live drill-down |
| Formula-like text in uploaded categories was exported as executable spreadsheet input | Prefix those string values with an apostrophe in the exported copy; preserve original inputs, row IDs, and numeric negative scores |

The audit reviewed data validation, trusted model loading, positive-class score mapping, session state, chart calculations, escaped HTML, and export behavior. It found no further reproducible failures in those checks. This is a bounded audit, not a guarantee that every environment and input is defect-free.

## Reference result checked in the browser

The bundled test set contains **2,466 sessions**, including **382 purchases** and **2,084 nonpurchases**. Selecting **k-Nearest Neighbors** reproduced these displayed values:

| Metric | Displayed value |
|---|---:|
| Accuracy | 0.8731 |
| ROC AUC | 0.8164 |
| Precision | 0.6927 |
| Recall | 0.3246 |
| F1 | 0.4421 |
| MCC | 0.4159 |

The full-precision reference remains in [model_comparison.csv](../results/model_comparison.csv).

| Actual outcome | Predicted no purchase | Predicted purchase |
|---|---:|---:|
| No purchase | **2,029 TN** | **55 FP** |
| Purchase | **258 FN** | **124 TP** |

These counts sum to 2,466. The missed-buyer drill-down displayed **258 matching sessions**. Its downloaded CSV contained **258 unique session identifiers**, and every exported record was a false negative.

## Local browser checks

| Scenario | Observed result |
|---|---|
| Bundled evaluation | k-Nearest Neighbors metrics and all four matrix counts matched the reference above |
| Matrix drill-down | Selecting missed buyers opened Session Explorer with exactly 258 matching rows |
| Filtered export | Download contained all 258 filtered sessions, with unique stable identifiers and only false negatives |
| Keyboard session details | Enter on a session row opened its details; closing the dialog completed the return flow |
| Valid labeled upload | An 80-row file activated successfully and updated the active context |
| Invalid replacement upload | A file with missing headers showed validation feedback and retained the previous valid 80-row dataset |
| Feature-only upload | A 20-row file displayed predictions and scores, with observed outcomes and evaluation results unavailable |
| Return to bundled data | The included test set could be restored after the upload checks |
| Model comparison | MCC comparison displayed all five models on the same active dataset |
| Appearance persistence | Dark appearance remained selected after reload |
| Theme consistency | Native controls and charts both followed the selected appearance |

These browser checks supplement the automated tests. They do not imply that every possible browser interaction or input combination was manually exercised.

## Local responsive and visual checks

DOM measurements at widths of **360, 390, 768, 1024, and 1440 pixels** found no page-level horizontal overflow. At **390 × 844 pixels**, checks also confirmed that buyer coverage appears before the matrix, settings and session-detail dialogs remain within the viewport, and the keyboard detail flow remains usable. The final mobile keyboard pass followed the 258 missed buyers into a session, closed its details, and returned to Overview. An additional 727-pixel review confirmed the Evaluation settings control remained readable after the header stacking adjustment. These width checks do not replace testing zoom or every internal scrolling surface.

Visual review identified and corrected caption opacity and an outcome-legend font mismatch. The release screenshots show both appearances and both layout sizes:

| Screenshot | Frame size |
|---|---:|
| [Desktop, light](screenshots/desktop-light.png) | 1440 × 1360 |
| [Desktop, dark](screenshots/desktop-dark.png) | 1440 × 1360 |
| [Mobile, light](screenshots/mobile-light.png) | 390 × 1680 |
| [Mobile, dark](screenshots/mobile-dark.png) | 390 × 1680 |

The taller mobile screenshot frames show more of the page. Normal mobile interaction was checked separately at 390 × 844 pixels.

The exact base palette pairs were also checked using relative luminance. Main text, secondary text, action text, and highlight text all exceed a 4.5:1 contrast ratio in both appearances; the ratios are recorded in the [README](../README.md#appearance-and-accessibility). This calculation does not certify every widget state, chart mark, or dynamically rendered surface.

The subsequent color refresh replaces the muted palette with cobalt, coral, mint, and amber on white or navy surfaces. All 14 existing chart tests passed for the refreshed colors. Text on outcome cells and coverage bars exceeds 4.5:1 in both appearances, and input borders exceed 3:1 against their panel surfaces. Desktop and mobile screenshots were refreshed after checking both themes; evaluation behavior and scientific artifacts are unchanged.

## Streamlit Community Cloud verification

The public application is **[shopper-signals.streamlit.app](https://shopper-signals.streamlit.app/)**. Deployment and the checks below completed on **4 November 2026**, using repository `akriti-adarsh/online-shopper-purchase-prediction`, branch `main`, and entry point `app.py`. Advanced settings explicitly selected Python **3.12**; the build log reported Python **3.12.16** and Streamlit **1.66.0**.

| Hosted scenario | Observed result |
|---|---|
| Cloud startup | Dependency installation completed and the interactive application rendered successfully |
| Default evaluation | Decision Tree displayed the reference metrics for 2,466 sessions, including 382 purchasers |
| Five-model comparison | All five classifiers rendered; the downloaded comparison CSV matched all **30 metric values at full precision** in the committed reference, mapping its `ROC AUC` heading to `AUC` |
| Matrix drill-down | k-Nearest Neighbors missed buyers opened Session Explorer with **258 matching sessions** |
| Filtered download | The actual downloaded CSV contained **258 rows and 258 unique session IDs**, all false negatives; source features, labels, IDs, and scores matched the locally evaluated reference exactly |
| Keyboard details and return | Enter opened a session's details; return through the session list to Overview restored focus to the missed-buyer button, and Tab moved to the identified-buyer button |
| Valid labeled upload | The 80-row fixture activated with **17 purchasers** and **21.25%** purchase prevalence |
| Invalid replacement upload | Missing-column feedback appeared and the previously valid 80-row dataset remained active |
| Feature-only upload | A 20-row file produced **4 k-Nearest Neighbors purchase predictions**; all six evaluation metrics were unavailable because observed outcomes were absent |
| Return to bundled data | Restoring the included test set returned the active context to **2,466 sessions** |
| SVM score semantics | Session Explorer displayed **Decision score**, including negative values, with the explanation that these are uncalibrated ranking scores rather than probabilities |
| Appearance persistence | Dark remained selected after reload; the dataset and model returned to their expected session defaults |
| Hosted desktop layout | At **1440 × 1360 pixels** in light appearance, the page had no horizontal overflow; the main content's client and scroll widths were both 1,430 pixels |
| Hosted mobile layout | At **390 × 844 pixels** in dark appearance, the page had no horizontal overflow; the main content's client and scroll widths were both 380 pixels, and settings stayed between horizontal coordinates 14 and 380 |
| Project information | About displayed the dataset, preprocessing, model limitations, upload behavior, and active model identifiers; navigation returned to Overview |
| Public HTTP access | A request starting without account or browser cookies reached the public URL with HTTP 200 after the platform's anonymous-session redirects |

The public HTTP response was the platform's JavaScript bootstrap. This confirms access to that public entry point; the interactive browser checks above establish application behavior. The platform also returned bootstrap HTML for `/_stcore/health`, so no independent backend-health claim is based on that endpoint. An independent signed-out interactive-browser session was not exercised.

The cloud platform replaced the initially resolved, unpinned PyArrow **26.0.1** with **24.0.0** during setup, reporting a known upstream issue. The application then started and passed the recorded checks. The pinned training and inference package versions remained those in `requirements.txt`; this host-managed transitive dependency change is recorded for reproducibility.

The upload fixtures contain anonymous session features from the bundled public dataset, or synthetic invalid headers. They do not introduce private customer records. The [hosted overview screenshot](screenshots/hosted-overview.png) records the deployed interface.

## Verification limits and remaining checks

- **200% zoom: unverified.** The attempted keyboard zoom shortcut had no effect in the embedded browser. No zoom-pass claim is made; repeat the check in a browser with verifiable zoom controls.
- **Assistive technology: not tested.** Screen-reader behavior and a complete WCAG 2.2 audit remain outstanding. Accessibility is a design target, not a certification.
- **Initial theme paint:** Streamlit owns the page bootstrap. Browser preferences reach Python after the appearance component mounts, so a brief initial palette mismatch remains possible.
- **Remaining browser coverage:** repeat manual single-class and empty-filter flows, every outcome-button route, system-preference changes, and reduced-motion behavior as part of broader release checks. Corresponding calculation and state cases are covered where applicable by the automated tests; that is distinct from a full manual browser pass.

The assignment also still requires the BITS Virtual Lab execution screenshot and the final submission PDF containing the repository link, live application link, screenshot, and complete README.

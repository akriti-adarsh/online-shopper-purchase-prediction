# Application validation record

Recorded on **4 November 2026** for the local Shopper Signals application, using Python 3.12 and Streamlit 1.66.0 with the repository's pinned dependencies. This record covers the completed checks below. The application has **not yet been deployed to Streamlit Community Cloud**.

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
| Unit tests | **61 passed**, covering dataset validation, saved-model evaluation, chart calculations, explorer behavior, and Streamlit interactions |
| Ruff 0.17.3 | Clean for `app.py`, `shopper_app`, and `tests` |
| Python compilation | Passed |
| Browser-component JavaScript syntax | Passed |
| Experiment integrity | All **21 artifact checksums** matched |
| Saved-model regression | All five classifiers reproduced all six published metrics and their confusion-matrix counts |

The tests cover positive-class score mapping, SVM score semantics, supported and unavailable scores, repeated scores, verified operating points, single-class inputs, optional labels, upload limits, validation errors, category handling, stable session identifiers, all four outcome filters, pagination, sorting, filtered exports, and recovery after invalid input. Comparison tests cover all six metrics, ties, unavailable values, and negative MCC ranges.

The notebook was not rerun and the fitted scientific artifacts were not regenerated for this interface update. Artifact verification reloads the saved models and checks the recorded experiment. No separate JavaScript production build is required: Streamlit serves the small native browser components directly, and production does not require Node.js.

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

## Live browser checks

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

## Responsive and visual checks

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

## Verification limits and remaining checks

- **200% zoom: unverified.** The attempted keyboard zoom shortcut had no effect in the embedded browser. No zoom-pass claim is made; repeat the check in a browser with verifiable zoom controls.
- **Assistive technology: not tested.** Screen-reader behavior and a complete WCAG 2.2 audit remain outstanding. Accessibility is a design target, not a certification.
- **Initial theme paint:** Streamlit owns the page bootstrap. Browser preferences reach Python after the appearance component mounts, so a brief initial palette mismatch remains possible.
- **Hosted behavior: unverified.** Deploy to Streamlit Community Cloud, then repeat the main evaluation, upload, download, theme, and keyboard flows against the public URL.
- **Remaining browser coverage:** repeat manual single-class and empty-filter flows, every outcome-button route, system-preference changes, and reduced-motion behavior as part of broader release checks. Corresponding calculation and state cases are covered where applicable by the automated tests; that is distinct from a full manual browser pass.

The assignment also still requires the BITS Virtual Lab execution screenshot and the final submission PDF containing the repository link, live application link, screenshot, and complete README.

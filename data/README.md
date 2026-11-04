# Dataset provenance

`online_shoppers_intention.csv` is the original UCI CSV, copied without changing its bytes. It contains 12,330 sessions, 17 input features, and the `Revenue` outcome. Keeping this small source file in the repository makes the experiment reproducible without a network download.

**Source:** Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C5F88Q).

**License:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), as stated on the [UCI dataset page](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). This attribution also applies to the derived training and test CSV files. The dataset license does not grant a separate license to the project source code.

**Original archive:** [Download from UCI](https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip).

| Property | Value |
|---|---|
| CSV size | 1,072,063 bytes |
| SHA-256 | `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9` |
| Missing values | 0 |
| Exact duplicate rows beyond the first occurrence | 125 |
| Changes to source CSV | None |

The analysis preserves the original row order and repeated rows. It creates a stratified 80/20 split with `random_state=42`; `test_data.csv` contains the held-out rows and their labels. Exporting the subsets changes their row order and CSV formatting, but does not alter their values. `train_data.csv` is regenerated locally and excluded from version control because the complete source and split rule are already included.

Repeated feature values can appear on both sides of a random split. `results/data_quality.json` records the measured overlap; the reported results should be interpreted with that limitation.

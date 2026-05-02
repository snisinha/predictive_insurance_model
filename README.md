# Predictive Insurance Model — Final Report

Binary classification of **car insurance claims**: predict `is_claim` (0/1) from vehicle and policy attributes. This repository contains the full pipeline—loading, cleaning, exploratory analysis, feature encoding, resampling, model training, and evaluation.

**Video Presentation:** [YouTube](https://youtu.be/GmfD6dbCbRY)

---

## 1. How to build and run 

**Use the Makefile first.** It creates a virtual environment, installs dependencies from `requirements.txt`, and runs tests or the full pipeline.

### Prerequisites

- **Python 3.10+** (3.11 is used in GitHub Actions)
- **GNU Make**
- **Dataset file**: placed the dataset at  
  `datasets/dataset_main.csv`  
  (see [Data collection](#3-data-collection) for the source and how this file is produced.)

### Commands

| Command | What it does |
|--------|----------------|
| `make install` | Create `.venv/` and `pip install -r requirements.txt` |
| `make build` | Same as `make install` (reproducible environment) |
| `make test` | Install if needed, then run `pytest tests/ -q` |
| `make run` | Full pipeline: load data → EDA → preprocess → train 3 models → comparison plots & `model_summary.csv` |
| `make run-fast` | Same as `make run` but skips EDA (`python main.py --skip-eda`) |
| `make run-eda-only` | EDA only, then exit |
| `make clean` | Remove `.venv/` |
| `make` or `make help` | Show the table above |

**Reproduction:**

```bash
make install   
make test       
make run       
```

Generated artifacts are saved under **`outputs/`** (this folder is **tracked in git** so the main figures and metrics are visible on GitHub without re-running the pipeline). Re-run `make run` or `make run-fast` to refresh them after code or data changes.

**CLI without Make**:

```bash
python3 -m venv .venv
source .venv/bin/activate   
pip install -r requirements.txt
pytest tests/ -q
python main.py             
```

---

## 2. Repository layout

| Path | Role |
|------|------|
| `main.py` | Entry point: orchestrates load → EDA → preprocess → models → comparison |
| `config.py` | Paths, random seed, column lists, model hyperparameters |
| `data_loader.py` | Load CSV, drop rows with missing values |
| `preprocessing.py` | Encoding, scaling, SMOTE, train/valid/test arrays |
| `eda.py` | Descriptive stats, chi-square, countplots, heatmap, pairplots |
| `evaluation.py` | Metrics, confusion matrix & ROC per model, joint ROC, summary CSV |
| `models/` | `logistic_regression.py`, `decision_tree.py`, `random_forest.py` |
| `tests/test_pipeline.py` | Unit tests for loader, preprocessing, and two sklearn models |
| `datasets/` | Expected location of `dataset_main.csv` |
| `outputs/` | EDA plots, confusion matrices, ROC curves, `model_summary.csv` (versioned for the report) |
| `Makefile` | **Install, test, run** |
| `.github/workflows/ci.yml` | CI: install deps + pytest on push/PR |

---

## 3. Data collection

### Source and justification

- **Primary source:** [Car Insurance Claim Prediction (Kaggle)](https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification/data).  
- **About this dataset:** It combines vehicle features, policy metadata, and a **binary claim outcome** suitable for underwriting-style risk screening. Claims are **imbalanced** (majority non-claims), which matches real portfolios.


The pipeline reads **`config.DATA_PATH`** → `datasets/dataset_main.csv`. That file is the **merged, tabular** policy-level table (one row per policy, mix of numeric and categorical fields, plus `is_claim`).

If you start from the original Kaggle bundle (e.g. separate vehicle / policy / claim tables), merge and align on `policy_id`, then export a single CSV with the same column names as the Kaggle schema used by this project, and save it as `dataset_main.csv`.


---

## 4. Data cleaning

### Steps

1. **Missing values:** Any row with **any** null is dropped in `data_loader.py` (`dropna(inplace=True)`). This is simple and transparent; the trade-off is losing partial information that could be imputed.
2. **Identifiers:** `policy_id` is dropped before modeling (not predictive).
3. **Inconsistent encodings:** `Yes`/`No` strings are mapped to **1/0** in preprocessing.
4. After one-hot encoding configured columns, **remaining object-dtype columns** are dropped so all inputs to the models are numeric.



---

## 5. Feature extraction and engineering

### Feature definitions

- **Target:** `is_claim` ∈ {0, 1}.
- **Numeric (examples):** `displacement`, `cylinder`, `airbags`, `make`, `policy_tenure`, `age_of_car`, `age_of_policyholder`, `population_density`, `ncap_rating`, etc. (see loader printout / notebook for full list).
- **Binary (post-mapping):** equipment flags originally `Yes`/`No`.
- **Categorical → one-hot:** columns listed in `config.OHE_COLUMNS` (`max_torque`, `area_cluster`, `engine_type`, `max_power`, `rear_brakes_type`, `model`, `segment`, `fuel_type`).
- **Scaling:** `MinMaxScaler` on `config.COLS_TO_SCALE` (subset of numerics present after encoding): `displacement`, `cylinder`, `airbags`, `make`.


- One-hot encoding avoids false ordinality on nominal fields.
- Scaling puts selected continuous features on a comparable scale for **logistic regression** (tree models are less sensitive but still receive the same matrix).
- Dropping residual object columns yields a **fully numeric design matrix** for sklearn.

---




## 6. Model training and evaluation

### Procedure 

1. **Load & clean** → `data_loader.load_data`.
2. **EDA** → `eda.run_eda` writes plots under `outputs/eda/`.
3. **Encode & scale** → `preprocessing.encode_and_scale`.
4. **Split & balance:**
   - `train_test_split(..., stratify=y)` on raw encoded data (80/20).
   - **SMOTE** on the training portion to oversample the minority class.

5. **Models trained** on `X_train`, `y_train`:
   - **Logistic Regression** — linear baseline, `max_iter=1000`.
   - **Decision Tree** — `max_depth`, `gini` (see `config.py`).
   - **Random Forest** — bagged trees, hyperparameters in `config.py`.
6. **Evaluation** — for each model: accuracy, sensitivity/recall on positive class, specificity, ROC-AUC; confusion matrix and ROC PNGs; then **`roc_all_models.png`** (overlay) and **`model_comparison.png`** (train vs reported test accuracy bars) + **`model_summary.csv`**.

### Model choice

- **Logistic regression:** Interpretable linear probabilities; strong baseline.
- **Decision tree / random forest:** Natural fit for **mixed numeric + binary + sparse OHE** tabular data and RF reduces variance vs a single tree. It is helpful since the features are not linearly correlated.

All the models are hyperparameter tuned, the best parameters are used here.

### Evaluation strategy

- **Metrics:** Accuracy, per-class rates from confusion matrix, ROC-AUC.

### Limitations and failure modes


- **Logistic regression** is relatively weak here as a strictly linear model for this dataset.
- **Dropped residual object columns / missing values:** If important signal lived only in removed columns or dropped rows, models cannot use it.
- **SMOTE assumptions:** Synthetic minority examples are built from feature space neighborhoods. They are not real policies. They can sharpen boundaries in ways that do not match future real claim rates, even though they help class imbalance.

---

## 7. Data visualizations and results

### Key figures (in repository)

These images are stored under `outputs/` and render directly on GitHub when you view this README.

**Target class imbalance (`is_claim`)** — counts and share of all policies after cleaning:

![Distribution of target class is_claim](outputs/eda/target_distribution_is_claim.png)

**Numeric feature correlations** (Pearson; includes `is_claim`):

![Correlation heatmap](outputs/eda/correlation_heatmap.png)

**Model discrimination** — ROC curves for all trained classifiers on the same evaluation slice (legend shows AUC):

![ROC curves for all models](outputs/models/roc_all_models.png)

### Where other outputs are saved

After `make run` (or `python main.py`):

| Location | Contents |
|----------|-----------|
| `outputs/eda/target_distribution_is_claim.png` | **Target class distribution:** bar chart of `is_claim` (0 vs 1) with counts and **% of all policies** (imbalance check) |
| `outputs/eda/countplot_*.png` | Count plots of categorical features **hue = `is_claim`** |
| `outputs/eda/correlation_heatmap.png` | Pearson correlations among numerics |
| `outputs/eda/pairplot_policy.png`, `pairplot_vehicle.png` | Pairwise numeric views (hue claim when applicable) |
| `outputs/models/cm_*.png` | Confusion matrices (one per model) |
| `outputs/models/roc_*.png` | ROC curves (one per model) |
| `outputs/models/roc_all_models.png` | **All models on one ROC chart** (legend with AUC) |
| `outputs/models/model_comparison.png` | Train vs test accuracy by model |
| `outputs/models/model_summary.csv` | Scalar metrics table |

### Insights from EDA

- **Class imbalance:** On the cleaned dataset used for the latest EDA run, **`is_claim`** is 0 for 54,701 policies (**93.6%**) and 1 for 3,741 (**6.4%**). The figure `outputs/eda/target_distribution_is_claim.png` matches these counts. Heavy skew motivates SMOTE (or similar) in the modeling pipeline.
- The correlation heatmap shows that individual numeric features are **not strongly linearly** correlated with `is_claim`, which supports using **nonlinear** models (e.g. tree-based) in addition to logistic regression.



### Final results



| Model | Test acc % | Sensitivity | Specificity | ROC AUC | Train acc % |
|-------|------------|-------------|-------------|---------|-------------|
| Logistic Regression | ~58.0 | ~0.60 | ~0.56 | ~0.62 | ~58.1 |
| Decision Tree | ~75.4 | ~0.84 | ~0.67 | ~0.84 | ~75.9 |
| Random Forest | ~73.2 | ~0.80 | ~0.66 | ~0.81 | ~75.2 |


---

## 8. Tests 

- **Data loader:** null rows removed; returns a non-empty `DataFrame`.
- **Preprocessing:** `policy_id` removed after encode; no object dtypes left; `is_claim` preserved; scaled columns in [0, 1]; split shapes consistent; training labels contain both classes after SMOTE path.
- **Models:** Logistic regression and decision tree **predict shapes**, **binary predictions**, and **probability scores in [0, 1]**.

---

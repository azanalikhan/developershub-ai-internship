# Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

## Objective
Build a reusable, production-ready machine learning pipeline that predicts
customer churn for a telecom company using the **Telco Customer Churn**
dataset.

---

## Files in this folder — what each one does, and the order to use them

| Order | File | What it does | When to run it |
|---|---|---|---|
| 1 | `requirements.txt` | Lists the packages needed (pandas, scikit-learn, joblib, matplotlib) | Once, before anything else |
| 2 | `data/Telco-Customer-Churn.csv` | The real Telco churn dataset — already included, nothing to download | Used automatically by `train_pipeline.py` |
| 3 | `train_pipeline.py` | Builds the preprocessing + model `Pipeline`, tunes Logistic Regression & Random Forest with `GridSearchCV`, evaluates both, and **exports the winning pipeline** to `models/churn_pipeline.joblib` | Run this first — nothing else works until this has run |
| 4 | `predict.py` | Loads the exported pipeline and scores new, raw (unprocessed) customer records | Run any time after training — this is the "production usage" demo |

**In short: `train_pipeline.py` -> `predict.py`.**

---

## Step-by-step: how to actually run it

### 0. Install dependencies (once)
```powershell
python -m pip install -r requirements.txt
```

### 1. Train, tune, and export the pipeline
```powershell
python train_pipeline.py
```
Takes a few minutes — Random Forest's grid search (36 candidate combos x 5
folds = 180 fits) is the slow part. It prints progress for each model, then a
side-by-side comparison, then saves:
- `models/churn_pipeline.joblib` — the complete, self-contained trained pipeline
- `models/metrics.json` — all metrics for both candidate models
- `models/evaluation_plots.png` — confusion matrix + ROC curve for the winning model

### 2. Score new customers with the exported pipeline
```powershell
python predict.py
```
Loads `models/churn_pipeline.joblib` and predicts on two example raw customer
rows defined in the script — no manual scaling/encoding needed, since all
preprocessing is baked into the pipeline itself. Edit `sample_new_customers()`
inside `predict.py` to try your own customer records.

### 3. View the plots
Open `models/evaluation_plots.png` directly in File Explorer (double-click
it) — `train_pipeline.py` saves it silently with `plt.savefig()`, no window
pops up during the run, so this is the normal way to view it afterward.

---

## Troubleshooting notes
- **`Pandas4Warning` about `select_dtypes(include=["object"])`** — cosmetic
  only, caused by a pandas 4.0 deprecation. Safe to ignore. If you want it
  gone, change that line in `train_pipeline.py` to
  `include=["object", "string"]`.

---

## Methodology / Approach
1. **Data loading** — `data/Telco-Customer-Churn.csv` (IBM's public Telco
   Churn dataset).
2. **Preprocessing** — handled entirely inside a scikit-learn `Pipeline` /
   `ColumnTransformer`:
   - Numeric features: median imputation + `StandardScaler`
   - Categorical features: most-frequent imputation + `OneHotEncoder`
   - `TotalCharges` is coerced to numeric (it ships as a string with blanks
     in the raw data)
3. **Modeling** — two candidate estimators compared inside the same pipeline:
   `LogisticRegression` and `RandomForestClassifier`.
4. **Hyperparameter tuning** — `GridSearchCV` (5-fold, scoring on ROC-AUC)
   tunes each model's hyperparameters.
5. **Export** — the entire winning pipeline (preprocessing + model) is
   serialized with `joblib` as a single artifact
   (`models/churn_pipeline.joblib`), so it can be loaded and used on raw,
   unprocessed input in production with a single `.predict()` call.

## Key Results / Observations
From the actual run on this machine:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.806 | 0.657 | 0.559 | 0.604 | 0.841 |
| **Random Forest (chosen)** | 0.751 | 0.520 | **0.794** | 0.629 | **0.843** |

- **Random Forest was selected** because `GridSearchCV` optimizes for
  ROC-AUC, and Random Forest edged out Logistic Regression there (0.843 vs
  0.841) — an extremely close margin.
- The winning Random Forest config used `class_weight='balanced'`
  (`max_depth=8`, `min_samples_leaf=4`, `n_estimators=400`). This is why its
  **recall (0.794) is much higher than Logistic Regression's (0.559)** but
  its raw accuracy is actually *lower* (0.751 vs 0.806) — the model
  deliberately trades some false alarms for catching far more real churners.
  For a churn use case, this is usually the right trade-off: missing an
  actual churner (false negative) tends to be costlier to the business than
  a false alarm.
- On the confusion matrix, this shows up as a small bottom-left cell (few
  missed churners) but a larger top-right cell (customers flagged as
  at-risk who were actually going to stay) — consistent with 52% precision.
- Because all preprocessing lives inside the `Pipeline`, the exported
  `.joblib` file is fully self-contained — `predict.py` proves this by
  scoring raw, un-preprocessed customer rows directly.

## Files
- `train_pipeline.py` — builds, tunes, evaluates, and exports the pipeline
- `predict.py` — loads the exported pipeline and scores new raw records
- `data/Telco-Customer-Churn.csv` — dataset
- `models/` — saved pipeline, metrics, and plots (generated after running)

"""
Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API
================================================================
Trains a production-ready churn prediction pipeline on the Telco Customer
Churn dataset, tunes hyperparameters with GridSearchCV, evaluates the
result, and exports the whole pipeline (preprocessing + model) with joblib.

Usage:
    python train_pipeline.py
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "Telco-Customer-Churn.csv"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_COL = "Churn"
ID_COL = "customerID"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # TotalCharges is shipped as an object dtype with some blank strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=[ID_COL])
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def get_candidate_grids(preprocessor: ColumnTransformer):
    """Returns a dict of {name: (pipeline, param_grid)} for each candidate model."""
    candidates = {}

    lr_pipe = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=2000))]
    )
    lr_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    }
    candidates["LogisticRegression"] = (lr_pipe, lr_grid)

    rf_pipe = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42))]
    )
    rf_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 8, 16],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__class_weight": [None, "balanced"],
    }
    candidates["RandomForest"] = (rf_pipe, rf_grid)

    return candidates


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    return metrics, preds, probs


def main():
    print("Loading data...")
    df = load_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(X)
    candidates = get_candidate_grids(preprocessor)

    all_metrics = []
    best_overall = {"score": -1, "name": None, "estimator": None}

    for name, (pipe, grid) in candidates.items():
        print(f"\n=== Tuning {name} with GridSearchCV ===")
        start = time.time()
        search = GridSearchCV(
            pipe, param_grid=grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1
        )
        search.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"Best CV ROC-AUC for {name}: {search.best_score_:.4f} (took {elapsed:.1f}s)")
        print(f"Best params: {search.best_params_}")

        metrics, preds, probs = evaluate(name, search.best_estimator_, X_test, y_test)
        metrics["best_cv_roc_auc"] = search.best_score_
        metrics["best_params"] = search.best_params_
        all_metrics.append(metrics)

        if metrics["roc_auc"] > best_overall["score"]:
            best_overall = {
                "score": metrics["roc_auc"],
                "name": name,
                "estimator": search.best_estimator_,
                "preds": preds,
                "probs": probs,
            }

    print("\n=== Test-set results for all candidates ===")
    for m in all_metrics:
        print(
            f"{m['model']:>16s} | acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
            f"rec={m['recall']:.4f}  f1={m['f1']:.4f}  roc_auc={m['roc_auc']:.4f}"
        )

    print(f"\nBest model on held-out test set: {best_overall['name']} (ROC-AUC={best_overall['score']:.4f})")

    # Save metrics
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(
            {"best_model": best_overall["name"], "all_candidates": all_metrics},
            f,
            indent=2,
            default=str,
        )

    # Save plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, best_overall["preds"], ax=axes[0], cmap="Blues")
    axes[0].set_title(f"Confusion Matrix - {best_overall['name']}")
    RocCurveDisplay.from_predictions(y_test, best_overall["probs"], ax=axes[1])
    axes[1].set_title(f"ROC Curve - {best_overall['name']}")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "evaluation_plots.png", dpi=150)
    print(f"Saved evaluation plots to {MODELS_DIR / 'evaluation_plots.png'}")

    # Export the full pipeline (preprocessing + model) as one artifact
    export_path = MODELS_DIR / "churn_pipeline.joblib"
    joblib.dump(best_overall["estimator"], export_path)
    print(f"Exported best pipeline ({best_overall['name']}) to {export_path}")


if __name__ == "__main__":
    main()

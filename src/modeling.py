"""
Predictive modelling: Logistic Regression vs Random Forest.

Both models predict the binary ``late_delivery`` label from the encoded
feature matrix produced by ``src.preprocess``.  We run a stratified 80/20
train/test split, report accuracy / precision / recall / F1 / confusion
matrix for both models side by side, and extract the Random Forest's feature
importance ranking.

Artifacts written to ``outputs/``:
- ``model.joblib``            – fitted {logistic, random_forest} estimators
- ``metrics.json``            – all evaluation numbers + bottlenecks
- ``feature_importance.csv``  – full RF importance ranking
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)

# Models compared in this project.
MODEL_NAMES = ["Logistic Regression", "Random Forest"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_modeling_table(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the encoded modelling table produced by the preprocessing step."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at '{path}'. "
            "Run `python src/preprocess.py` first."
        )
    return pd.read_csv(path)


def build_models() -> dict:
    """Return fresh (unfitted) estimators, seeded for reproducibility."""
    logistic = Pipeline([
        # Logistic Regression is scale-sensitive -> standardise the features.
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    random_forest = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,              # use all CPU cores
        random_state=RANDOM_STATE,
    )
    return {"Logistic Regression": logistic, "Random Forest": random_forest}


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy / precision / recall / F1 + confusion matrix for one model."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_modeling(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    """Train both models, evaluate them and persist the artifacts."""
    if df is None:
        df = load_modeling_table()

    feature_columns = [c for c in df.columns if c != TARGET_COL]
    X, y = df[feature_columns], df[TARGET_COL]

    # Stratified 80/20 split keeps the 57/43 class balance in both halves.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[model] Train: {len(X_train):,} rows | Test: {len(X_test):,} rows "
          f"| features: {X.shape[1]}")

    models = build_models()
    results: dict[str, dict] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = compute_metrics(y_test, y_pred)
        results[name]["model"] = model
        m = results[name]
        print(f"[model] {name:22s} acc={m['accuracy']:.4f} "
              f"prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}")

    # --- side-by-side comparison table -------------------------------------
    comparison = pd.DataFrame({
        name: {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        for name, metrics in results.items()
    }).T[["accuracy", "precision", "recall", "f1"]]
    print("\n" + "=" * 62)
    print("MODEL COMPARISON (hold-out test set, 20% of data)")
    print("=" * 62)
    print(comparison.to_string(float_format=lambda v: f"{v:.4f}"))
    print("=" * 62)
    for name in MODEL_NAMES:
        cm = results[name]["confusion_matrix"]
        print(f"{name} confusion matrix: "
              f"TN={cm['true_negatives']:,} FP={cm['false_positives']:,} "
              f"FN={cm['false_negatives']:,} TP={cm['true_positives']:,}")

    # --- Random Forest feature importance ----------------------------------
    rf = results["Random Forest"]["model"]
    importance = (
        pd.DataFrame({"feature": feature_columns,
                      "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "-" * 62)
    print("TOP 5 FEATURE IMPORTANCES (Random Forest) — drivers of late delivery")
    print("-" * 62)
    for i, row in importance.head(5).iterrows():
        print(f"  {i + 1}. {row['feature']:<45s} {row['importance']:.4f}")
    print("-" * 62)

    # --- persist artifacts ---------------------------------------------------
    # Exclude the fitted estimator itself — only serializable metrics go to JSON.
    payload = {
        name: {k: v for k, v in metrics.items() if k != "model"}
        for name, metrics in results.items()
    }
    metrics_out = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "models": payload,
        "top5_feature_importance": [
            {"feature": r["feature"], "importance": round(float(r["importance"]), 4)}
            for r in importance.head(5).to_dict("records")
        ],
    }

    if save:
        OUTPUT_DIR = MODEL_PATH.parent
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({n: results[n]["model"] for n in MODEL_NAMES}, MODEL_PATH)
        importance.to_csv(IMPORTANCE_PATH, index=False)
        METRICS_PATH.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
        print(f"[save] Model        -> {MODEL_PATH}")
        print(f"[save] Importance   -> {IMPORTANCE_PATH}")
        print(f"[save] Metrics      -> {METRICS_PATH}")

    return {"metrics": metrics_out, "comparison": comparison,
            "importance": importance, "results": results}


if __name__ == "__main__":
    run_modeling()

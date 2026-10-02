"""
Data cleaning, target creation and feature encoding.

Pipeline
--------
1. ``load_raw``             – read the 180k-row DataCo CSV
2. ``create_target``        – build binary ``late_delivery`` label
3. ``drop_irrelevant``      – remove IDs, PII/free text and leaky columns
4. ``add_date_features``    – turn the raw order timestamp into usable features
5. ``handle_missing``       – median-impute numerics, mode-impute categoricals
6. ``encode_features``      – one-hot encode the categorical features

The final DataFrame (numeric matrix + target) is written to
``data/processed/modeling_data.csv`` together with a small JSON metadata file
that the dashboard uses to rebuild feature vectors for the risk lookup.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    META_PATH,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    REAL_DAYS_COL,
    REQUIRED_COLUMNS,
    SCHEDULED_DAYS_COL,
    TARGET_COL,
)
from src.data_loader import load_raw  # noqa: E402


# --------------------------------------------------------------------------- #
# Step 1 — target
# --------------------------------------------------------------------------- #
def create_target(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create the binary ``late_delivery`` target.

    late = actual shipping days strictly greater than scheduled shipping days.
    The dataset ships its own ``Late_delivery_risk`` flag; we keep our derived
    label (as required) and report how well the two agree.
    """
    missing = [c for c in REQUIRED_COLUMNS[:-1] if c not in df.columns]
    if missing:
        raise KeyError(f"Raw dataset is missing expected columns: {missing}")

    df = df.copy()
    df[TARGET_COL] = (df[REAL_DAYS_COL] > df[SCHEDULED_DAYS_COL]).astype("int8")

    stats = {
        "rows": int(len(df)),
        "late_rate": round(float(df[TARGET_COL].mean()), 4),
    }
    if "Late_delivery_risk" in df.columns:
        agreement = float((df[TARGET_COL] == df["Late_delivery_risk"]).mean())
        stats["agreement_with_dataset_flag"] = round(agreement, 4)

    print(
        f"[target] late_delivery rate = {stats['late_rate']:.1%} "
        f"({stats['late_rate'] * stats['rows']:,.0f} late orders)"
    )
    if "agreement_with_dataset_flag" in stats:
        print(
            "[target] agreement with the dataset's own Late_delivery_risk "
            f"flag = {stats['agreement_with_dataset_flag']:.1%}"
        )
    return df, stats


# --------------------------------------------------------------------------- #
# Step 2 — drop irrelevant / high-cardinality / leaky columns
# --------------------------------------------------------------------------- #
def drop_irrelevant(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop identifiers, PII, free text, leaky and redundant columns."""
    present = [c for c in DROP_COLUMNS if c in df.columns]
    # Never drop the target or its inputs.
    present = [c for c in present if c not in (TARGET_COL, SCHEDULED_DAYS_COL)]

    df = df.drop(columns=present)

    # Safety net: remove any constant (zero-variance) leftovers except target.
    constants = [
        c for c in df.columns
        if c != TARGET_COL and df[c].nunique(dropna=False) <= 1
    ]
    df = df.drop(columns=constants)

    dropped = present + constants
    print(f"[clean] Dropped {len(dropped)} irrelevant/high-cardinality columns")
    return df, dropped


# --------------------------------------------------------------------------- #
# Step 3 — date feature engineering
# --------------------------------------------------------------------------- #
def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar features from the raw order timestamp."""
    df = df.copy()
    order_dt = pd.to_datetime(
        df.pop("order date (DateOrders)"), format="%m/%d/%Y %H:%M",
        errors="coerce",
    )
    n_bad = int(order_dt.isna().sum())
    if n_bad:
        print(f"[dates] {n_bad} unparseable timestamps set to NaT")

    df["order_month"] = order_dt.dt.month.fillna(0).astype("int8")
    df["order_year"] = order_dt.dt.year.fillna(0).astype("int16")
    # Monday=0 ... Sunday=6
    df["order_day_of_week"] = order_dt.dt.dayofweek.fillna(0).astype("int8")
    df["order_is_weekend"] = (df["order_day_of_week"] >= 5).astype("int8")
    df["order_hour"] = order_dt.dt.hour.fillna(0).astype("int8")

    print("[dates] Engineered order_month / order_year / order_day_of_week "
          "/ order_is_weekend / order_hour")
    return df


# --------------------------------------------------------------------------- #
# Step 4 — missing values
# --------------------------------------------------------------------------- #
def handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Median-impute numeric columns, mode-impute categorical columns."""
    df = df.copy()
    report: dict[str, int] = {}

    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        report[col] = n_missing

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "Unknown")

    total = sum(report.values())
    if report:
        print(f"[missing] Imputed {total} values across {len(report)} columns: "
              f"{report}")
    else:
        print("[missing] No missing values found")
    return df, report


# --------------------------------------------------------------------------- #
# Step 5 — encoding
# --------------------------------------------------------------------------- #
def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categorical features; leave numerics untouched."""
    missing = [c for c in CATEGORICAL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot encode, missing categorical columns: {missing}")

    # pd.get_dummies creates one binary column per category level.
    encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dtype=np.int8)

    feature_columns = [c for c in encoded.columns if c != TARGET_COL]
    print(f"[encode] One-hot encoded {len(CATEGORICAL_FEATURES)} categorical "
          f"columns -> {len(feature_columns)} total features")
    return encoded, feature_columns


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_cleaned() -> tuple[pd.DataFrame, dict]:
    """Load raw data and run every cleaning step *except* encoding.

    Returns the cleaned frame with the original (human-readable) categorical
    columns intact — this is what the EDA/bottleneck analysis operates on.
    """
    df = load_raw()

    df, target_stats = create_target(df)
    df, dropped = drop_irrelevant(df)
    df = add_date_features(df)
    df, missing_report = handle_missing(df)

    # Sanity: no NaNs may remain before encoding.
    assert df.isna().sum().sum() == 0, "NaNs remain after imputation"

    info = {
        "dropped_columns": dropped,
        "missing_imputed": missing_report,
        "target_stats": target_stats,
    }
    return df, info


def encode_and_persist(cleaned: pd.DataFrame, info: dict, save: bool = True) -> tuple[pd.DataFrame, dict]:
    """One-hot encode the cleaned frame, persist table + metadata."""
    model_df, feature_columns = encode_features(cleaned)

    # Defaults used by the dashboard to fill unspecified lookup inputs.
    defaults: dict[str, object] = {}
    for col in (NUMERIC_FEATURES
                + ["order_month", "order_year", "order_day_of_week",
                   "order_is_weekend", "order_hour"]):
        if col in cleaned.columns:
            defaults[col] = float(cleaned[col].median())
    for col in CATEGORICAL_FEATURES:
        defaults[col] = str(cleaned[col].mode().iloc[0])

    meta = {
        "feature_columns": feature_columns,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": [c for c in NUMERIC_FEATURES if c in cleaned.columns],
        "defaults": defaults,
        "dropped_columns": info["dropped_columns"],
        "missing_imputed": info["missing_imputed"],
        "target": TARGET_COL,
        "target_stats": info["target_stats"],
        "random_state": RANDOM_STATE,
    }

    if save:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        model_df.to_csv(PROCESSED_DATA_PATH, index=False)
        META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[save] Modelling table -> {PROCESSED_DATA_PATH} "
              f"({model_df.shape[0]:,} rows x {model_df.shape[1]} cols)")
        print(f"[save] Metadata       -> {META_PATH}")

    return model_df, meta


def run_preprocessing(save: bool = True) -> tuple[pd.DataFrame, dict]:
    """Run the full cleaning pipeline and persist the modelling table."""
    cleaned, info = build_cleaned()
    return encode_and_persist(cleaned, info, save=save)


if __name__ == "__main__":
    run_preprocessing()

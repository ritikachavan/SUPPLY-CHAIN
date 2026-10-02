"""
One-command pipeline: preprocess -> EDA/bottlenecks -> train & evaluate.

Usage:
    python run_pipeline.py

Produces:
    data/processed/modeling_data.csv
    outputs/preprocess_meta.json
    outputs/figures/*.png
    outputs/bottlenecks.csv
    outputs/model.joblib
    outputs/metrics.json
    outputs/feature_importance.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eda import run_eda  # noqa: E402
from src.modeling import run_modeling  # noqa: E402
from src.preprocess import build_cleaned, encode_and_persist  # noqa: E402


def main() -> None:
    print("=" * 72)
    print("SUPPLY CHAIN DELAY RISK PREDICTOR — full pipeline")
    print("=" * 72)

    # 1. Clean + target + encode -------------------------------------------
    cleaned, info = build_cleaned()
    model_df, meta = encode_and_persist(cleaned, info, save=True)

    # 2. Bottleneck analysis + charts --------------------------------------
    bottlenecks = run_eda(cleaned)

    # 3. Train / evaluate both models --------------------------------------
    out = run_modeling(model_df, save=True)

    # 4. Final summary ------------------------------------------------------
    print("\n" + "#" * 72)
    print("FINAL SUMMARY")
    print("#" * 72)
    print(f"Orders analysed      : {info['target_stats']['rows']:,}")
    print(f"Overall late rate    : {info['target_stats']['late_rate']:.1%}")
    print("\nModel comparison:")
    print(out["comparison"].to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nTop 3 bottleneck segments:")
    for i, row in bottlenecks.iterrows():
        print(f"  {i + 1}. {row['dimension']}: {row['segment']} "
              f"({row['late_rate']:.1f}% late, {row['orders']:,} orders)")
    print("\nTop 5 feature importances (Random Forest):")
    for i, row in enumerate(out["metrics"]["top5_feature_importance"], start=1):
        print(f"  {i}. {row['feature']:<45s} {row['importance']:.4f}")
    print("\nNext step: launch the dashboard with "
          "`streamlit run dashboard.py`")
    print("#" * 72)


if __name__ == "__main__":
    main()
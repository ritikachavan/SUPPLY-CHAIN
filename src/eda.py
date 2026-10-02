"""
Exploratory / bottleneck analysis.

Computes the late-delivery rate for every shipping mode, market, region and
product category, prints the three worst ("bottleneck") segments, and saves
three bar charts as PNG files under ``outputs/figures/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend so charts render without a display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    BOTTLENECK_PATH,
    CATEGORY_COL,
    FIGURES_DIR,
    MARKET_COL,
    MIN_ORDERS_FOR_SEGMENT,
    REGION_COL,
    SHIPPING_MODE_COL,
    TARGET_COL,
)

# Dimensions analysed for bottlenecks (label used in printed output).
ANALYSIS_DIMENSIONS = {
    SHIPPING_MODE_COL: "Shipping mode",
    MARKET_COL: "Market",
    REGION_COL: "Region",
    CATEGORY_COL: "Product category",
}


# --------------------------------------------------------------------------- #
# Grouped late-delivery rates
# --------------------------------------------------------------------------- #
def late_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return late-delivery rate and order count for each level of *column*."""
    grouped = (
        df.groupby(column)[TARGET_COL]
        .agg(late_rate="mean", orders="count")
        .sort_values("late_rate", ascending=False)
    )
    grouped["late_rate"] = (grouped["late_rate"] * 100).round(1)
    return grouped.reset_index()


def top_bottlenecks(
    df: pd.DataFrame, top_n: int = 3, min_orders: int = MIN_ORDERS_FOR_SEGMENT
) -> pd.DataFrame:
    """Worst *top_n* segments across all analysed dimensions.

    Segments with fewer than *min_orders* orders are ignored so that a handful
    of orders cannot dominate the ranking.
    """
    frames = []
    for column, label in ANALYSIS_DIMENSIONS.items():
        rates = late_rate_by(df, column)
        rates = rates[rates["orders"] >= min_orders].copy()
        rates.insert(0, "dimension", label)
        rates = rates.rename(columns={column: "segment"})
        frames.append(rates)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["late_rate", "orders"], ascending=[False, False]
    ).head(top_n)
    return combined.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _bar_chart(frame: pd.DataFrame, title: str, filename: str, top: int | None = None) -> Path:
    """Save a horizontal bar chart of late-delivery rates as a PNG."""
    data = frame.head(top) if top else frame
    data = data.sort_values("late_rate")  # ascending -> biggest problem on top

    label_col = data.columns[0]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(data) + 1.6))

    bars = ax.barh(data[label_col].astype(str), data["late_rate"], color="#d9534f")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_xlim(0, max(100, data["late_rate"].max() * 1.15))
    ax.set_xlabel("Late-delivery rate (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_charts(df: pd.DataFrame) -> list[Path]:
    """Create and save the three bottleneck bar charts."""
    saved: list[Path] = []

    # 1. Late rate by shipping mode (all 4 levels)
    saved.append(_bar_chart(
        late_rate_by(df, SHIPPING_MODE_COL),
        "Late-delivery rate by shipping mode",
        "late_rate_by_shipping_mode.png",
    ))

    # 2. Late rate by region (top 12 of 23 regions)
    saved.append(_bar_chart(
        late_rate_by(df, REGION_COL),
        "Late-delivery rate by region (top 12)",
        "late_rate_by_region.png",
        top=12,
    ))

    # 3. Late rate by product category (top 12 of 50 categories)
    saved.append(_bar_chart(
        late_rate_by(df, CATEGORY_COL),
        "Late-delivery rate by product category (top 12)",
        "late_rate_by_category.png",
        top=12,
    ))
    return saved


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Print grouped rates + top-3 bottlenecks and save the charts."""
    print("\n" + "=" * 72)
    print("BOTTLENECK ANALYSIS — late-delivery rate by segment")
    print("=" * 72)

    for column, label in ANALYSIS_DIMENSIONS.items():
        rates = late_rate_by(df, column)
        print(f"\n--- {label} ({column}) ---")
        print(rates.to_string(index=False))

    bottlenecks = top_bottlenecks(df, top_n=3)

    print("\n" + "-" * 72)
    print(f"TOP 3 BOTTLENECK SEGMENTS (min {MIN_ORDERS_FOR_SEGMENT} orders):")
    print("-" * 72)
    for i, row in bottlenecks.iterrows():
        print(f"  {i + 1}. {row['dimension']}: {row['segment']!r} "
              f"-> {row['late_rate']:.1f}% late ({row['orders']:,} orders)")
    print("-" * 72)

    charts = make_charts(df)
    for path in charts:
        print(f"[chart] Saved {path}")

    if save:
        BOTTLENECK_PATH.parent.mkdir(parents=True, exist_ok=True)
        bottlenecks.to_csv(BOTTLENECK_PATH, index=False)

    return bottlenecks


if __name__ == "__main__":
    from src.preprocess import build_cleaned

    cleaned_df, _ = build_cleaned()
    run_eda(cleaned_df)

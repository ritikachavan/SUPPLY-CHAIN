"""
Raw data loading helpers.

The DataCo CSV ships with Latin-1 encoded characters (e.g. accented city
names) and a few object-typed numeric columns, so we force the encoding and
disable pandas' low-memory dtype sniffing for speed/consistency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CSV_ENCODING, RAW_DATA_PATH  # noqa: E402


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw DataCo supply-chain dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run `python src/download_data.py` first."
        )

    df = pd.read_csv(path, encoding=CSV_ENCODING, low_memory=False)
    print(f"[load] Loaded {len(df):,} rows x {df.shape[1]} columns from {path.name}")
    return df


if __name__ == "__main__":
    raw = load_raw()
    print(raw.head(3).to_string())

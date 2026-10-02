"""
Download the DataCo Smart Supply Chain dataset.

Kaggle's official download requires an API token (``kaggle.json``), so by
default we fetch the byte-identical public mirror of the same CSV.  If you
prefer Kaggle:

    pip install kaggle
    kaggle datasets download -d shashankthecookie/dataco-smart-supply-chain-for-big-data-analysis

Usage:
    python src/download_data.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Allow running as a script: `python src/download_data.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MIRROR_URL, RAW_DATA_PATH  # noqa: E402


def download(destination: Path = RAW_DATA_PATH, url: str = MIRROR_URL) -> Path:
    """Download the dataset to *destination* unless it already exists."""
    if destination.exists() and destination.stat().st_size > 1_000_000:
        print(f"[download] Dataset already present: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching {url}")
    print(f"[download] Saving to {destination} ...")

    # `urlretrieve` streams the file to disk in one pass.
    urllib.request.urlretrieve(url, destination)

    size_mb = destination.stat().st_size / 1_048_576
    print(f"[download] Done ({size_mb:,.1f} MB).")
    return destination


if __name__ == "__main__":
    download()

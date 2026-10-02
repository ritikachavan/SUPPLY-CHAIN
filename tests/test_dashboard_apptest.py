"""
Headless execution test for dashboard.py using Streamlit's AppTest.

Runs the whole Streamlit script in-process (no browser/server needed) and
asserts that it executes without exceptions and renders the key elements.

Usage:
    python tests/test_dashboard_apptest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


def main() -> int:
    print("[apptest] Running dashboard.py headlessly ...")
    at = AppTest.from_file(str(PROJECT_ROOT / "dashboard.py"), default_timeout=180)
    at.run()

    # 1. no unhandled exception --------------------------------------------
    if at.exception:
        print(f"[apptest] FAIL: exception raised:\n{at.exception}")
        return 1
    print("[apptest] OK: script ran without exceptions")

    # 2. KPI metrics present ------------------------------------------------
    metrics = [m.label for m in at.metric]
    expected = {"Overall late-delivery rate", "Orders analysed",
                "Random Forest accuracy", "Random Forest F1"}
    missing = expected - set(metrics)
    if missing:
        print(f"[apptest] FAIL: missing KPIs: {missing}")
        return 1
    print(f"[apptest] OK: {len(metrics)} KPI metrics rendered")

    # 3. risk lookup widgets present ---------------------------------------
    selects = [s.label for s in at.selectbox]
    if not {"Shipping mode", "Region", "Product category"} <= set(selects):
        print(f"[apptest] FAIL: lookup widgets missing: {selects}")
        return 1
    print(f"[apptest] OK: risk lookup widgets present ({selects})")

    # 4. exercise the risk lookup end-to-end -------------------------------
    at.selectbox[0].select("Standard Class").run()
    at.selectbox[1].select("Canada").run()
    at.selectbox[2].select("Computers").run()
    if at.exception:
        print(f"[apptest] FAIL: exception after widget change:\n{at.exception}")
        return 1

    labels = [m.label for m in at.metric]
    prob_idx = labels.index("Predicted late-delivery probability")
    print(f"[apptest] OK: prediction = {at.metric[prob_idx].value}")

    print("[apptest] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

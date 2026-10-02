"""
Dashboard smoke test.

Starts the Streamlit app as a subprocess, waits for its health endpoint,
fetches the main page, then shuts it down.  Exits 0 on success.

Usage:
    python tests/test_dashboard.py
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORT = 8501
BASE = f"http://127.0.0.1:{PORT}"
STARTUP_TIMEOUT_S = 90


def probe(url: str, timeout: float = 3.0) -> int | None:
    """Return the HTTP status for *url*, or None if unreachable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except (urllib.error.URLError, OSError):
        return None


def main() -> int:
    cmd = [
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.headless", "true",
        "--server.address", "127.0.0.1",
        "--server.port", str(PORT),
        "--browser.gatherUsageStats", "false",
    ]
    print(f"[smoke] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # 1. wait for the health endpoint -----------------------------------
        deadline = time.time() + STARTUP_TIMEOUT_S
        status = None
        while time.time() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=5)
                print(f"[smoke] Process exited early with code {proc.returncode}")
                print(out)
                print(err)
                return 1
            status = probe(f"{BASE}/_stcore/health")
            if status == 200:
                break
            time.sleep(2)

        if status != 200:
            print(f"[smoke] FAIL: health endpoint never returned 200 "
                  f"(last status={status})")
            return 1
        print("[smoke] OK: /_stcore/health -> 200")

        # 2. fetch the main page --------------------------------------------
        page_status = probe(BASE)
        if page_status != 200:
            print(f"[smoke] FAIL: main page returned {page_status}")
            return 1

        with urllib.request.urlopen(BASE, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        has_title = "Supply Chain Delay Risk Predictor" in html
        print(f"[smoke] OK: / -> 200 (app shell served: {has_title})")
        if not has_title:
            # The title lives in the React shell only after the WS session
            # starts, so treat its absence as informational, not fatal.
            print("[smoke] note: app title not in static shell (expected)")
        return 0
    finally:
        # 3. shut down -------------------------------------------------------
        proc.terminate()
        try:
            out, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate(timeout=10)
        print("[smoke] server stdout:")
        print(out or "(empty)")
        print("[smoke] server stderr:")
        print(err or "(empty)")


if __name__ == "__main__":
    raise SystemExit(main())

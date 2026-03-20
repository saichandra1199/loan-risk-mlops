#!/usr/bin/env python3
"""Download the UCI Default of Credit Card Clients dataset.

Dataset: Default of Credit Card Clients
Source:  UCI ML Repository (OpenML dataset ID 350)
URL:     https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Size:    30,000 rows × 24 columns
License: Creative Commons Attribution 4.0 International (CC BY 4.0)

This dataset contains information about credit card clients in Taiwan from
April 2005 to September 2005. The prediction task is to determine whether
a client will default on their next payment.

Usage:
    uv run python scripts/download_dataset.py
    uv run python scripts/download_dataset.py --output data/raw/credit_default_raw.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_via_openml(output_path: Path) -> None:
    """Download using scikit-learn's fetch_openml (no auth required).

    OpenML hosts the UCI dataset publicly. fetch_openml caches the download
    in ~/scikit_learn_data/ so subsequent runs are instant.
    """
    print("Downloading UCI Credit Card Default dataset via OpenML...")
    print("  Source: https://www.openml.org/d/350")
    print("  This is free and requires no account or API key.")
    print()

    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        print("ERROR: scikit-learn not installed. Run: uv sync --extra dev")
        sys.exit(1)

    # data_id=350 is the stable OpenML identifier for this dataset
    # as_frame=True returns pandas DataFrames (easier for CSV export)
    dataset = fetch_openml(data_id=350, as_frame=True, parser="auto")

    df = dataset.frame  # full DataFrame including target column

    if df is None or len(df) == 0:
        print("ERROR: Downloaded dataset is empty.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"  Downloaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Saved to: {output_path}")
    print()
    print("Column summary:")
    print(f"  Features: {list(df.columns[:-1])}")
    print(f"  Target:   {df.columns[-1]}")
    print()

    # Compute and display default rate
    target_col = "default-payment-next-month"
    if target_col in df.columns:
        default_rate = df[target_col].astype(int).mean()
        print(f"  Default rate: {default_rate:.1%}")


def download_via_requests(output_path: Path) -> None:
    """Fallback: download directly from UCI repository.

    Uses the direct ZIP download URL from UCI ML Repository.
    This is a fallback in case OpenML is unavailable.
    """
    import io
    import zipfile

    try:
        import requests
    except ImportError:
        print("ERROR: requests not installed. Run: uv sync")
        sys.exit(1)

    # Direct UCI download URL (no authentication required)
    UCI_URL = (
        "https://archive.ics.uci.edu/static/public/350/"
        "default+of+credit+card+clients.zip"
    )

    print("Downloading from UCI ML Repository...")
    print(f"  URL: {UCI_URL}")

    response = requests.get(UCI_URL, timeout=120)
    response.raise_for_status()

    # Extract the Excel file from the ZIP
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        names = z.namelist()
        xlsx_files = [n for n in names if n.endswith(".xls") or n.endswith(".xlsx")]
        if not xlsx_files:
            print(f"ERROR: No Excel file found in ZIP. Contents: {names}")
            sys.exit(1)

        with z.open(xlsx_files[0]) as f:
            try:
                import pandas as pd
                df = pd.read_excel(f, header=1)
            except ImportError:
                print("ERROR: pandas not installed. Run: uv sync")
                sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df):,} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download UCI Credit Card Default dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        default="data/raw/credit_default_raw.csv",
        help="Output path for the raw CSV file (default: data/raw/credit_default_raw.csv)",
    )
    parser.add_argument(
        "--method",
        choices=["openml", "uci"],
        default="openml",
        help="Download method: openml (default) or uci (direct download)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    if output_path.exists():
        print(f"File already exists: {output_path}")
        print("Delete it manually if you want to re-download.")
        return

    if args.method == "openml":
        try:
            download_via_openml(output_path)
        except Exception as exc:
            print(f"OpenML download failed: {exc}")
            print("Trying direct UCI download as fallback...")
            download_via_requests(output_path)
    else:
        download_via_requests(output_path)

    print("Download complete.")


if __name__ == "__main__":
    main()

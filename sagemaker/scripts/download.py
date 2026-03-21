"""SageMaker ProcessingStep: Download UCI Credit Card Default dataset.

Wraps scripts/download_dataset.py logic and writes output to
/opt/ml/processing/output/ (mapped to S3 by SageMaker).

Invoked by SageMaker as a ProcessingJob using SKLearnProcessor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# SageMaker mounts the project source at /opt/ml/processing/input/code/
# Add src/ to path so loan_risk imports work
sys.path.insert(0, "/opt/ml/processing/input/code/src")
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import requests  # noqa: PLC0415

    # UCI OpenML dataset: Credit Card Default (Taiwan, 2005)
    DATASET_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00350/default%20of%20credit%20card%20clients.xls"
    )
    output_path = output_dir / "credit_default_raw.xls"

    print(f"Downloading dataset from UCI OpenML to {output_path}")
    response = requests.get(DATASET_URL, timeout=120)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    print(f"Downloaded {len(response.content) / 1024:.1f} KB")

    # Verify file is non-empty
    assert output_path.stat().st_size > 10_000, "Downloaded file too small — download may have failed"
    print("Download complete.")


if __name__ == "__main__":
    main()

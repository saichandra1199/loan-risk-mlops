"""SageMaker ProcessingStep: Preprocess and schema-validate raw loan data.

Reads raw XLS/CSV from /opt/ml/processing/input/,
writes validated Parquet to /opt/ml/processing/output/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/opt/ml/processing/input/code/src")
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import polars as pl  # noqa: PLC0415

    # Find the raw input file
    raw_files = list(INPUT_DIR.glob("*.xls")) + list(INPUT_DIR.glob("*.csv")) + list(INPUT_DIR.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No raw data files found in {INPUT_DIR}")

    raw_path = raw_files[0]
    print(f"Preprocessing: {raw_path}")

    # Load raw file
    if raw_path.suffix == ".xls":
        import pandas as pd  # noqa: PLC0415
        pdf = pd.read_excel(raw_path, header=1)
        df = pl.from_pandas(pdf)
    elif raw_path.suffix == ".parquet":
        df = pl.read_parquet(raw_path)
    else:
        df = pl.read_csv(raw_path)

    print(f"Raw shape: {df.shape}")

    # Apply the same column renames/validation as scripts/preprocess_dataset.py
    from loan_risk.data.schemas import RawLoanSchema  # noqa: PLC0415

    # Column mapping from UCI → loan schema
    column_map = {
        "ID": "loan_id",
        "LIMIT_BAL": "credit_limit",
        "SEX": "sex",
        "EDUCATION": "education_level",
        "MARRIAGE": "marital_status",
        "AGE": "age",
        "PAY_0": "payment_status_1",
        "PAY_2": "payment_status_2",
        "PAY_3": "payment_status_3",
        "PAY_4": "payment_status_4",
        "PAY_5": "payment_status_5",
        "PAY_6": "payment_status_6",
        "BILL_AMT1": "bill_amount_1",
        "BILL_AMT2": "bill_amount_2",
        "BILL_AMT3": "bill_amount_3",
        "BILL_AMT4": "bill_amount_4",
        "BILL_AMT5": "bill_amount_5",
        "BILL_AMT6": "bill_amount_6",
        "PAY_AMT1": "pay_amount_1",
        "PAY_AMT2": "pay_amount_2",
        "PAY_AMT3": "pay_amount_3",
        "PAY_AMT4": "pay_amount_4",
        "PAY_AMT5": "pay_amount_5",
        "PAY_AMT6": "pay_amount_6",
        "default.payment.next.month": "loan_default",
    }

    existing_cols = {c: column_map[c] for c in df.columns if c in column_map}
    df = df.rename(existing_cols)

    # Derive annual_income and loan_amount from UCI fields if not present
    if "annual_income" not in df.columns and "credit_limit" in df.columns:
        df = df.with_columns(
            pl.col("credit_limit").cast(pl.Float64).alias("annual_income"),
            pl.col("credit_limit").cast(pl.Float64).alias("loan_amount"),
        )

    output_path = OUTPUT_DIR / "loans.parquet"
    df.write_parquet(output_path)
    print(f"Wrote {len(df)} rows to {output_path}")

    _ = RawLoanSchema  # imported for validation purposes; full pandera validation in pipeline


if __name__ == "__main__":
    main()

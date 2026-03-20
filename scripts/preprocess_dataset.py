#!/usr/bin/env python3
"""Preprocess the UCI Credit Card Default dataset into our canonical loan schema.

This script bridges the UCI raw dataset to the loan risk pipeline's expected
feature schema. It performs:
  1. Column renaming and type casting
  2. Feature engineering (derived features from UCI columns)
  3. Data cleaning (outlier clipping, missing value handling)
  4. Pandera schema validation

== Column Mapping ==

UCI Raw Column         → Our Schema Column
----------------------------------------------
ID                     → loan_id              (e.g., "LOAN_0000001")
LIMIT_BAL              → loan_amount          (credit limit, clipped to 100–100k)
LIMIT_BAL * 3          → annual_income        (proxy: 3× credit limit, clipped)
max(0, AGE - 22)       → employment_years     (assumed start working at age 22)
payment_reliability    → credit_score         (300–850 derived from PAY_0..6)
BILL_AMT1/income_mo    → debt_to_income_ratio (bill utilisation)
count(PAY_AMT_i > 0)   → num_open_accounts    (months with active payments)
count(PAY_i > 0)       → num_delinquencies    (months with payment delay)
EDUCATION              → loan_purpose         (mapped to 6 categories)
MARRIAGE               → home_ownership       (RENT/MORTGAGE/OWN)
LIMIT_BAL threshold    → loan_term_months     (36 or 60 months)
default.payment.…      → loan_default         (0=repaid, 1=defaulted)

Usage:
    uv run python scripts/preprocess_dataset.py
    uv run python scripts/preprocess_dataset.py --input data/raw/credit_default_raw.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_raw(input_path: Path):
    """Load the raw UCI CSV into a pandas DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. Run: uv sync")
        sys.exit(1)

    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Normalize column names: lowercase, replace spaces/hyphens with underscores
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    print(f"  Columns: {list(df.columns)}")
    return df


def derive_credit_score(df) -> "pd.Series":
    """Derive a FICO-style credit score (300–850) from payment history.

    Logic:
      - Each of the 6 payment status columns (pay_0, pay_2..pay_6) can be:
          -2 or -1: paid on time or no consumption (good)
          0:        revolving credit (neutral)
          1-9:      payment delay of N months (bad)
      - payment_reliability = fraction of months that were on time (PAY_X <= 0)
      - credit_score = 300 + 550 * payment_reliability → range [300, 850]
    """
    import numpy as np

    # Payment status columns (note: UCI uses pay_0 for most recent, pay_2..6 for older)
    pay_cols = [c for c in df.columns if c.startswith("pay_") and c not in
                [c2 for c2 in df.columns if c2.startswith("pay_amt")]]

    if not pay_cols:
        # Fallback if column names differ slightly
        pay_cols = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
        pay_cols = [c for c in pay_cols if c in df.columns]

    if not pay_cols:
        print("WARNING: No payment status columns found. Using default credit score 650.")
        return 650

    on_time = df[pay_cols].apply(lambda row: (row <= 0).sum(), axis=1)
    reliability = on_time / len(pay_cols)  # 0.0 to 1.0
    credit_score = (300 + 550 * reliability).clip(300, 850).astype(int)
    return credit_score


def derive_delinquencies(df) -> "pd.Series":
    """Count months with payment delay (PAY_X > 0)."""
    pay_cols = [c for c in df.columns if c.startswith("pay_") and
                not c.startswith("pay_amt")]

    if not pay_cols:
        return 0

    delayed = df[pay_cols].apply(lambda row: (row > 0).sum(), axis=1)
    return delayed.clip(0, 50).astype(int)


def derive_num_open_accounts(df) -> "pd.Series":
    """Count months where a payment was actually made (PAY_AMT > 0)."""
    pay_amt_cols = [c for c in df.columns if c.startswith("pay_amt")]

    if not pay_amt_cols:
        return 1

    active = df[pay_amt_cols].apply(lambda row: (row > 0).sum(), axis=1)
    return active.clip(1, 100).astype(int)


def map_education_to_purpose(education_series) -> "pd.Series":
    """Map UCI EDUCATION codes to loan purpose categories.

    UCI EDUCATION values:
      1 = graduate school → professionals → home improvement
      2 = university      → young adults  → debt consolidation
      3 = high school     → working class → major purchase
      4 = others
      5/6 = unknown       → other
      0 = unknown (some datasets have this)
    """
    mapping = {
        "1": "home_improvement",
        "2": "debt_consolidation",
        "3": "major_purchase",
        "4": "other",
        "5": "vacation",
        "6": "medical",
        "0": "other",
        1: "home_improvement",
        2: "debt_consolidation",
        3: "major_purchase",
        4: "other",
        5: "vacation",
        6: "medical",
        0: "other",
    }
    return education_series.map(mapping).fillna("other")


def map_marriage_to_ownership(marriage_series) -> "pd.Series":
    """Map UCI MARRIAGE codes to home ownership categories.

    UCI MARRIAGE values:
      1 = married → likely has mortgage
      2 = single  → likely renting
      3 = others  → assume own
      0 = unknown → assume renting
    """
    mapping = {
        "1": "MORTGAGE",
        "2": "RENT",
        "3": "OWN",
        "0": "RENT",
        1: "MORTGAGE",
        2: "RENT",
        3: "OWN",
        0: "RENT",
    }
    return marriage_series.map(mapping).fillna("RENT")


def preprocess(df) -> "pd.DataFrame":
    """Apply all transformations to produce the canonical loan schema DataFrame.

    Returns a DataFrame with exactly the columns expected by RawLoanSchema.
    """
    import numpy as np
    import pandas as pd

    print("\nApplying feature transformations...")
    result = pd.DataFrame()

    # --- loan_id ---
    id_col = "id" if "id" in df.columns else df.columns[0]
    result["loan_id"] = df[id_col].apply(lambda x: f"LOAN_{int(x):07d}")
    print(f"  loan_id: from '{id_col}' → 'LOAN_XXXXXXX' format")

    # --- loan_amount (LIMIT_BAL) ---
    limit_col = "limit_bal" if "limit_bal" in df.columns else "LIMIT_BAL"
    if limit_col not in df.columns:
        # Try case-insensitive search
        matches = [c for c in df.columns if "limit" in c.lower()]
        limit_col = matches[0] if matches else df.columns[1]
    limit_bal = pd.to_numeric(df[limit_col], errors="coerce").fillna(10000)
    result["loan_amount"] = limit_bal.clip(100, 100_000).round(2)
    print(f"  loan_amount: from '{limit_col}' (credit limit) → clipped to [100, 100k]")

    # --- annual_income (proxy: LIMIT_BAL × 3) ---
    annual_income = (limit_bal * 3).clip(1_000, 10_000_000).round(2)
    result["annual_income"] = annual_income
    print("  annual_income: LIMIT_BAL × 3 (proxy for earning capacity)")

    # --- employment_years (proxy: AGE - 22) ---
    age_col = "age" if "age" in df.columns else "AGE"
    if age_col not in df.columns:
        matches = [c for c in df.columns if "age" in c.lower()]
        age_col = matches[0] if matches else None
    if age_col:
        age = pd.to_numeric(df[age_col], errors="coerce").fillna(30)
        result["employment_years"] = (age - 22).clip(0, 50).astype(int)
        print(f"  employment_years: from '{age_col}' → max(0, AGE - 22)")
    else:
        result["employment_years"] = 5
        print("  employment_years: AGE not found, using default 5")

    # --- credit_score (derived from payment history) ---
    result["credit_score"] = derive_credit_score(df)
    print("  credit_score: derived from payment status columns (300-850 scale)")

    # --- debt_to_income_ratio ---
    bill_col = None
    for candidate in ["bill_amt1", "bill_amt_1", "bill_amt"]:
        if candidate in df.columns:
            bill_col = candidate
            break
    if bill_col is None:
        matches = [c for c in df.columns if "bill" in c.lower() and "amt" in c.lower()]
        bill_col = matches[0] if matches else None

    if bill_col is not None:
        bill_amt = pd.to_numeric(df[bill_col], errors="coerce").fillna(0)
        monthly_income = (annual_income / 12).replace(0, 1)
        dti = (bill_amt / monthly_income).clip(0, 2).round(4)
        result["debt_to_income_ratio"] = dti
        print(f"  debt_to_income_ratio: BILL_AMT1 / (annual_income/12)")
    else:
        result["debt_to_income_ratio"] = 0.3
        print("  debt_to_income_ratio: bill columns not found, using 0.3")

    # --- num_open_accounts ---
    result["num_open_accounts"] = derive_num_open_accounts(df)
    print("  num_open_accounts: count of months with actual payment made")

    # --- num_delinquencies ---
    result["num_delinquencies"] = derive_delinquencies(df)
    print("  num_delinquencies: count of months with payment delay > 0")

    # --- loan_purpose (from EDUCATION) ---
    edu_col = "education" if "education" in df.columns else "EDUCATION"
    if edu_col not in df.columns:
        matches = [c for c in df.columns if "edu" in c.lower()]
        edu_col = matches[0] if matches else None

    if edu_col:
        result["loan_purpose"] = map_education_to_purpose(df[edu_col])
        print(f"  loan_purpose: mapped from '{edu_col}' (education level)")
    else:
        result["loan_purpose"] = "debt_consolidation"
        print("  loan_purpose: education column not found, using debt_consolidation")

    # --- home_ownership (from MARRIAGE) ---
    marriage_col = "marriage" if "marriage" in df.columns else "MARRIAGE"
    if marriage_col not in df.columns:
        matches = [c for c in df.columns if "marr" in c.lower()]
        marriage_col = matches[0] if matches else None

    if marriage_col:
        result["home_ownership"] = map_marriage_to_ownership(df[marriage_col])
        print(f"  home_ownership: mapped from '{marriage_col}' (marital status)")
    else:
        result["home_ownership"] = "RENT"
        print("  home_ownership: marriage column not found, using RENT")

    # --- loan_term_months ---
    # Clients with higher credit limits tend to take longer-term loans
    result["loan_term_months"] = limit_bal.apply(
        lambda x: 60 if x > 200_000 else 36
    ).astype(int)
    print("  loan_term_months: 36 months if LIMIT_BAL ≤ 200k else 60 months")

    # --- loan_default (target) ---
    target_col = None
    for candidate in [
        "default_payment_next_month",
        "default-payment-next-month",
        "y",
        "default",
        "class",
    ]:
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is None:
        # Use the last column (typically the target in UCI datasets)
        target_col = df.columns[-1]
        print(f"  WARNING: target column not found by name, using last column: '{target_col}'")

    result["loan_default"] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    default_rate = result["loan_default"].mean()
    print(f"  loan_default: from '{target_col}' | default rate = {default_rate:.1%}")

    return result


def validate_output(df, output_path: Path) -> dict:
    """Validate the preprocessed DataFrame against Pandera RawLoanSchema.

    Returns a validation report dict.
    """
    import polars as pl
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from loan_risk.data.schemas import validate_raw
    from loan_risk.exceptions import DataValidationError

    print("\nValidating against Pandera schema...")
    pl_df = pl.from_pandas(df)

    try:
        validated = validate_raw(pl_df)
        n_valid = len(validated)
        report = {
            "status": "PASSED",
            "n_rows": n_valid,
            "n_columns": len(validated.columns),
            "default_rate": float(validated["loan_default"].mean()),
            "validation_errors": [],
        }
        print(f"  Schema validation PASSED: {n_valid:,} rows")
    except DataValidationError as exc:
        report = {
            "status": "FAILED",
            "n_rows": len(df),
            "validation_errors": exc.errors,
        }
        print(f"  Schema validation FAILED: {exc}")
        print("  Attempting to fix common issues...")

        # Auto-fix: clip values to schema bounds
        df["credit_score"] = df["credit_score"].clip(300, 850)
        df["employment_years"] = df["employment_years"].clip(0, 50)
        df["debt_to_income_ratio"] = df["debt_to_income_ratio"].clip(0, 2)
        df["loan_amount"] = df["loan_amount"].clip(100, 100_000)
        df["annual_income"] = df["annual_income"].clip(1_000, 10_000_000)
        df["num_delinquencies"] = df["num_delinquencies"].clip(0, 50)
        df["num_open_accounts"] = df["num_open_accounts"].clip(0, 100)

        # Retry validation
        try:
            pl_df_fixed = pl.from_pandas(df)
            validated = validate_raw(pl_df_fixed)
            report["status"] = "PASSED_AFTER_FIX"
            report["n_rows"] = len(validated)
            print(f"  Schema validation PASSED after auto-fix: {len(validated):,} rows")
        except DataValidationError as exc2:
            print(f"  Schema validation still failing after fix: {exc2}")

    # Write validation report
    report_dir = Path("reports/validation")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "validation_report.json"
    import json
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Validation report saved to: {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess UCI Credit Card Default dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/raw/credit_default_raw.csv",
        help="Path to raw UCI CSV (default: data/raw/credit_default_raw.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/raw/loans.parquet",
        help="Output Parquet path (default: data/raw/loans.parquet)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Run download_dataset.py first.")
        sys.exit(1)

    # Load and preprocess
    raw_df = load_raw(input_path)
    processed_df = preprocess(raw_df)

    print(f"\nPreprocessed shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")
    print("\nSample (first 3 rows):")
    print(processed_df.head(3).to_string())

    # Validate against schema
    validation_report = validate_output(processed_df, output_path)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import polars as pl
    pl.from_pandas(processed_df).write_parquet(output_path)
    print(f"\nSaved preprocessed data to: {output_path}")
    print(f"  Rows: {len(processed_df):,}")
    print(f"  Default rate: {processed_df['loan_default'].mean():.1%}")

    if validation_report["status"].startswith("PASSED"):
        print("\nPreprocessing complete!")
    else:
        print(f"\nWARNING: Validation status: {validation_report['status']}")
        sys.exit(1)


if __name__ == "__main__":
    main()

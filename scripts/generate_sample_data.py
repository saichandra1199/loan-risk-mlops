#!/usr/bin/env python3
"""Generate synthetic loan application data for development and testing.

Usage:
    uv run python scripts/generate_sample_data.py --n-rows 50000
    uv run python scripts/generate_sample_data.py --n-rows 1000 --output data/raw/sample.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def generate_loan_data(n_rows: int = 50000, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic but realistic loan application data.

    Default rate is approximately 15%, calibrated via a logistic model
    on the synthetic features so feature-target correlations are meaningful.
    """
    rng = np.random.default_rng(seed)

    loan_ids = [f"LOAN_{i:07d}" for i in range(1, n_rows + 1)]

    loan_amounts = rng.lognormal(mean=9.5, sigma=0.6, size=n_rows).clip(1_000, 40_000)
    annual_incomes = rng.lognormal(mean=10.9, sigma=0.5, size=n_rows).clip(15_000, 300_000)
    employment_years = rng.integers(0, 31, size=n_rows)
    credit_scores = rng.normal(loc=680, scale=80, size=n_rows).clip(300, 850).astype(int)
    debt_to_income = rng.beta(2, 5, size=n_rows).clip(0.01, 0.99)
    num_open_accounts = rng.integers(1, 20, size=n_rows)
    num_delinquencies = rng.negative_binomial(1, 0.7, size=n_rows).clip(0, 10)

    loan_purposes = rng.choice(
        ["debt_consolidation", "home_improvement", "major_purchase", "medical", "vacation", "other"],
        size=n_rows,
        p=[0.45, 0.20, 0.15, 0.08, 0.05, 0.07],
    )
    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN"],
        size=n_rows,
        p=[0.45, 0.40, 0.15],
    )
    loan_term_months = rng.choice([36, 60], size=n_rows, p=[0.6, 0.4])

    # Calibrated logistic model to produce ~15% default rate
    log_odds = (
        -3.5
        + 0.8 * (debt_to_income - 0.3)
        + (-0.008) * (credit_scores - 680)
        + 0.5 * num_delinquencies
        + (-0.03) * employment_years
        + 0.2 * (loan_amounts / annual_incomes - 0.3)
        + rng.normal(0, 0.5, size=n_rows)
    )
    default_prob = 1 / (1 + np.exp(-log_odds))
    loan_defaults = (rng.uniform(size=n_rows) < default_prob).astype(int)

    return pl.DataFrame(
        {
            "loan_id": loan_ids,
            "loan_amount": loan_amounts.round(2),
            "annual_income": annual_incomes.round(2),
            "employment_years": employment_years.tolist(),
            "credit_score": credit_scores.tolist(),
            "debt_to_income_ratio": debt_to_income.round(4),
            "num_open_accounts": num_open_accounts.tolist(),
            "num_delinquencies": num_delinquencies.tolist(),
            "loan_purpose": loan_purposes.tolist(),
            "home_ownership": home_ownership.tolist(),
            "loan_term_months": loan_term_months.tolist(),
            "loan_default": loan_defaults.tolist(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic loan data")
    parser.add_argument("--n-rows", type=int, default=50_000, help="Number of rows to generate")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/loans.parquet",
        help="Output path (.parquet or .csv)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_rows:,} synthetic loan records...")
    df = generate_loan_data(n_rows=args.n_rows, seed=args.seed)

    default_rate = df["loan_default"].mean()
    print(f"Default rate: {default_rate:.1%}")
    print(f"Schema: {df.schema}")

    if output_path.suffix == ".csv":
        df.write_csv(output_path)
    else:
        df.write_parquet(output_path)

    print(f"Saved {len(df):,} rows to {output_path}")

    # Also write a small fixture for tests
    fixture_path = Path("tests/fixtures/sample_loan_data.csv")
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    df.head(200).write_csv(fixture_path)
    print(f"Saved 200-row fixture to {fixture_path}")


if __name__ == "__main__":
    main()

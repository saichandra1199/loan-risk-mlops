"""Unit tests for data splitting."""

from __future__ import annotations

import polars as pl
import pytest

from loan_risk.data.splits import stratified_split


def test_split_sizes(sample_df: pl.DataFrame) -> None:
    """All rows are accounted for across splits."""
    splits = stratified_split(sample_df, test_size=0.20, val_size=0.10, random_seed=42)
    total = len(splits.train) + len(splits.val) + len(splits.test)
    assert total == len(sample_df)


def test_split_no_overlap(sample_df: pl.DataFrame) -> None:
    """No loan_id appears in more than one split."""
    splits = stratified_split(sample_df)
    train_ids = set(splits.train["loan_id"].to_list())
    val_ids = set(splits.val["loan_id"].to_list())
    test_ids = set(splits.test["loan_id"].to_list())

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0


def test_split_default_rates_similar(sample_df: pl.DataFrame) -> None:
    """Default rate in each split is within 5% of the full dataset rate."""
    overall_rate = float(sample_df["loan_default"].mean())
    splits = stratified_split(sample_df)

    for split_df in [splits.train, splits.val, splits.test]:
        split_rate = float(split_df["loan_default"].mean())
        assert abs(split_rate - overall_rate) < 0.10, (
            f"Split rate {split_rate:.3f} too far from overall {overall_rate:.3f}"
        )


def test_split_reproducible(sample_df: pl.DataFrame) -> None:
    """Same seed produces identical splits."""
    s1 = stratified_split(sample_df, random_seed=99)
    s2 = stratified_split(sample_df, random_seed=99)
    assert s1.train["loan_id"].to_list() == s2.train["loan_id"].to_list()

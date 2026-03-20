"""Stratified train/validation/test splitting for the loan dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split

from loan_risk.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class DataSplits:
    """Container for train/val/test DataFrames."""

    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame

    @classmethod
    def from_dir(cls, directory: str | Path) -> DataSplits:
        """Load train/val/test splits from a directory of Parquet files."""
        path = Path(directory)
        return cls(
            train=pl.read_parquet(path / "train.parquet"),
            val=pl.read_parquet(path / "val.parquet"),
            test=pl.read_parquet(path / "test.parquet"),
        )

    @property
    def target_column(self) -> str:
        return "loan_default"

    def summary(self) -> dict[str, dict[str, int | float]]:
        """Return size and default rate for each split."""
        result = {}
        for name, df in [("train", self.train), ("val", self.val), ("test", self.test)]:
            result[name] = {
                "n_rows": len(df),
                "n_defaults": int(df[self.target_column].sum()),
                "default_rate": float(df[self.target_column].mean()),
            }
        return result


def stratified_split(
    df: pl.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.10,
    target_column: str = "loan_default",
    random_seed: int = 42,
) -> DataSplits:
    """Stratified train/val/test split preserving class ratios.

    The val_size is relative to the remaining train+val data after
    the test set is carved out.

    Args:
        df: Full dataset with target column.
        test_size: Fraction for test set (relative to full dataset).
        val_size: Fraction for validation set (relative to full dataset).
        target_column: Name of binary target column.
        random_seed: Reproducibility seed.

    Returns:
        DataSplits with train, val, and test DataFrames.
    """
    y = df[target_column].to_numpy()

    # First split: carve out test set
    idx_trainval, idx_test = train_test_split(
        range(len(df)),
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )

    # Second split: carve val from remaining train+val
    # val_relative = val_size / (1 - test_size) makes val_size absolute
    val_relative = val_size / (1.0 - test_size)
    y_trainval = y[list(idx_trainval)]

    idx_train_local, idx_val_local = train_test_split(
        range(len(idx_trainval)),
        test_size=val_relative,
        stratify=y_trainval,
        random_state=random_seed,
    )

    # Map local indices back to full DataFrame indices
    idx_trainval_list = list(idx_trainval)
    idx_train = [idx_trainval_list[i] for i in idx_train_local]
    idx_val = [idx_trainval_list[i] for i in idx_val_local]
    idx_test_list = list(idx_test)

    train_df = df[idx_train]
    val_df = df[idx_val]
    test_df = df[idx_test_list]

    splits = DataSplits(train=train_df, val=val_df, test=test_df)
    summary = splits.summary()

    logger.info(
        "data_split_complete",
        total=len(df),
        train=summary["train"]["n_rows"],
        val=summary["val"]["n_rows"],
        test=summary["test"]["n_rows"],
        train_default_rate=f"{summary['train']['default_rate']:.3f}",
        val_default_rate=f"{summary['val']['default_rate']:.3f}",
        test_default_rate=f"{summary['test']['default_rate']:.3f}",
    )

    return splits

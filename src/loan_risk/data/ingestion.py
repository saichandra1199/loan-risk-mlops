"""Data ingestion: load raw CSV/Parquet files into Polars DataFrames."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from loan_risk.exceptions import DataIngestionError
from loan_risk.logging_setup import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str | Path) -> pl.DataFrame:
    """Load raw loan data from CSV or Parquet.

    Args:
        path: Path to the data file (.csv or .parquet).

    Returns:
        Polars DataFrame with all original columns.

    Raises:
        DataIngestionError: If the file cannot be read or is empty.
    """
    path = Path(path)

    if not path.exists():
        raise DataIngestionError(f"Data file not found: {path}")

    logger.info("loading_data", path=str(path), suffix=path.suffix)

    try:
        if path.suffix == ".csv":
            df = pl.read_csv(path, infer_schema_length=10_000)
        elif path.suffix == ".parquet":
            df = pl.read_parquet(path)
        else:
            raise DataIngestionError(
                f"Unsupported file format: {path.suffix}. Expected .csv or .parquet"
            )
    except Exception as exc:
        if isinstance(exc, DataIngestionError):
            raise
        raise DataIngestionError(f"Failed to read {path}: {exc}") from exc

    if len(df) == 0:
        raise DataIngestionError(f"Data file is empty: {path}")

    logger.info(
        "data_loaded",
        path=str(path),
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=df.columns,
    )
    return df


def load_data_dir(directory: str | Path, glob_pattern: str = "*.parquet") -> pl.DataFrame:
    """Load and concatenate all matching files from a directory.

    Args:
        directory: Directory to scan for data files.
        glob_pattern: File pattern to match (default: *.parquet).

    Returns:
        Concatenated Polars DataFrame.

    Raises:
        DataIngestionError: If no files found or concatenation fails.
    """
    directory = Path(directory)
    files = sorted(directory.glob(glob_pattern))

    if not files:
        raise DataIngestionError(
            f"No files matching '{glob_pattern}' in {directory}"
        )

    logger.info("loading_directory", directory=str(directory), n_files=len(files))

    frames = [load_raw_data(f) for f in files]
    combined = pl.concat(frames, how="diagonal")

    logger.info("directory_loaded", n_rows=len(combined), n_files=len(files))
    return combined

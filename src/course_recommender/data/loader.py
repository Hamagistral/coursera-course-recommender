"""Data loading utilities for the course recommender."""

import os

import pandas as pd

from course_recommender.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load raw Coursera dataset from a CSV file.

    Handles common encoding issues and provides basic statistics
    about the loaded data.

    Args:
        filepath: Path to the raw CSV file.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """

    def __init__(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Dataset not found at '{filepath}'. "
                "Run scripts/download_data.sh to download it from Kaggle."
            )
        self.filepath = filepath
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Load the raw CSV into a DataFrame.

        Tries UTF-8 first, then falls back to latin-1 to handle
        special characters common in course descriptions.

        Returns:
            Raw DataFrame with all original columns.
        """
        logger.info("Loading dataset from '%s'", self.filepath)
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                self._df = pd.read_csv(self.filepath, encoding=encoding)
                logger.info(
                    "Loaded %d rows x %d columns (encoding=%s)",
                    len(self._df),
                    len(self._df.columns),
                    encoding,
                )
                return self._df
            except UnicodeDecodeError:
                logger.debug("Encoding %s failed, trying next...", encoding)
                continue

        raise ValueError(f"Could not decode '{self.filepath}' with any supported encoding.")

    def get_statistics(self) -> dict:
        """Return basic statistics about the raw dataset.

        Returns:
            Dictionary with shape, dtypes summary, missing value counts,
            and memory usage.

        Raises:
            RuntimeError: If load() has not been called yet.
        """
        if self._df is None:
            raise RuntimeError("Call load() before get_statistics().")

        df = self._df
        return {
            "shape": df.shape,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "duplicate_rows": int(df.duplicated().sum()),
        }

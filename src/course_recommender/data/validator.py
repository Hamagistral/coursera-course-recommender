"""Data validation utilities."""

import pandas as pd

from course_recommender.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["course_id", "combined_text"]


def validate_cleaned_dataframe(df: pd.DataFrame) -> list[str]:
    """Validate a cleaned DataFrame before model training.

    Args:
        df: Cleaned DataFrame to validate.

    Returns:
        List of validation error messages. Empty list means valid.
    """
    errors: list[str] = []

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    if "course_id" in df.columns and df["course_id"].duplicated().any():
        errors.append("Duplicate course_id values found")

    if "combined_text" in df.columns:
        empty_mask = df["combined_text"].fillna("").str.strip() == ""
        n_empty = int(empty_mask.sum())
        if n_empty > 0:
            errors.append(f"{n_empty} courses have empty combined_text")

    if len(df) == 0:
        errors.append("DataFrame is empty after cleaning")

    if errors:
        for err in errors:
            logger.warning("Validation error: %s", err)
    else:
        logger.info("DataFrame validation passed (%d rows)", len(df))

    return errors

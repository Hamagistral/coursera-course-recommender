"""Data cleaning and transformation logic for the course recommender."""

import re
import unicodedata

import pandas as pd

from course_recommender.utils.logger import get_logger

logger = get_logger(__name__)

# Columns that must be present and non-null for a course to be kept
CRITICAL_COLUMNS = ["Course Title", "Course Url"]

# Accepted difficulty levels (case-insensitive matching)
VALID_LEVELS = {"beginner", "intermediate", "advanced", "mixed"}


class DataCleaner:
    """Clean and transform the raw Coursera DataFrame.

    Implements the Builder / method-chaining pattern so operations can
    be composed fluently:

    Example:
        >>> cleaner = (
        ...     DataCleaner(raw_df)
        ...     .remove_duplicates()
        ...     .handle_missing_values()
        ...     .standardize_text()
        ...     .validate_data()
        ...     .create_features()
        ... )
        >>> cleaned_df = cleaner.get_cleaned_data()
        >>> report = cleaner.get_cleaning_report()

    Args:
        df: Raw DataFrame loaded by DataLoader.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()
        self._original_len = len(df)
        self._report: dict = {
            "original_rows": self._original_len,
            "duplicates_removed": 0,
            "missing_critical_dropped": 0,
            "invalid_ratings_dropped": 0,
            "steps_applied": [],
        }

    # ------------------------------------------------------------------
    # Public methods (builder pattern)
    # ------------------------------------------------------------------

    def remove_duplicates(self) -> "DataCleaner":
        """Remove duplicate courses.

        Deduplication strategy:
        1. By Course URL (most reliable unique identifier).
        2. By Course Name + University (catches URL variants of same course).

        Returns:
            Self for method chaining.
        """
        before = len(self._df)

        # Step 1: deduplicate by URL
        url_col = self._find_column(["Course Url", "Course URL", "course_url", "URL"])
        if url_col:
            self._df = self._df.drop_duplicates(subset=[url_col], keep="first")

        # Step 2: deduplicate by name + university
        name_col = self._find_column(["Course Title", "Course Name", "course_name", "Title"])
        uni_col = self._find_column(["Offered By", "University", "university", "Organization"])
        if name_col and uni_col:
            self._df = self._df.drop_duplicates(subset=[name_col, uni_col], keep="first")
        elif name_col:
            self._df = self._df.drop_duplicates(subset=[name_col], keep="first")

        removed = before - len(self._df)
        self._report["duplicates_removed"] = removed
        self._report["steps_applied"].append("remove_duplicates")
        logger.info("Removed %d duplicate courses (%d remaining)", removed, len(self._df))
        return self

    def handle_missing_values(self) -> "DataCleaner":
        """Handle missing values.

        Strategy:
        - Drop rows where Course Name or Course URL is missing.
        - Fill missing text fields (description, skills) with empty strings.
        - Fill missing ratings with NaN (kept for optional filtering).

        Returns:
            Self for method chaining.
        """
        before = len(self._df)

        # Drop rows with missing critical fields
        critical_cols = [c for c in CRITICAL_COLUMNS if c in self._df.columns]
        if critical_cols:
            self._df = self._df.dropna(subset=critical_cols)

        dropped = before - len(self._df)
        self._report["missing_critical_dropped"] = dropped

        # Fill text fields
        text_fill_cols = [
            c
            for c in self._df.columns
            if any(
                kw in c.lower()
                for kw in ["description", "skill", "about", "summary", "detail"]
            )
        ]
        for col in text_fill_cols:
            self._df[col] = self._df[col].fillna("")

        self._report["steps_applied"].append("handle_missing_values")
        logger.info(
            "Dropped %d rows with missing critical fields (%d remaining)", dropped, len(self._df)
        )
        return self

    def standardize_text(self) -> "DataCleaner":
        """Clean and standardize all text columns.

        Operations applied:
        - Normalize unicode characters (NFKC).
        - Remove non-printable / control characters.
        - Collapse multiple whitespace into single space.
        - Strip leading/trailing whitespace.

        Returns:
            Self for method chaining.
        """
        text_cols = self._df.select_dtypes(include="object").columns.tolist()
        for col in text_cols:
            self._df[col] = self._df[col].apply(self._clean_text)

        self._report["steps_applied"].append("standardize_text")
        logger.info("Standardized text in %d columns", len(text_cols))
        return self

    def validate_data(self) -> "DataCleaner":
        """Validate data quality and remove invalid records.

        Checks:
        - Course Rating must be in [0, 5] if present.
        - Difficulty level must be one of the accepted values (or unknown).

        Returns:
            Self for method chaining.
        """
        before = len(self._df)

        # Validate ratings
        rating_col = self._find_column(["Course Rating", "Rating", "rating"])
        if rating_col:
            self._df[rating_col] = pd.to_numeric(self._df[rating_col], errors="coerce")
            invalid_rating_mask = self._df[rating_col].notna() & (
                (self._df[rating_col] < 0) | (self._df[rating_col] > 5)
            )
            self._df = self._df[~invalid_rating_mask]
            self._report["invalid_ratings_dropped"] = int(invalid_rating_mask.sum())

        # Standardize difficulty levels
        level_col = self._find_column(
            ["Difficulty Level", "difficulty_level", "Level", "level"]
        )
        if level_col:
            self._df[level_col] = self._df[level_col].apply(self._standardize_level)

        dropped = before - len(self._df)
        self._report["steps_applied"].append("validate_data")
        logger.info(
            "Validation removed %d invalid rows (%d remaining)", dropped, len(self._df)
        )
        return self

    def create_features(self) -> "DataCleaner":
        """Create derived features for modeling.

        New columns created:
        - ``course_id``: Sequential integer index.
        - ``combined_text``: Concatenation of Name + Description + Skills + Level.
        - ``has_rating``: Boolean flag.
        - ``num_skills``: Count of skills (comma-separated).
        - ``text_length``: Character length of description.

        Returns:
            Self for method chaining.
        """
        # Reset index so course_id is sequential after all drops
        self._df = self._df.reset_index(drop=True)
        self._df["course_id"] = self._df.index

        # Identify relevant columns
        name_col = self._find_column(["Course Title", "Course Name", "Title", "name"]) or ""
        desc_col = self._find_column(["What you will learn", "Course Description", "Description", "About"]) or ""
        skills_col = self._find_column(["Skill gain", "Skills", "skills", "Skills You'll Learn"]) or ""
        level_col = self._find_column(["Difficulty Level", "Level", "level"]) or ""
        rating_col = self._find_column(["Course Rating", "Rating"]) or ""

        def _get_col(col: str) -> pd.Series:
            return self._df[col].fillna("").astype(str) if col and col in self._df.columns else pd.Series([""] * len(self._df))

        self._df["combined_text"] = (
            _get_col(name_col)
            + " "
            + _get_col(desc_col)
            + " "
            + _get_col(skills_col)
            + " "
            + _get_col(level_col)
        ).str.strip()

        if rating_col and rating_col in self._df.columns:
            self._df["has_rating"] = self._df[rating_col].notna()
        else:
            self._df["has_rating"] = False

        if skills_col and skills_col in self._df.columns:
            self._df["num_skills"] = (
                self._df[skills_col]
                .fillna("")
                .astype(str)
                .apply(lambda x: len([s for s in x.split(",") if s.strip()]))
            )
        else:
            self._df["num_skills"] = 0

        if desc_col and desc_col in self._df.columns:
            self._df["text_length"] = self._df[desc_col].fillna("").astype(str).str.len()
        else:
            self._df["text_length"] = 0

        self._report["steps_applied"].append("create_features")
        self._report["final_rows"] = len(self._df)
        logger.info("Created derived features. Final dataset: %d rows", len(self._df))
        return self

    def get_cleaned_data(self) -> pd.DataFrame:
        """Return the cleaned DataFrame.

        Returns:
            Cleaned DataFrame with all transformations applied.
        """
        return self._df.copy()

    def get_cleaning_report(self) -> dict:
        """Return a summary report of all cleaning actions taken.

        Returns:
            Dictionary with counts of removed rows, steps applied, etc.
        """
        report = self._report.copy()
        report["final_rows"] = report.get("final_rows", len(self._df))
        report["total_removed"] = self._original_len - report["final_rows"]
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_column(self, candidates: list[str]) -> str | None:
        """Return the first candidate column name that exists in the DataFrame."""
        for name in candidates:
            if name in self._df.columns:
                return name
        return None

    @staticmethod
    def _clean_text(value: object) -> str:
        """Normalize and clean a single text value."""
        if not isinstance(value, str):
            return str(value) if value is not None else ""
        # Unicode normalization
        text = unicodedata.normalize("NFKC", value)
        # Remove control characters
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _standardize_level(level: object) -> str:
        """Map a raw difficulty level string to a canonical value."""
        if not isinstance(level, str) or not level.strip():
            return "Unknown"
        normalized = level.strip().lower()
        if "begin" in normalized or "basic" in normalized or "introduct" in normalized:
            return "Beginner"
        if "intermedia" in normalized:
            return "Intermediate"
        if "advanc" in normalized or "expert" in normalized:
            return "Advanced"
        if "mix" in normalized or "all" in normalized:
            return "Mixed"
        # Capitalize known single-word levels
        canonical = normalized.capitalize()
        return canonical if canonical.lower() in VALID_LEVELS else canonical

"""Tests for DataCleaner."""

import pandas as pd
import pytest

from course_recommender.data.cleaner import DataCleaner


class TestRemoveDuplicates:
    def test_removes_url_duplicates(self, sample_raw_df: pd.DataFrame) -> None:
        cleaner = DataCleaner(sample_raw_df).remove_duplicates()
        result = cleaner.get_cleaned_data()
        # Duplicate URL should be removed
        assert result["Course URL"].duplicated().sum() == 0

    def test_report_tracks_duplicates_removed(self, sample_raw_df: pd.DataFrame) -> None:
        cleaner = DataCleaner(sample_raw_df).remove_duplicates()
        report = cleaner.get_cleaning_report()
        assert report["duplicates_removed"] >= 1

    def test_no_duplicates_unchanged(self) -> None:
        df = pd.DataFrame(
            {
                "Course Name": ["A", "B"],
                "Course URL": ["https://a.com", "https://b.com"],
                "University": ["X", "Y"],
            }
        )
        cleaner = DataCleaner(df).remove_duplicates()
        assert len(cleaner.get_cleaned_data()) == 2


class TestHandleMissingValues:
    def test_drops_rows_with_missing_course_name(self, sample_raw_df: pd.DataFrame) -> None:
        cleaner = DataCleaner(sample_raw_df).handle_missing_values()
        result = cleaner.get_cleaned_data()
        assert result["Course Name"].isna().sum() == 0

    def test_fills_text_fields(self) -> None:
        df = pd.DataFrame(
            {
                "Course Name": ["A"],
                "Course URL": ["https://a.com"],
                "Course Description": [None],
                "Skills": [None],
            }
        )
        cleaner = DataCleaner(df).handle_missing_values()
        result = cleaner.get_cleaned_data()
        assert result["Course Description"].iloc[0] == ""
        assert result["Skills"].iloc[0] == ""


class TestStandardizeText:
    def test_strips_whitespace(self) -> None:
        df = pd.DataFrame(
            {
                "Course Name": ["  Python Course  "],
                "Course URL": ["https://a.com"],
                "Course Description": ["  Some description.  "],
            }
        )
        cleaner = DataCleaner(df).standardize_text()
        result = cleaner.get_cleaned_data()
        assert result["Course Name"].iloc[0] == "Python Course"
        assert result["Course Description"].iloc[0] == "Some description."

    def test_removes_control_characters(self) -> None:
        df = pd.DataFrame(
            {
                "Course Name": ["Python\x00Course"],
                "Course URL": ["https://a.com"],
            }
        )
        cleaner = DataCleaner(df).standardize_text()
        result = cleaner.get_cleaned_data()
        assert "\x00" not in result["Course Name"].iloc[0]


class TestValidateData:
    def test_removes_invalid_ratings(self, sample_raw_df: pd.DataFrame) -> None:
        cleaner = DataCleaner(sample_raw_df).validate_data()
        result = cleaner.get_cleaned_data()
        ratings = pd.to_numeric(result["Course Rating"], errors="coerce").dropna()
        assert (ratings <= 5).all()
        assert (ratings >= 0).all()

    def test_standardizes_difficulty_level(self, sample_raw_df: pd.DataFrame) -> None:
        cleaner = DataCleaner(sample_raw_df).validate_data()
        result = cleaner.get_cleaned_data()
        levels = result["Difficulty Level"].dropna().unique()
        # Should not have lowercase raw values
        for level in levels:
            assert level[0].isupper() or level == "Unknown"


class TestCreateFeatures:
    def test_creates_course_id(self, sample_raw_df: pd.DataFrame) -> None:
        cleaner = (
            DataCleaner(sample_raw_df)
            .remove_duplicates()
            .handle_missing_values()
            .create_features()
        )
        result = cleaner.get_cleaned_data()
        assert "course_id" in result.columns
        assert result["course_id"].is_monotonic_increasing
        assert result["course_id"].duplicated().sum() == 0

    def test_creates_combined_text(self, sample_raw_df: pd.DataFrame) -> None:
        cleaner = (
            DataCleaner(sample_raw_df)
            .remove_duplicates()
            .handle_missing_values()
            .create_features()
        )
        result = cleaner.get_cleaned_data()
        assert "combined_text" in result.columns
        assert result["combined_text"].str.len().min() > 0

    def test_creates_num_skills(self, sample_raw_df: pd.DataFrame) -> None:
        cleaner = (
            DataCleaner(sample_raw_df)
            .remove_duplicates()
            .handle_missing_values()
            .create_features()
        )
        result = cleaner.get_cleaned_data()
        assert "num_skills" in result.columns
        assert (result["num_skills"] >= 0).all()

    def test_full_pipeline(self, sample_raw_df: pd.DataFrame) -> None:
        """End-to-end pipeline should not raise and should reduce row count."""
        cleaner = (
            DataCleaner(sample_raw_df)
            .remove_duplicates()
            .handle_missing_values()
            .standardize_text()
            .validate_data()
            .create_features()
        )
        result = cleaner.get_cleaned_data()
        report = cleaner.get_cleaning_report()

        assert len(result) < len(sample_raw_df)
        assert "course_id" in result.columns
        assert "combined_text" in result.columns
        assert report["total_removed"] > 0

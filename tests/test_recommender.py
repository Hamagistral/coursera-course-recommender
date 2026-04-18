"""Tests for ContentBasedRecommender."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from course_recommender.models.recommender import ContentBasedRecommender


@pytest.fixture
def fitted_recommender(sample_cleaned_df: pd.DataFrame) -> ContentBasedRecommender:
    """Return a fitted recommender using the smallest available model."""
    rec = ContentBasedRecommender(model_name="all-MiniLM-L6-v2")
    rec.fit(sample_cleaned_df, show_progress=False)
    return rec


class TestFit:
    def test_fit_generates_embeddings(self, sample_cleaned_df: pd.DataFrame) -> None:
        rec = ContentBasedRecommender("all-MiniLM-L6-v2")
        rec.fit(sample_cleaned_df, show_progress=False)
        embeddings = rec.get_embeddings()
        assert embeddings.shape[0] == len(sample_cleaned_df)
        assert embeddings.shape[1] == 384  # MiniLM output dim

    def test_fit_returns_self(self, sample_cleaned_df: pd.DataFrame) -> None:
        rec = ContentBasedRecommender("all-MiniLM-L6-v2")
        returned = rec.fit(sample_cleaned_df, show_progress=False)
        assert returned is rec

    def test_fit_raises_on_missing_column(self, sample_cleaned_df: pd.DataFrame) -> None:
        rec = ContentBasedRecommender("all-MiniLM-L6-v2")
        with pytest.raises(ValueError, match="not found"):
            rec.fit(sample_cleaned_df, text_column="nonexistent_column")

    def test_not_fitted_raises(self) -> None:
        rec = ContentBasedRecommender()
        with pytest.raises(RuntimeError, match="fitted"):
            rec.recommend_similar(0)


class TestRecommendSimilar:
    def test_returns_correct_number(
        self, fitted_recommender: ContentBasedRecommender
    ) -> None:
        recs = fitted_recommender.recommend_similar(course_id=0, top_k=2)
        assert len(recs) == 2

    def test_excludes_source_course(
        self, fitted_recommender: ContentBasedRecommender
    ) -> None:
        recs = fitted_recommender.recommend_similar(course_id=0, top_k=3, exclude_same_course=True)
        assert 0 not in recs["course_id"].tolist()

    def test_includes_source_when_disabled(
        self, fitted_recommender: ContentBasedRecommender
    ) -> None:
        recs = fitted_recommender.recommend_similar(
            course_id=0, top_k=4, exclude_same_course=False
        )
        assert 0 in recs["course_id"].tolist()

    def test_similarity_scores_in_range(
        self, fitted_recommender: ContentBasedRecommender
    ) -> None:
        recs = fitted_recommender.recommend_similar(course_id=0, top_k=3)
        assert "similarity_score" in recs.columns
        scores = recs["similarity_score"]
        assert (scores >= -1.0).all()
        assert (scores <= 1.0).all()

    def test_invalid_course_id_raises(
        self, fitted_recommender: ContentBasedRecommender
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            fitted_recommender.recommend_similar(course_id=9999)

    def test_results_sorted_by_similarity(
        self, fitted_recommender: ContentBasedRecommender
    ) -> None:
        recs = fitted_recommender.recommend_similar(course_id=0, top_k=3)
        scores = recs["similarity_score"].tolist()
        assert scores == sorted(scores, reverse=True)


class TestSearch:
    def test_returns_results(self, fitted_recommender: ContentBasedRecommender) -> None:
        results = fitted_recommender.search("machine learning", top_k=3)
        assert len(results) == 3
        assert "similarity_score" in results.columns

    def test_search_scores_sorted(self, fitted_recommender: ContentBasedRecommender) -> None:
        results = fitted_recommender.search("python programming", top_k=4)
        scores = results["similarity_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_python_query_returns_python_course(
        self, fitted_recommender: ContentBasedRecommender
    ) -> None:
        results = fitted_recommender.search("Python for beginners", top_k=1)
        # Top result should be Python-related
        name_col = next(
            (c for c in results.columns if "name" in c.lower() and c != "combined_text"),
            None,
        )
        if name_col:
            assert "Python" in str(results.iloc[0][name_col])


class TestSaveLoad:
    def test_save_and_load(
        self,
        fitted_recommender: ContentBasedRecommender,
        sample_cleaned_df: pd.DataFrame,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rec.pkl")
            fitted_recommender.save(path)
            assert os.path.exists(path)

            loaded = ContentBasedRecommender.load(path)
            assert loaded.model_name == fitted_recommender.model_name
            assert loaded.num_courses == fitted_recommender.num_courses
            assert loaded.embedding_dim == fitted_recommender.embedding_dim

            # Verify recommendations work after load
            recs = loaded.recommend_similar(0, top_k=2)
            assert len(recs) == 2

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            ContentBasedRecommender.load("/nonexistent/path.pkl")

"""Tests for RecommenderEvaluator."""

import pandas as pd
import pytest

from course_recommender.models.evaluator import RecommenderEvaluator
from course_recommender.models.recommender import ContentBasedRecommender


@pytest.fixture
def evaluator(sample_cleaned_df: pd.DataFrame) -> RecommenderEvaluator:
    """Return an evaluator with a fitted recommender."""
    rec = ContentBasedRecommender("all-MiniLM-L6-v2")
    rec.fit(sample_cleaned_df, show_progress=False)
    return RecommenderEvaluator(rec)


class TestComputeDiversity:
    def test_returns_float(self, evaluator: RecommenderEvaluator) -> None:
        score = evaluator.compute_diversity(sample_size=3, top_k=2)
        assert isinstance(score, float)

    def test_in_valid_range(self, evaluator: RecommenderEvaluator) -> None:
        score = evaluator.compute_diversity(sample_size=3, top_k=2)
        assert 0.0 <= score <= 1.0


class TestComputeCoverage:
    def test_returns_float(self, evaluator: RecommenderEvaluator) -> None:
        score = evaluator.compute_coverage(sample_size=3, top_k=2)
        assert isinstance(score, float)

    def test_in_valid_range(self, evaluator: RecommenderEvaluator) -> None:
        score = evaluator.compute_coverage(sample_size=3, top_k=2)
        assert 0.0 <= score <= 1.0

    def test_positive_coverage(self, evaluator: RecommenderEvaluator) -> None:
        score = evaluator.compute_coverage(sample_size=3, top_k=2)
        assert score > 0.0


class TestComputeAvgSimilarity:
    def test_returns_float(self, evaluator: RecommenderEvaluator) -> None:
        score = evaluator.compute_avg_similarity(sample_size=3, top_k=2)
        assert isinstance(score, float)

    def test_in_valid_range(self, evaluator: RecommenderEvaluator) -> None:
        score = evaluator.compute_avg_similarity(sample_size=3, top_k=2)
        assert -1.0 <= score <= 1.0

    def test_positive_similarity(self, evaluator: RecommenderEvaluator) -> None:
        # For a good recommender, avg similarity should be positive
        score = evaluator.compute_avg_similarity(sample_size=3, top_k=2)
        assert score > 0.0


class TestEvaluate:
    def test_returns_all_metrics(self, evaluator: RecommenderEvaluator) -> None:
        metrics = evaluator.evaluate(sample_size=3, top_k=2)
        assert set(metrics.keys()) == {
            "diversity",
            "coverage",
            "avg_similarity",
            "inference_time_ms",
        }

    def test_all_metrics_are_numeric(self, evaluator: RecommenderEvaluator) -> None:
        metrics = evaluator.evaluate(sample_size=3, top_k=2)
        for key, val in metrics.items():
            assert isinstance(val, (int, float)), f"{key} is not numeric: {val}"

    def test_inference_time_positive(self, evaluator: RecommenderEvaluator) -> None:
        metrics = evaluator.evaluate(sample_size=3, top_k=2)
        assert metrics["inference_time_ms"] > 0

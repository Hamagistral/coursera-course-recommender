"""Evaluation metrics for the content-based recommender."""

import random
import time

import numpy as np
import pandas as pd

from course_recommender.utils.logger import get_logger

logger = get_logger(__name__)


class RecommenderEvaluator:
    """Compute offline evaluation metrics for a ContentBasedRecommender.

    Metrics computed:
    - **Diversity**: Fraction of unique subject categories (Keyword) across all
      recommendation lists.  Higher → recommendations span more subject areas.
    - **Coverage**: Fraction of catalog courses that appear in at least
      one recommendation list.
    - **Avg Similarity**: Mean cosine similarity score across all
      recommendation pairs.  Higher → semantically tighter matches.

    Args:
        recommender: A fitted ContentBasedRecommender instance.

    Example:
        >>> evaluator = RecommenderEvaluator(recommender)
        >>> metrics = evaluator.evaluate(sample_size=100, top_k=5)
        >>> print(metrics)
    """

    def __init__(self, recommender) -> None:  # type: ignore[no-untyped-def]
        self._rec = recommender

    def compute_diversity(self, sample_size: int = 100, top_k: int = 5) -> float:
        """Compute recommendation diversity.

        Samples ``sample_size`` courses, gets ``top_k`` recommendations for
        each, then measures the fraction of unique difficulty levels across
        all results.

        Args:
            sample_size: Number of seed courses to sample.
            top_k: Recommendations per seed course.

        Returns:
            Diversity score in [0, 1].
        """
        assert self._rec.courses_df is not None
        df = self._rec.courses_df
        level_col = self._find_level_col(df)

        sample_ids = self._sample_course_ids(sample_size)
        all_categories: list[str] = []

        # Prefer subject category column over difficulty level for diversity
        assert self._rec.courses_df is not None
        category_col = self._find_category_col(self._rec.courses_df) or level_col

        for cid in sample_ids:
            try:
                recs = self._rec.recommend_similar(cid, top_k=top_k)
                if category_col and category_col in recs.columns:
                    all_categories.extend(recs[category_col].fillna("Unknown").tolist())
                else:
                    all_categories.extend(["Unknown"] * len(recs))
            except Exception:
                continue

        if not all_categories:
            return 0.0

        diversity = len(set(all_categories)) / max(len(all_categories), 1)
        return round(float(diversity), 4)

    def compute_coverage(self, sample_size: int = 100, top_k: int = 5) -> float:
        """Compute catalog coverage.

        Measures the fraction of all catalog courses that are recommended
        at least once across the sampled queries.

        Args:
            sample_size: Number of seed courses to sample.
            top_k: Recommendations per seed course.

        Returns:
            Coverage score in [0, 1].
        """
        assert self._rec.courses_df is not None
        total_courses = len(self._rec.courses_df)
        sample_ids = self._sample_course_ids(sample_size)
        recommended_ids: set[int] = set()

        for cid in sample_ids:
            try:
                recs = self._rec.recommend_similar(cid, top_k=top_k)
                recommended_ids.update(recs["course_id"].tolist())
            except Exception:
                continue

        coverage = len(recommended_ids) / max(total_courses, 1)
        return round(float(coverage), 4)

    def compute_avg_similarity(self, sample_size: int = 100, top_k: int = 5) -> float:
        """Compute average cosine similarity of recommendations.

        Args:
            sample_size: Number of seed courses to sample.
            top_k: Recommendations per seed course.

        Returns:
            Mean cosine similarity score in [0, 1].
        """
        sample_ids = self._sample_course_ids(sample_size)
        scores: list[float] = []

        for cid in sample_ids:
            try:
                recs = self._rec.recommend_similar(cid, top_k=top_k)
                if "similarity_score" in recs.columns:
                    scores.extend(recs["similarity_score"].tolist())
            except Exception:
                continue

        if not scores:
            return 0.0
        return round(float(np.mean(scores)), 4)

    def compute_inference_time(self, n_queries: int = 20, top_k: int = 5) -> float:
        """Measure average inference time per recommendation request.

        Args:
            n_queries: Number of queries to average over.
            top_k: Recommendations per query.

        Returns:
            Average inference time in milliseconds.
        """
        sample_ids = self._sample_course_ids(n_queries)
        times: list[float] = []

        for cid in sample_ids:
            try:
                start = time.perf_counter()
                self._rec.recommend_similar(cid, top_k=top_k)
                times.append((time.perf_counter() - start) * 1000)
            except Exception:
                continue

        if not times:
            return 0.0
        return round(float(np.mean(times)), 2)

    def evaluate(self, sample_size: int = 100, top_k: int = 5) -> dict:
        """Run all evaluation metrics.

        Args:
            sample_size: Number of seed courses to sample.
            top_k: Recommendations per seed course.

        Returns:
            Dictionary with keys: diversity, coverage, avg_similarity,
            inference_time_ms.
        """
        logger.info(
            "Evaluating recommender (%s) — sample_size=%d, top_k=%d",
            self._rec.model_name,
            sample_size,
            top_k,
        )
        metrics = {
            "diversity": self.compute_diversity(sample_size, top_k),
            "coverage": self.compute_coverage(sample_size, top_k),
            "avg_similarity": self.compute_avg_similarity(sample_size, top_k),
            "inference_time_ms": self.compute_inference_time(20, top_k),
        }
        logger.info("Metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_course_ids(self, n: int) -> list[int]:
        """Return up to n random course IDs from the catalog."""
        assert self._rec.courses_df is not None
        ids = self._rec.courses_df["course_id"].tolist()
        k = min(n, len(ids))
        return random.sample(ids, k)

    @staticmethod
    def _find_level_col(df: pd.DataFrame) -> str | None:
        """Find the difficulty level column in the DataFrame."""
        for col in ["Difficulty Level", "difficulty_level", "Level", "level"]:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _find_category_col(df: pd.DataFrame) -> str | None:
        """Find the subject category column in the DataFrame."""
        for col in ["Keyword", "keyword", "Category", "category", "Subject", "subject"]:
            if col in df.columns:
                return col
        return None

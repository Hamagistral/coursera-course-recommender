"""A/B testing framework for comparing recommender variants."""

import pandas as pd
import plotly.graph_objects as go

from course_recommender.models.evaluator import RecommenderEvaluator
from course_recommender.utils.logger import get_logger

logger = get_logger(__name__)


class ABTest:
    """Compare two ContentBasedRecommender variants head-to-head.

    Evaluates both variants with the same metrics and provides
    visualizations and a winner determination.

    Args:
        variant_a: Fitted recommender for Variant A.
        variant_b: Fitted recommender for Variant B.

    Example:
        >>> ab = ABTest(recommender_a, recommender_b)
        >>> comparison = ab.compare_metrics(sample_size=100)
        >>> winner = ab.determine_winner()
    """

    METRIC_LABELS = {
        "diversity": "Diversity",
        "coverage": "Coverage",
        "avg_similarity": "Avg Similarity",
        "inference_time_ms": "Inference Time (ms)",
    }

    def __init__(self, variant_a, variant_b) -> None:  # type: ignore[no-untyped-def]
        self._variant_a = variant_a
        self._variant_b = variant_b
        self._metrics_a: dict = {}
        self._metrics_b: dict = {}

    def compare_metrics(self, sample_size: int = 100, top_k: int = 5) -> pd.DataFrame:
        """Evaluate both variants and return a comparison DataFrame.

        Args:
            sample_size: Number of seed courses to sample per variant.
            top_k: Number of recommendations per seed course.

        Returns:
            DataFrame with a row per metric and columns for Variant A and B.
        """
        logger.info("Running A/B evaluation — sample_size=%d, top_k=%d", sample_size, top_k)

        eval_a = RecommenderEvaluator(self._variant_a)
        eval_b = RecommenderEvaluator(self._variant_b)

        self._metrics_a = eval_a.evaluate(sample_size=sample_size, top_k=top_k)
        self._metrics_b = eval_b.evaluate(sample_size=sample_size, top_k=top_k)

        rows = []
        for key in self._metrics_a:
            val_a = self._metrics_a[key]
            val_b = self._metrics_b[key]
            diff = val_b - val_a
            rows.append(
                {
                    "Metric": self.METRIC_LABELS.get(key, key),
                    "key": key,
                    f"Variant A ({self._variant_a.model_name})": val_a,
                    f"Variant B ({self._variant_b.model_name})": val_b,
                    "Difference (B-A)": round(diff, 4),
                }
            )

        return pd.DataFrame(rows)

    def visualize_comparison(self, title: str = "A/B Test: Variant Comparison") -> go.Figure:
        """Create a Plotly bar chart comparing variant metrics.

        Returns:
            Plotly Figure object.

        Raises:
            RuntimeError: If compare_metrics() has not been called yet.
        """
        if not self._metrics_a or not self._metrics_b:
            raise RuntimeError("Call compare_metrics() before visualize_comparison().")

        # Exclude inference_time_ms from bar chart (different scale)
        exclude = {"inference_time_ms"}
        metrics = [k for k in self._metrics_a if k not in exclude]
        labels = [self.METRIC_LABELS.get(m, m) for m in metrics]

        fig = go.Figure(
            data=[
                go.Bar(
                    name=f"Variant A ({self._variant_a.model_name})",
                    x=labels,
                    y=[self._metrics_a[m] for m in metrics],
                    marker_color="#636EFA",
                ),
                go.Bar(
                    name=f"Variant B ({self._variant_b.model_name})",
                    x=labels,
                    y=[self._metrics_b[m] for m in metrics],
                    marker_color="#EF553B",
                ),
            ]
        )
        fig.update_layout(
            title=title,
            barmode="group",
            yaxis_title="Score",
            legend_title="Variant",
            template="plotly_white",
        )
        return fig

    def determine_winner(self, primary_metric: str = "avg_similarity") -> str:
        """Determine which variant wins based on a primary metric.

        For ``inference_time_ms``, lower is better.  For all other metrics,
        higher is better.

        Args:
            primary_metric: Metric key to base the decision on.

        Returns:
            ``"A"`` or ``"B"`` string.

        Raises:
            RuntimeError: If compare_metrics() has not been called yet.
        """
        if not self._metrics_a or not self._metrics_b:
            raise RuntimeError("Call compare_metrics() before determine_winner().")

        val_a = self._metrics_a.get(primary_metric, 0)
        val_b = self._metrics_b.get(primary_metric, 0)

        lower_is_better = primary_metric == "inference_time_ms"

        if lower_is_better:
            winner = "A" if val_a <= val_b else "B"
        else:
            winner = "B" if val_b >= val_a else "A"

        logger.info(
            "Winner (by %s): Variant %s  [A=%.4f, B=%.4f]",
            primary_metric,
            winner,
            val_a,
            val_b,
        )
        return winner

    @property
    def metrics_a(self) -> dict:
        """Metrics for Variant A (populated after compare_metrics())."""
        return self._metrics_a

    @property
    def metrics_b(self) -> dict:
        """Metrics for Variant B (populated after compare_metrics())."""
        return self._metrics_b

"""MLflow tracking utilities for the course recommender."""

import os

import mlflow

from course_recommender.utils.config import config
from course_recommender.utils.logger import get_logger

logger = get_logger(__name__)


def setup_mlflow(experiment_name: str = config.MLFLOW_EXPERIMENT_NAME) -> None:
    """Configure MLflow tracking URI and set the active experiment.

    Args:
        experiment_name: Name of the MLflow experiment to create/use.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    logger.info(
        "MLflow configured — URI: %s  |  Experiment: %s",
        config.MLFLOW_TRACKING_URI,
        experiment_name,
    )


def log_model_training(
    variant_name: str,
    model_name: str,
    recommender,  # type: ignore[no-untyped-def]
    metrics: dict,
    model_path: str,
) -> str:
    """Log a model training run to MLflow.

    Args:
        variant_name: Variant label, e.g. ``"A"`` or ``"B"``.
        model_name: Sentence-transformers model identifier.
        recommender: Fitted ContentBasedRecommender instance.
        metrics: Evaluation metrics dictionary (from RecommenderEvaluator).
        model_path: Local path to the saved model pickle file.

    Returns:
        MLflow run ID string.
    """
    with mlflow.start_run(run_name=f"variant_{variant_name}") as run:
        # Parameters
        mlflow.log_params(
            {
                "variant": variant_name,
                "model_name": model_name,
                "num_courses": recommender.num_courses,
                "embedding_dim": recommender.embedding_dim,
            }
        )

        # Metrics
        mlflow.log_metrics(metrics)

        # Artifact: saved model file
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="model")

        run_id = run.info.run_id
        logger.info("MLflow run logged — Variant %s | Run ID: %s", variant_name, run_id)
        return run_id


def get_best_run(
    experiment_name: str = config.MLFLOW_EXPERIMENT_NAME,
    metric: str = "avg_similarity",
) -> dict:
    """Retrieve the best MLflow run based on a metric.

    Args:
        experiment_name: MLflow experiment name.
        metric: Metric name to maximise.

    Returns:
        Dictionary with ``run_id``, ``params``, and ``metrics`` of the best run.
        Returns empty dict if no runs exist.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning("Experiment '%s' not found.", experiment_name)
        return {}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        logger.warning("No runs found in experiment '%s'.", experiment_name)
        return {}

    best = runs[0]
    return {
        "run_id": best.info.run_id,
        "params": best.data.params,
        "metrics": best.data.metrics,
    }

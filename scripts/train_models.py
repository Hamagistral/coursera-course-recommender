"""Automated training pipeline for both recommender variants.

Usage:
    uv run python scripts/train_models.py

Workflow:
    1. Load configuration
    2. Setup MLflow
    3. Load and clean data
    4. Train Variant A (all-MiniLM-L6-v2)
    5. Train Variant B (all-mpnet-base-v2)
    6. Evaluate and compare
    7. Save models and metadata
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from course_recommender.data.cleaner import DataCleaner
from course_recommender.data.loader import DataLoader
from course_recommender.data.validator import validate_cleaned_dataframe
from course_recommender.mlops.ab_testing import ABTest
from course_recommender.mlops.mlflow_utils import log_model_training, setup_mlflow
from course_recommender.models.evaluator import RecommenderEvaluator
from course_recommender.models.recommender import ContentBasedRecommender
from course_recommender.utils.config import config
from course_recommender.utils.logger import get_logger

logger = get_logger("train_models")


def load_and_clean_data() -> pd.DataFrame:
    """Load raw data, run cleaning pipeline, and return cleaned DataFrame."""
    logger.info("Step 1/6: Loading raw data from %s", config.RAW_DATA_PATH)
    loader = DataLoader(config.RAW_DATA_PATH)
    raw_df = loader.load()

    logger.info("Step 2/6: Cleaning data...")
    cleaner = (
        DataCleaner(raw_df)
        .remove_duplicates()
        .handle_missing_values()
        .standardize_text()
        .validate_data()
        .create_features()
    )
    cleaned_df = cleaner.get_cleaned_data()
    report = cleaner.get_cleaning_report()

    logger.info(
        "Cleaning complete: %d → %d rows (%d removed)",
        report["original_rows"],
        report["final_rows"],
        report["total_removed"],
    )

    # Validate
    errors = validate_cleaned_dataframe(cleaned_df)
    if errors:
        for err in errors:
            logger.error("Validation error: %s", err)
        sys.exit(1)

    # Save processed data
    Path(config.PROCESSED_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    logger.info("Saved cleaned data to %s", config.PROCESSED_DATA_PATH)

    return cleaned_df


def train_variant(
    model_name: str, variant_name: str, courses_df: pd.DataFrame
) -> tuple[ContentBasedRecommender, dict, str]:
    """Train a single recommender variant and log to MLflow.

    Returns:
        Tuple of (fitted recommender, metrics dict, model path)
    """
    logger.info("Training Variant %s (%s)...", variant_name, model_name)

    rec = ContentBasedRecommender(model_name=model_name)
    rec.fit(courses_df)

    # Save embeddings
    embeddings_path = os.path.join(
        config.EMBEDDINGS_DIR, f"embeddings_variant_{variant_name.lower()}.npy"
    )
    Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, rec.get_embeddings())
    logger.info("Saved embeddings to %s", embeddings_path)

    # Evaluate
    evaluator = RecommenderEvaluator(rec)
    metrics = evaluator.evaluate(sample_size=config.EVAL_SAMPLE_SIZE)

    # Save model
    model_path = os.path.join(config.MODELS_DIR, f"recommender_variant_{variant_name.lower()}.pkl")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    rec.save(model_path)

    # Log to MLflow
    run_id = log_model_training(variant_name, model_name, rec, metrics, model_path)

    logger.info("Variant %s complete — metrics: %s", variant_name, metrics)
    return rec, metrics, run_id


def main() -> None:
    """Run the full training pipeline."""
    logger.info("=" * 60)
    logger.info("  Course Recommender — Training Pipeline")
    logger.info("=" * 60)

    # Setup MLflow
    setup_mlflow()

    # Load and clean data
    courses_df = load_and_clean_data()

    # Train Variant A
    logger.info("Step 3/6: Training Variant A...")
    rec_a, metrics_a, run_id_a = train_variant(config.VARIANT_A_MODEL, "A", courses_df)

    # Train Variant B
    logger.info("Step 4/6: Training Variant B...")
    rec_b, metrics_b, run_id_b = train_variant(config.VARIANT_B_MODEL, "B", courses_df)

    # A/B comparison
    logger.info("Step 5/6: A/B comparison...")
    ab = ABTest(rec_a, rec_b)
    comparison = ab.compare_metrics(sample_size=config.EVAL_SAMPLE_SIZE)
    winner = ab.determine_winner(primary_metric="avg_similarity")

    # Save metadata
    logger.info("Step 6/6: Saving metadata...")
    metadata = {
        "variant_a": {
            "model_name": rec_a.model_name,
            "num_courses": rec_a.num_courses,
            "embedding_dim": rec_a.embedding_dim,
            "metrics": metrics_a,
            "mlflow_run_id": run_id_a,
        },
        "variant_b": {
            "model_name": rec_b.model_name,
            "num_courses": rec_b.num_courses,
            "embedding_dim": rec_b.embedding_dim,
            "metrics": metrics_b,
            "mlflow_run_id": run_id_b,
        },
        "winner": winner,
    }
    metadata_path = os.path.join(config.MODELS_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("\n%s", comparison.drop(columns=["key"], errors="ignore").to_string(index=False))
    logger.info("\n🏆 Winner: Variant %s", winner)
    logger.info("\nModels saved to: %s/", config.MODELS_DIR)
    logger.info("MLflow runs: variant_A=%s  variant_B=%s", run_id_a, run_id_b)
    logger.info("\nNext: run 'streamlit run app.py' to launch the UI")


if __name__ == "__main__":
    main()

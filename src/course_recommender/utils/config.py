"""Centralized configuration management using Pydantic settings."""

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration loaded from environment variables / .env file.

    All fields can be overridden via environment variables or a .env file
    in the project root.
    """

    # --- Data paths ---
    RAW_DATA_PATH: str = "data/raw/CourseraDataset-Unclean.csv"
    PROCESSED_DATA_PATH: str = "data/processed/courses_clean.csv"
    EMBEDDINGS_DIR: str = "data/processed"
    MODELS_DIR: str = "models"

    # --- Model settings ---
    VARIANT_A_MODEL: str = "all-MiniLM-L6-v2"
    VARIANT_B_MODEL: str = "all-mpnet-base-v2"
    TOP_K_RECOMMENDATIONS: int = 5
    SEARCH_TOP_K: int = 10

    # --- Evaluation ---
    EVAL_SAMPLE_SIZE: int = 100

    # --- MLflow ---
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "course-recommender-ab-test"

    # --- Streamlit ---
    APP_TITLE: str = "AI Course Recommender"
    APP_LAYOUT: str = "wide"

    # --- AWS ---
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton instance used throughout the application
config = Config()

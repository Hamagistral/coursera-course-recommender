"""Content-based course recommender using sentence transformers."""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from course_recommender.utils.logger import get_logger

logger = get_logger(__name__)


class ContentBasedRecommender:
    """Content-based course recommender powered by sentence-transformer embeddings.

    Encodes course text (name + description + skills + level) into dense
    vector embeddings and retrieves similar courses using cosine similarity.

    Args:
        model_name: HuggingFace sentence-transformers model identifier.
            - ``"all-MiniLM-L6-v2"`` — 384-dim, fast (Variant A)
            - ``"all-mpnet-base-v2"`` — 768-dim, more accurate (Variant B)

    Example:
        >>> rec = ContentBasedRecommender("all-MiniLM-L6-v2")
        >>> rec.fit(courses_df)
        >>> similar = rec.recommend_similar(course_id=42, top_k=5)
        >>> results = rec.search("machine learning for beginners", top_k=10)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._embeddings: np.ndarray | None = None
        self._courses_df: pd.DataFrame | None = None
        self._fit_time: float = 0.0

    # ------------------------------------------------------------------
    # Training / fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        courses_df: pd.DataFrame,
        text_column: str = "combined_text",
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> "ContentBasedRecommender":
        """Generate embeddings for all courses and store them.

        Args:
            courses_df: Cleaned DataFrame with at least a ``course_id`` and
                the specified ``text_column``.
            text_column: Name of the column containing text to embed.
            batch_size: Encoding batch size (tune for GPU/CPU memory).
            show_progress: Show tqdm progress bar during encoding.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If ``text_column`` is not in the DataFrame.
        """
        if text_column not in courses_df.columns:
            raise ValueError(
                f"Column '{text_column}' not found in DataFrame. "
                f"Available columns: {list(courses_df.columns)}"
            )

        logger.info("Loading sentence-transformer model: %s", self.model_name)
        # Import here so it's not loaded at module level (heavy import)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        self._courses_df = courses_df.reset_index(drop=True).copy()

        texts = self._courses_df[text_column].fillna("").astype(str).tolist()
        logger.info("Encoding %d courses...", len(texts))

        start = time.time()
        self._embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize → dot product = cosine sim
        )
        self._fit_time = time.time() - start

        logger.info(
            "Embeddings shape: %s  |  fit time: %.1fs",
            self._embeddings.shape,
            self._fit_time,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend_similar(
        self,
        course_id: int,
        top_k: int = 5,
        exclude_same_course: bool = True,
    ) -> pd.DataFrame:
        """Return the most similar courses for a given course.

        Args:
            course_id: The ``course_id`` value of the source course.
            top_k: Number of recommendations to return.
            exclude_same_course: Whether to exclude the source course itself.

        Returns:
            DataFrame with columns from the catalog plus ``similarity_score``,
            sorted by descending similarity.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If ``course_id`` is not found in the catalog.
        """
        self._assert_fitted()
        assert self._courses_df is not None
        assert self._embeddings is not None

        mask = self._courses_df["course_id"] == course_id
        if not mask.any():
            raise ValueError(f"course_id={course_id} not found in catalog.")

        idx = int(mask.idxmax())
        query_embedding = self._embeddings[idx : idx + 1]

        # Cosine similarities (embeddings are L2-normalized so this is dot product)
        sims = cosine_similarity(query_embedding, self._embeddings)[0]

        df = self._courses_df.copy()
        df["similarity_score"] = sims

        if exclude_same_course:
            df = df[df["course_id"] != course_id]

        return df.nlargest(top_k, "similarity_score").reset_index(drop=True)

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Search courses by a free-text query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            DataFrame with catalog columns plus ``similarity_score``,
            sorted by descending similarity.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        self._assert_fitted()
        assert self._model is not None
        assert self._courses_df is not None
        assert self._embeddings is not None

        query_embedding = self._model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        sims = cosine_similarity(query_embedding, self._embeddings)[0]

        df = self._courses_df.copy()
        df["similarity_score"] = sims
        return df.nlargest(top_k, "similarity_score").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def get_embeddings(self) -> np.ndarray:
        """Return the raw embeddings matrix.

        Returns:
            NumPy array of shape (n_courses, embedding_dim).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        self._assert_fitted()
        assert self._embeddings is not None
        return self._embeddings

    def save(self, filepath: str) -> None:
        """Serialize the fitted recommender to disk.

        Args:
            filepath: Destination path (e.g. ``models/recommender_a.pkl``).
        """
        self._assert_fitted()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "embeddings": self._embeddings,
                    "courses_df": self._courses_df,
                    "fit_time": self._fit_time,
                },
                f,
            )
        logger.info("Saved recommender to '%s'", filepath)

    @classmethod
    def load(cls, filepath: str) -> "ContentBasedRecommender":
        """Load a previously saved recommender from disk.

        Args:
            filepath: Path to the pickled file.

        Returns:
            Fitted ContentBasedRecommender instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: '{filepath}'")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        rec = cls(model_name=data["model_name"])
        rec._embeddings = data["embeddings"]
        rec._courses_df = data["courses_df"]
        rec._fit_time = data.get("fit_time", 0.0)

        # Re-load the sentence transformer (needed for search())
        from sentence_transformers import SentenceTransformer

        rec._model = SentenceTransformer(rec.model_name)
        logger.info(
            "Loaded recommender from '%s' (%d courses, %s)",
            filepath,
            len(rec._courses_df),
            rec.model_name,
        )
        return rec

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_courses(self) -> int:
        """Number of courses in the fitted catalog."""
        if self._courses_df is None:
            return 0
        return len(self._courses_df)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embeddings."""
        if self._embeddings is None:
            return 0
        return self._embeddings.shape[1]

    @property
    def courses_df(self) -> pd.DataFrame | None:
        """The fitted course catalog DataFrame."""
        return self._courses_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if self._embeddings is None or self._courses_df is None:
            raise RuntimeError("Recommender has not been fitted. Call fit() first.")

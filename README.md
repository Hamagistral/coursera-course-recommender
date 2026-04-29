# 🎓 Coursera Course Recommender

End-to-end ML engineering project — content-based recommendation system using BERT embeddings, MLflow experiment tracking, A/B testing, and an interactive Streamlit UI.

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🛠 Tech Stack

```
ML / NLP        sentence-transformers · scikit-learn · numpy · pandas
MLOps           MLflow (SQLite backend) · A/B testing framework
UI              Streamlit · Plotly
Packaging       uv · hatchling
Code Quality    Black · Ruff · isort · mypy
Testing         pytest · pytest-cov
CI/CD           GitHub Actions
```

- **Deep Learning / NLP** | BERT sentence embeddings via `sentence-transformers` for semantic similarity |
- **Recommender System** | Content-based filtering using cosine similarity on 6 000+ courses |
- **MLOps** | MLflow experiment tracking — params, metrics, artifacts logged per run |
- **A/B Testing** | Two embedding models compared (`all-MiniLM-L6-v2` vs `all-mpnet-base-v2`) |
- **Data Engineering** | Full EDA → cleaning → feature engineering pipeline on raw Coursera data |
- **Software Engineering** | Type hints, docstrings, modular architecture, 70%+ test coverage |
- **CI/CD** | GitHub Actions — automated tests + code quality checks on every push |
- **UI** | Interactive Streamlit app — search, similar courses, A/B test dashboard |


---

## Quick Start

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — `pip install uv`
- The raw dataset in `data/raw/CourseraDataset-Unclean.csv`  
  → Download from [Kaggle](https://www.kaggle.com/datasets/elvinrustam/coursera-dataset)

### 1. Clone & install

```bash
git clone https://github.com/Hamagistral/coursera-course-recommender
cd course-recommender

uv venv --python 3.11
# Activate:
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

uv pip install -e ".[dev,notebooks]"
```

### 2. Configure environment

```bash
cp .env.example .env
# No mandatory variables for local use — Kaggle credentials optional
```

### 3. Place the dataset

Download `CourseraDataset-Unclean.csv` from Kaggle and place it at: 

```
data/raw/CourseraDataset-Unclean.csv
```

- Dataset: [Coursera Dataset — Kaggle](https://www.kaggle.com/datasets/elvinrustam/coursera-dataset) by Elvin Rustam

### 4. Train both models

```bash
uv run python scripts/train_models.py
```

This single command:
1. Cleans the raw dataset → `data/processed/courses_clean.csv`
2. Generates embeddings for Variant A (`all-MiniLM-L6-v2`, 384-dim)
3. Generates embeddings for Variant B (`all-mpnet-base-v2`, 768-dim)
4. Evaluates both variants (diversity, coverage, avg similarity)
5. Logs all runs and metrics to MLflow
6. Saves models to `models/` and A/B results to `models/metadata.json`

### 5. Launch the Streamlit app

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 6. (Optional) Explore with Jupyter notebooks

```bash
uv run jupyter notebook notebooks/
```

Run in order:
- `01_exploratory_data_analysis.ipynb` — dataset profiling and visualisations
- `02_data_cleaning_transformation.ipynb` — cleaning pipeline with before/after report
- `03_modeling_evaluation.ipynb` — training, evaluation, A/B comparison

### 7. (Optional) View MLflow experiment dashboard

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://localhost:5000](http://localhost:5000) → experiment `course-recommender-ab-test`

---

## 📁 Project Structure

```
course-recommender/
│
├── src/course_recommender/
│   ├── data/
│   │   ├── loader.py          # DataLoader — loads raw CSV with encoding handling
│   │   ├── cleaner.py         # DataCleaner — builder pattern, method chaining
│   │   └── validator.py       # validate_cleaned_dataframe()
│   │
│   ├── models/
│   │   ├── recommender.py     # ContentBasedRecommender — fit / recommend / search
│   │   └── evaluator.py       # RecommenderEvaluator — diversity, coverage, similarity
│   │
│   ├── mlops/
│   │   ├── mlflow_utils.py    # setup_mlflow, log_model_training, get_best_run
│   │   └── ab_testing.py      # ABTest — compare_metrics, determine_winner
│   │
│   └── utils/
│       ├── config.py          # Pydantic Settings — all paths and hyperparams
│       └── logger.py          # Structured logging setup
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_cleaning_transformation.ipynb
│   └── 03_modeling_evaluation.ipynb
│
├── tests/
│   ├── test_data_cleaner.py
│   ├── test_recommender.py
│   └── test_evaluator.py
│
├── scripts/
│   └── train_models.py        # Full training pipeline (clean → embed → eval → log)
│
├── data/
│   ├── raw/                   # CourseraDataset-Unclean.csv (not committed)
│   └── processed/             # courses_clean.csv, embeddings_*.npy
│
├── models/                    # recommender_variant_{a,b}.pkl, metadata.json
│
├── .github/workflows/
│   ├── ci.yml                 # pytest + coverage
│   └── quality.yml            # Black, Ruff, isort, mypy
│
├── app.py                     # Streamlit application
├── pyproject.toml             # uv / hatchling project config + tool settings
└── requirements.txt           # Pinned dependencies (fallback)
```

---

## 📊 Model Performance (A/B Test Results)

Evaluated on a random sample of 100 courses, top-5 recommendations each.

| Metric | Variant A — `all-MiniLM-L6-v2` | Variant B — `all-mpnet-base-v2` | Winner |
|---|---|---|---|
| **Avg Similarity** | 0.639 | 0.691 | ✅ B (+8%) |
| **Coverage** | 7.7% | 7.8% | ≈ tie |
| **Diversity** | 0.020 | 0.020 | ≈ tie |
| **Inference Time** | ~9 ms | ~21 ms | ✅ A (2.3× faster) |

**Conclusion:** Variant B wins on semantic quality (+8% avg similarity). Variant A is preferred for latency-sensitive applications. Diversity is inherently low — by design, content-based systems cluster tightly within subject areas.

---

## 🧪 Development

### Run tests

```bash
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html
```

### Code quality checks

```bash
uv run black src/ tests/ app.py          # format
uv run isort src/ tests/ app.py          # sort imports
uv run ruff check src/ tests/ app.py     # lint
uv run mypy src/                         # type check
```

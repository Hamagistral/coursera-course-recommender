"""Streamlit application for the AI Course Recommender.

Run with:
    streamlit run app.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Course Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Course card */
.course-card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: box-shadow 0.2s;
}
.course-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }

.course-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1a1a2e;
    margin: 0 0 4px 0;
    line-height: 1.3;
}
.course-uni {
    font-size: 0.82rem;
    color: #666;
    margin-bottom: 8px;
}
.course-desc {
    font-size: 0.88rem;
    color: #444;
    line-height: 1.5;
    margin-bottom: 10px;
}
.course-skills {
    font-size: 0.82rem;
    color: #555;
    margin-bottom: 10px;
}


/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 18px;
    color: white;
    text-align: center;
}
.kpi-value { font-size: 2rem; font-weight: 700; margin: 0; }
.kpi-label { font-size: 0.85rem; opacity: 0.85; margin: 0; }

/* Winner banner */
.winner-banner {
    background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    border-radius: 12px;
    padding: 16px 24px;
    text-align: center;
    font-size: 1.1rem;
    font-weight: 700;
    color: #333;
    margin-bottom: 16px;
}

/* Section header */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 4px;
}
.section-sub {
    color: #888;
    font-size: 0.9rem;
    margin-bottom: 20px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
DATA_PATH = "data/processed/courses_clean.csv"
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

MODEL_INFO = {
    "A": {
        "model_name": "all-MiniLM-L6-v2",
        "label": "all-MiniLM-L6-v2  (Faster)",
        "specs": [
            "- **Architecture:** MiniLM (distilled BERT)",
            "- **Embedding dim:** 384",
            "- **Speed:** ~5× faster than MPNet",
            "- **Model size:** ~14 MB"
        ],
        "model_path": os.path.join(MODELS_DIR, "recommender_variant_a.pkl"),
    },
    "B": {
        "model_name": "all-mpnet-base-v2",
        "label": "all-mpnet-base-v2  (More Accurate)",
        "specs": [
            "- **Architecture:** MPNet (full-size transformer)",
            "- **Embedding dim:** 768",
            "- **Accuracy:** +8% avg similarity vs MiniLM",
            "- **Model size:** ~24 MB"
        ],
        "model_path": os.path.join(MODELS_DIR, "recommender_variant_b.pkl"),
    },
}


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading recommender model…")
def load_recommender(variant: str):
    import sys
    sys.path.insert(0, "src")
    from course_recommender.models.recommender import ContentBasedRecommender
    model_path = MODEL_INFO[variant]["model_path"]
    if not Path(model_path).exists():
        return None
    return ContentBasedRecommender.load(model_path)


@st.cache_data(show_spinner=False)
def load_courses() -> pd.DataFrame | None:
    if not Path(DATA_PATH).exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if not Path(METADATA_PATH).exists():
        return {}
    with open(METADATA_PATH) as f:
        return json.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None



def _course_card(row: pd.Series, df: pd.DataFrame, score: float | None = None) -> None:
    name_col   = _get_col(df, ["Course Title", "Course Name", "Title"])
    uni_col    = _get_col(df, ["Offered By", "University", "Organization"])
    rating_col = _get_col(df, ["Rating", "Course Rating"])
    url_col    = _get_col(df, ["Course Url", "Course URL", "URL"])
    level_col  = _get_col(df, ["Level", "Difficulty Level"])
    skills_col = _get_col(df, ["Skill gain", "Skills", "skills"])
    desc_col   = _get_col(df, ["What you will learn", "Course Description", "Description"])

    title  = str(row[name_col])   if name_col   else "Untitled"
    uni    = str(row[uni_col])    if uni_col    else ""
    url    = str(row[url_col])    if url_col    else ""
    level  = str(row[level_col])  if level_col  else ""
    skills = str(row[skills_col]) if skills_col else ""
    desc   = str(row[desc_col])   if desc_col   else ""

    if desc.lower() in ("not specified", "nan", ""):
        desc = ""
    if len(desc) > 280:
        desc = desc[:280] + "…"

    try:
        rating_val = float(row[rating_col]) if rating_col else 0.0
        rating_str = f"⭐ {rating_val:.1f}" if rating_val > 0 else ""
    except (ValueError, TypeError):
        rating_str = ""

    level_icon = (
        "🟢" if "begin" in level.lower()
        else "🟡" if "inter" in level.lower()
        else "🔴" if "advan" in level.lower()
        else "🔵" if "mix"   in level.lower()
        else "⚪"
    )

    skill_tags = (
        [s.strip() for s in skills.split(",") if s.strip()][:5]
        if skills and skills.lower() not in ("not specified", "nan") else []
    )

    with st.container(border=True):
        # Title + score row
        left, right = st.columns([5, 1])
        with left:
            title_md = f"[**{title}**]({url})" if url.startswith("http") else f"**{title}**"
            st.markdown(title_md)
        with right:
            if score is not None:
                pct = score * 100
                color = "green" if pct >= 65 else "orange" if pct >= 50 else "red"
                st.markdown(
                    f"<div style='text-align:right'>"
                    f"<span style='color:{color};font-weight:700;font-size:0.9rem;'>{pct:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Meta row: university / level / rating
        meta_parts = []
        if uni:
            meta_parts.append(f"🏫 {uni}")
        if level:
            meta_parts.append(f"{level_icon} {level}")
        if rating_str:
            meta_parts.append(rating_str)
        if meta_parts:
            st.caption("  ·  ".join(meta_parts))

        # Description
        if desc:
            st.markdown(f"<small style='color:#555;'>{desc}</small>", unsafe_allow_html=True)

        # Skill tags + link
        bottom_left, bottom_right = st.columns([4, 1])
        with bottom_left:
            if skill_tags:
                tags_html = " ".join(
                    f'<span style="background:#f0f2f6;border-radius:4px;padding:1px 8px;'
                    f'font-size:0.76rem;margin-right:4px;">{t}</span>'
                    for t in skill_tags
                )
                st.markdown(tags_html, unsafe_allow_html=True)
        with bottom_right:
            if url.startswith("http"):
                st.markdown(
                    f"<div style='text-align:right'><a href='{url}' target='_blank' "
                    f"style='font-size:0.8rem;'>Open ↗</a></div>",
                    unsafe_allow_html=True,
                )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 Coursera Course Recommender")

    variant = st.radio(
        "Embedding Model",
        options=["A", "B"],
        format_func=lambda v: MODEL_INFO[v]["label"],
        help="Select which sentence-transformer model powers the recommendations",
    )
    with st.container():
        for spec in MODEL_INFO[variant]["specs"]:
            st.markdown(spec)
    st.divider()

    st.markdown("### About")
    st.markdown(
        "An **end-to-end ML engineering project** showcasing production-ready skills:\n\n"
        "- 📊 **Data exploration & cleaning** — EDA on 8 000+ raw Coursera courses, "
        "handling missing values, duplicates and inconsistent formatting\n"
        "- 🤖 **Recommender system** — content-based recommendations using "
        "BERT sentence embeddings and cosine similarity\n"
        "- 🔬 **MLflow experiment tracking** — two embedding models compared "
        "as an A/B test with logged params, metrics and artifacts\n"
        "- ⚙️ **CI/CD** — automated quality checks & test suite via GitHub Actions\n"
        "- 🖥 **Interactive UI** — this Streamlit app for live exploration"
    )
    st.divider()
    st.caption("Built with ☕ by **Hamza El Belghiti**")


# ── Data & model loading ──────────────────────────────────────────────────────
courses_df = load_courses()
metadata   = load_metadata()

if courses_df is None:
    st.error(
        "⚠️ Cleaned dataset not found at `data/processed/courses_clean.csv`.  \n"
        "Run: `uv run python scripts/train_models.py`"
    )
    st.stop()

recommender = load_recommender(variant)
if recommender is None:
    st.warning(
        f"⚠️ Model for Variant {variant} not found. "
        "Run `uv run python scripts/train_models.py` to train both variants."
    )

# Shared column lookups
_name_col    = _get_col(courses_df, ["Course Title", "Course Name", "Title"])
_keyword_col = _get_col(courses_df, ["Keyword", "keyword", "Category"])
_level_col   = _get_col(courses_df, ["Level", "Difficulty Level"])
_rating_col  = _get_col(courses_df, ["Rating", "Course Rating"])

tab1, tab2, tab3 = st.tabs(["🔍 Search Courses", "💡 Similar Courses", "📊 A/B Test Results"])


# ── Tab 1: Search ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Search Courses by Topic</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Semantic search — finds relevant courses even when the exact words don\'t match.</p>', unsafe_allow_html=True)

    col_input, col_slider = st.columns([3, 1])
    with col_input:
        query = st.text_input(
            "Search query",
            placeholder="e.g.  deep learning for NLP  ·  data visualization with Python  ·  cloud architecture",
            label_visibility="collapsed",
        )
    with col_slider:
        top_k = st.slider("Results", min_value=3, max_value=20, value=10, label_visibility="collapsed")

    search_clicked = st.button("🔍  Search", type="primary", use_container_width=False)

    if (search_clicked or query) and query:
        if recommender is None:
            st.error("Model not loaded — run the training pipeline first.")
        else:
            with st.spinner("Searching…"):
                results = recommender.search(query, top_k=top_k)
            st.markdown(f"**{len(results)} results** for *{query}*")
            st.markdown("")
            for _, row in results.iterrows():
                _course_card(row, results, score=row.get("similarity_score"))


# ── Tab 2: Similar Courses ────────────────────────────────────────────────────
with tab2:
    # ── KPIs ──────────────────────────────────────────────────────────────────
    avg_rating = (
        courses_df[_rating_col].replace(0, float("nan")).mean()
        if _rating_col else 0.0
    )
    n_cats   = courses_df[_keyword_col].nunique() if _keyword_col else 0
    n_levels = courses_df[_level_col].nunique()   if _level_col   else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            '<div class="kpi-card"><p class="kpi-value">6 089</p><p class="kpi-label">Total Courses</p></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{n_cats}</p><p class="kpi-label">Subject Categories</p></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{n_levels}</p><p class="kpi-label">Difficulty Levels</p></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">⭐ {avg_rating:.2f}</p><p class="kpi-label">Avg Rating</p></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Course selector ───────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Find Similar Courses</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Pick any course — the model finds the most semantically similar ones.</p>', unsafe_allow_html=True)

    if _name_col:
        course_names = courses_df[_name_col].fillna("Unnamed").tolist()

        col_select, col_k = st.columns([4, 1])
        with col_select:
            selected_idx = st.selectbox(
                "Select a course",
                options=range(len(course_names)),
                format_func=lambda i: course_names[i],
                label_visibility="collapsed",
            )
        with col_k:
            top_k_sim = st.slider("# recs", min_value=3, max_value=10, value=5, key="sim_k", label_visibility="collapsed")

        selected_row = courses_df.iloc[selected_idx]
        selected_id  = int(selected_row["course_id"])

        st.markdown("**Selected course:**")
        _course_card(selected_row, courses_df)

        if st.button("✨  Get Recommendations", type="primary"):
            if recommender is None:
                st.error("Model not loaded — run the training pipeline first.")
            else:
                with st.spinner("Finding similar courses…"):
                    recs = recommender.recommend_similar(selected_id, top_k=top_k_sim)
                st.markdown(f"**Top {top_k_sim} similar courses:**")
                for _, row in recs.iterrows():
                    _course_card(row, recs, score=row.get("similarity_score"))
    else:
        st.warning("Course Title column not found in dataset.")


# ── Tab 3: A/B Test ───────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">A/B Test: Embedding Model Comparison</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Two sentence-transformer models were trained on the same catalog '
        'and evaluated on 100 randomly sampled courses.',
        unsafe_allow_html=True,
    )

    if not metadata:
        st.info("No A/B test results found. Run `uv run python scripts/train_models.py`.")
    else:
        meta_a  = metadata.get("variant_a", {})
        meta_b  = metadata.get("variant_b", {})
        winner  = metadata.get("winner", "?")
        winner_name = meta_a.get("model_name") if winner == "A" else meta_b.get("model_name")

        metrics_a = meta_a.get("metrics", {})
        metrics_b = meta_b.get("metrics", {})

        # Model overview cards
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class="course-card" style="border-left: 4px solid #636EFA;">
                    <b style="font-size:1rem;">{meta_a.get("model_name","—")}</b><br>
                    <span style="color:#888;font-size:0.85rem;">Variant A · {meta_a.get("embedding_dim", meta_a.get("metrics",{}).get("embedding_dim","384"))}-dim embeddings · {meta_a.get("num_courses",0):,} courses</span><br><br>
                    <b>Avg Similarity:</b> {metrics_a.get("avg_similarity", 0):.4f}<br>
                    <b>Coverage:</b> {metrics_a.get("coverage", 0):.4f}<br>
                    <b>Diversity:</b> {metrics_a.get("diversity", 0):.4f}<br>
                    <b>Inference:</b> {metrics_a.get("inference_time_ms", 0):.1f} ms
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="course-card" style="border-left: 4px solid #EF553B;">
                    <b style="font-size:1rem;">{meta_b.get("model_name","—")}</b><br>
                    <span style="color:#888;font-size:0.85rem;">Variant B · {meta_b.get("embedding_dim", meta_b.get("metrics",{}).get("embedding_dim","768"))}-dim embeddings · {meta_b.get("num_courses",0):,} courses</span><br><br>
                    <b>Avg Similarity:</b> {metrics_b.get("avg_similarity", 0):.4f}<br>
                    <b>Coverage:</b> {metrics_b.get("coverage", 0):.4f}<br>
                    <b>Diversity:</b> {metrics_b.get("diversity", 0):.4f}<br>
                    <b>Inference:</b> {metrics_b.get("inference_time_ms", 0):.1f} ms
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Grouped bar chart
        METRIC_LABELS = {
            "diversity": "Diversity",
            "coverage": "Coverage",
            "avg_similarity": "Avg Similarity",
        }
        chart_keys   = list(METRIC_LABELS.keys())
        chart_labels = list(METRIC_LABELS.values())

        fig = go.Figure(data=[
            go.Bar(
                name=meta_a.get("model_name", "Variant A"),
                x=chart_labels,
                y=[metrics_a.get(k, 0) for k in chart_keys],
                marker_color="#636EFA",
                text=[f"{metrics_a.get(k,0):.3f}" for k in chart_keys],
                textposition="outside",
            ),
            go.Bar(
                name=meta_b.get("model_name", "Variant B"),
                x=chart_labels,
                y=[metrics_b.get(k, 0) for k in chart_keys],
                marker_color="#EF553B",
                text=[f"{metrics_b.get(k,0):.3f}" for k in chart_keys],
                textposition="outside",
            ),
        ])
        fig.update_layout(
            barmode="group",
            title="Quality Metrics Comparison",
            yaxis_title="Score",
            template="plotly_white",
            legend_title="Model",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Inference time separately as a simple metric row
        st.markdown("**Inference time (ms per request):**")
        t1, t2, _ = st.columns([1, 1, 2])
        with t1:
            st.metric(meta_a.get("model_name","A"), f"{metrics_a.get('inference_time_ms',0):.1f} ms")
        with t2:
            st.metric(
                meta_b.get("model_name","B"),
                f"{metrics_b.get('inference_time_ms',0):.1f} ms",
                delta=f"{metrics_b.get('inference_time_ms',0)-metrics_a.get('inference_time_ms',0):+.1f} ms",
                delta_color="inverse",
            )

        st.divider()

        st.subheader("Metric Definitions")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Avg Similarity** — mean cosine similarity between a seed course and its recommendations. Higher = semantically tighter matches.")
            st.markdown("**Coverage** — fraction of the 6 089-course catalog that appears in at least one recommendation list.")
        with d2:
            st.markdown("**Diversity** — fraction of unique subject categories across all recommendation lists. Content-based systems naturally cluster by topic, so this is inherently low.")
            st.markdown("**Inference Time** — average ms per recommendation request on CPU. Variant A is ~1.7× faster.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.82em;padding:8px 0;'>"
    "Built with ☕ by <strong>Hamza El Belghiti</strong> &nbsp;·&nbsp; "
    "Data: <a href='https://www.kaggle.com/datasets/elvinrustam/coursera-dataset' style='color:#aaa;'>Coursera Dataset (Kaggle)</a>"
    "</div>",
    unsafe_allow_html=True,
)

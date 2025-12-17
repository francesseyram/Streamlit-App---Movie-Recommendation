"""
PHASE 2: STREAMLIT ANALYTICS DASHBOARD
====================================
Movie recommendation analytics with Cinematic Purple theme.

- Safe for Streamlit Cloud
- Downloads data once from Google Drive
- No UI calls inside cached functions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
import gdown

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="MovieLens Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM THEME
# ============================================================================
st.markdown(
    """
    <style>
    .stApp { background-color: #0D0221; color: #FFFFFF; }
    [data-testid="stSidebar"] {
        background-color: #1A0033;
        border-right: 2px solid #7B2CBF;
    }
    h1, h2, h3 { color: #E0AAFF; }
    [data-testid="metric-container"] {
        background-color: #1A0033;
        border: 1px solid #7B2CBF;
        border-radius: 8px;
        padding: 15px;
    }
    .stButton > button {
        background-color: #7B2CBF;
        color: #FFFFFF;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton > button:hover { background-color: #9D4EDD; }
    [data-testid="stExpander"] {
        border: 1px solid #7B2CBF;
        border-radius: 8px;
    }
    hr { border-color: #7B2CBF; }
    p { color: #D0D0D0; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================================
# DATA DOWNLOAD (SAFE FOR STREAMLIT CLOUD)
# ============================================================================
DATA_FILE = "ratings_sample_cleaned.parquet"
FILE_ID = "1S4Lklg1e1LpbjSnfmUdSxD3jhE0kGlBN"


def ensure_data_downloaded():
    """Download dataset once if missing (UI allowed here)."""
    if not os.path.exists(DATA_FILE):
        st.info("First load: Downloading dataset from Google Drive (≈30–60 seconds)...")
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={FILE_ID}",
                DATA_FILE,
                quiet=False
            )
            st.success("Dataset downloaded successfully!")
        except Exception as e:
            st.error("Failed to download dataset.")
            st.exception(e)
            st.stop()


@st.cache_data(show_spinner=False)
def load_data():
    """Pure cached data loader (NO Streamlit calls)."""
    df = pd.read_parquet(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["genres_list"] = df["genres"].str.split("|")
    return df


# ============================================================================
# APP STARTUP
# ============================================================================
ensure_data_downloaded()
df = load_data()

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("MovieLens Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Dashboard Pages",
    ["Overview", "User Behavior", "Content Analysis", "Hidden Gems"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

year_range = st.sidebar.slider(
    "Rating Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)

rating_threshold = st.sidebar.slider(
    "Minimum Rating", 0.5, 5.0, 2.0, step=0.5
)

available_genres = sorted(
    {g for sub in df["genres_list"] if isinstance(sub, list) for g in sub}
)

selected_genres = st.sidebar.multiselect(
    "Filter by Genre",
    available_genres
)

# ============================================================================
# FILTER FUNCTION
# ============================================================================
def apply_filters(data):
    filtered = data[
        (data["year"] >= year_range[0]) &
        (data["year"] <= year_range[1]) &
        (data["rating"] >= rating_threshold)
    ]

    if selected_genres:
        filtered = filtered[
            filtered["genres_list"].apply(
                lambda x: isinstance(x, list) and any(g in x for g in selected_genres)
            )
        ]

    return filtered


filtered_df = apply_filters(df)

st.sidebar.markdown("---")
st.sidebar.metric("Total Ratings", f"{len(filtered_df):,}")
st.sidebar.metric("Unique Users", f"{filtered_df['userId'].nunique():,}")
st.sidebar.metric("Unique Movies", f"{filtered_df['movieId'].nunique():,}")

# ============================================================================
# OVERVIEW PAGE
# ============================================================================
if page == "Overview":
    st.title("Overview")
    st.markdown("High-level metrics and trends")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Rating", f"{filtered_df['rating'].mean():.2f}")
    col2.metric("Median Rating", f"{filtered_df['rating'].median():.1f}")
    col3.metric("Std Deviation", f"{filtered_df['rating'].std():.2f}")
    col4.metric("Data Points", f"{len(filtered_df):,}")

    st.markdown("---")

    rating_counts = filtered_df["rating"].value_counts().sort_index()
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={"x": "Rating", "y": "Count"},
        title="Rating Distribution",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# USER BEHAVIOR
# ============================================================================
elif page == "User Behavior":
    st.title("User Behavior Analysis")

    user_stats = filtered_df.groupby("userId")["rating"].agg(
        ["count", "mean", "std"]
    ).reset_index()

    fig = px.histogram(
        user_stats,
        x="mean",
        nbins=40,
        title="User Rating Tendencies",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# CONTENT ANALYSIS
# ============================================================================
elif page == "Content Analysis":
    st.title("Content Analysis")

    movie_stats = filtered_df.groupby("movieId").agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean"),
        title=("title", "first"),
        release_year=("release_year", "first")
    ).reset_index()

    fig = px.scatter(
        movie_stats,
        x="num_ratings",
        y="avg_rating",
        hover_data=["title"],
        title="Popularity vs Rating",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# HIDDEN GEMS
# ============================================================================
elif page == "Hidden Gems":
    st.title("Hidden Gems Finder")

    movie_stats = filtered_df.groupby("movieId").agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean"),
        title=("title", "first"),
        release_year=("release_year", "first")
    ).reset_index()

    min_rating = st.slider("Minimum Rating", 3.0, 5.0, 4.0, 0.1)
    max_visibility = st.slider("Maximum Ratings", 1, 100, 20)

    gems = movie_stats[
        (movie_stats["avg_rating"] >= min_rating) &
        (movie_stats["num_ratings"] <= max_visibility)
    ].sort_values("avg_rating", ascending=False)

    st.metric("Hidden Gems Found", len(gems))
    st.dataframe(gems.head(20), use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "MovieLens Analytics Dashboard • Streamlit Cloud Ready • Phase 2 Complete"
)
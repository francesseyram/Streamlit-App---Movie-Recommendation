import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------------------
# Helper function to safely load data
# ------------------------------
@st.cache_data
def load_data(parquet_file):
    try:
        df = pd.read_parquet(parquet_file)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()  # Return empty DataFrame if failed

# ------------------------------
# Main App
# ------------------------------
def main():
    st.title("🎬 Movie Recommendation App")

    # Load dataset
    df = load_data("ratings_sample_cleaned.parquet")

    if df.empty:
        st.warning("Dataset is empty or failed to load. Please check your Parquet file.")
        return

    # Check if release_year column exists
    if "release_year" not in df.columns:
        st.error("The dataset does not contain a 'release_year' column.")
        st.write("Available columns:", df.columns.tolist())
        return

    # Limit dataset size for plotting
    df_plot = df.head(5000)

    # Sidebar filters
    st.sidebar.header("Filters")
    min_year = int(df['release_year'].min())
    max_year = int(df['release_year'].max())
    year_range = st.sidebar.slider("Release year range", min_year, max_year, (min_year, max_year))

    filtered_df = df_plot[(df_plot['release_year'] >= year_range[0]) & (df_plot['release_year'] <= year_range[1])]

    # Show basic stats
    st.subheader("Dataset Overview")
    st.write(filtered_df.describe())
    st.write("Total movies:", filtered_df['movieId'].nunique())

    # Plot: Number of movies per year
    st.subheader("Movies Released per Year")
    movies_per_year = filtered_df.groupby('release_year')['movieId'].nunique().reset_index()
    fig = px.bar(movies_per_year, x='release_year', y='movieId', labels={'movieId': 'Number of Movies', 'release_year': 'Year'})
    st.plotly_chart(fig)

    # Example Recommendation Section
    st.subheader("Example: Top Rated Movies")
    if 'rating' in df.columns:
        top_movies = df.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(10).reset_index()
        st.write(top_movies)
    else:
        st.info("No 'rating' column found in dataset to show top rated movies.")

# ------------------------------
# Run the app safely
# ------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Runtime error: {e}")
        raise e
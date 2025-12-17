import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="MovieLens Analytics", layout="wide")

st.markdown("""<style>
.stApp { background-color: #0D0221; color: #FFFFFF; }
[data-testid="stSidebar"] { background-color: #1A0033; border-right: 2px solid #7B2CBF; }
h1, h2, h3 { color: #E0AAFF; }
[data-testid="metric-container"] { background-color: #1A0033; border: 1px solid #7B2CBF; }
.stButton > button { background-color: #7B2CBF; color: #FFFFFF; }
</style>""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load():
    import gdown
    if not os.path.exists('ratings_sample_cleaned.parquet'):
        gdown.download("https://drive.google.com/uc?id=1S4Lklg1e1LpbjSnfmUdSxD3jhE0kGlBN", 
                      'ratings_sample_cleaned.parquet', quiet=False)
    df = pd.read_parquet('ratings_sample_cleaned.parquet')
    df['genres_list'] = df['genres'].fillna('').str.split('|')
    return df

df = load()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("MovieLens Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio("Dashboard", ["Overview", "User Behavior", "Content Analysis", "Hidden Gems"])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

year_range = st.sidebar.slider("Year Range", int(df['year'].min()), int(df['year'].max()), 
                               (int(df['year'].min()), int(df['year'].max())))

rating_threshold = st.sidebar.slider("Min Rating", 0.5, 5.0, 2.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Summary")
st.sidebar.metric("Total Ratings", f"{len(df):,}")
st.sidebar.metric("Users", f"{df['userId'].nunique():,}")
st.sidebar.metric("Movies", f"{df['movieId'].nunique():,}")

# ============================================================================
# APPLY FILTERS
# ============================================================================

filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1]) & (df['rating'] >= rating_threshold)].copy()

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "Overview":
    st.title("Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f}")
    with col2:
        st.metric("Median Rating", f"{filtered_df['rating'].median():.1f}")
    with col3:
        st.metric("Std Dev", f"{filtered_df['rating'].std():.2f}")
    with col4:
        st.metric("Data Points", f"{len(filtered_df):,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Distribution")
        rating_counts = filtered_df['rating'].value_counts().sort_index()
        fig = px.bar(x=rating_counts.index, y=rating_counts.values, 
                    labels={'x': 'Rating', 'y': 'Count'})
        fig.update_traces(marker_color='#7B2CBF')
        fig.update_layout(height=400, template='plotly_dark', paper_bgcolor='#0D0221',
                         plot_bgcolor='#1A0033', font=dict(color='#FFFFFF'), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Trends Over Time")
        ratings_by_year = filtered_df.groupby('year')['rating'].agg(['mean', 'count']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ratings_by_year['year'], y=ratings_by_year['mean'],
                                mode='lines+markers', name='Avg Rating',
                                line=dict(color='#E0AAFF', width=3),
                                marker=dict(size=8, color='#7B2CBF')))
        fig.update_layout(title="Are ratings increasing or decreasing?", xaxis_title="Year",
                         yaxis_title="Average Rating", height=400, template='plotly_dark',
                         paper_bgcolor='#0D0221', plot_bgcolor='#1A0033', font=dict(color='#FFFFFF'))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Genre Performance")
    
    all_genres = []
    genre_ratings = []
    for genres_list, rating in zip(filtered_df['genres_list'], filtered_df['rating']):
        if isinstance(genres_list, list):
            for g in genres_list:
                if g:
                    all_genres.append(g)
                    genre_ratings.append(rating)
    
    if all_genres:
        genre_df = pd.DataFrame({'genre': all_genres, 'rating': genre_ratings})
        genre_stats = genre_df.groupby('genre')['rating'].agg(['mean', 'count']).round(2).sort_values('mean', ascending=False)
        
        fig = px.bar(x=genre_stats.index, y=genre_stats['mean'],
                    labels={'x': 'Genre', 'y': 'Avg Rating'})
        fig.update_traces(marker=dict(color=genre_stats['mean'], colorscale=[[0, '#7B2CBF'], [1, '#E0AAFF']], showscale=True))
        fig.update_layout(height=400, template='plotly_dark', paper_bgcolor='#0D0221',
                         plot_bgcolor='#1A0033', font=dict(color='#FFFFFF'))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: USER BEHAVIOR
# ============================================================================

elif page == "User Behavior":
    st.title("User Behavior Analysis")
    
    user_stats = filtered_df.groupby('userId')['rating'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    user_stats.columns = ['userId', 'num_ratings', 'avg_rating', 'rating_std', 'min_rating', 'max_rating']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        harsh = (user_stats['avg_rating'] < 3).sum()
        st.metric("Harsh Raters", harsh, f"{harsh/len(user_stats)*100:.1f}%")
    with col2:
        moderate = ((user_stats['avg_rating'] >= 3) & (user_stats['avg_rating'] < 3.5)).sum()
        st.metric("Moderate Raters", moderate, f"{moderate/len(user_stats)*100:.1f}%")
    with col3:
        generous = (user_stats['avg_rating'] >= 3.5).sum()
        st.metric("Generous Raters", generous, f"{generous/len(user_stats)*100:.1f}%")
    with col4:
        active = (user_stats['num_ratings'] > 50).sum()
        st.metric("Highly Active", active, f"{active/len(user_stats)*100:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Behavior Distribution")
        fig = px.histogram(user_stats, x='avg_rating', nbins=50,
                          labels={'avg_rating': 'Avg Rating per User', 'count': 'Count'})
        fig.update_traces(marker_color='#7B2CBF')
        fig.update_layout(height=400, template='plotly_dark', paper_bgcolor='#0D0221',
                         plot_bgcolor='#1A0033', font=dict(color='#FFFFFF'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Activity Distribution")
        fig = px.histogram(user_stats, x='num_ratings', nbins=50,
                          labels={'num_ratings': 'Ratings per User', 'count': 'Count'})
        fig.update_traces(marker_color='#9D4EDD')
        fig.update_layout(height=400, template='plotly_dark', paper_bgcolor='#0D0221',
                         plot_bgcolor='#1A0033', font=dict(color='#FFFFFF'), yaxis_type='log')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Activity vs Consistency")
    fig = px.scatter(user_stats, x='num_ratings', y='rating_std', color='avg_rating',
                    labels={'num_ratings': 'Number of Ratings', 'rating_std': 'Std Dev'},
                    color_continuous_scale=['#7B2CBF', '#E0AAFF'])
    fig.update_layout(height=450, template='plotly_dark', paper_bgcolor='#0D0221',
                     plot_bgcolor='#1A0033', font=dict(color='#FFFFFF'))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: CONTENT ANALYSIS
# ============================================================================

elif page == "Content Analysis":
    st.title("Content Analysis")
    
    movie_stats = filtered_df.groupby('movieId')['rating'].agg(['count', 'mean']).reset_index()
    movie_stats.columns = ['movieId', 'num_ratings', 'avg_rating']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", movie_stats['movieId'].nunique())
    with col2:
        st.metric("Avg Ratings/Movie", f"{movie_stats['num_ratings'].mean():.0f}")
    with col3:
        st.metric("Median Ratings/Movie", f"{movie_stats['num_ratings'].median():.0f}")
    with col4:
        st.metric("Most Rated", movie_stats['num_ratings'].max())
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Release Year Impact")
        year_stats = filtered_df.dropna(subset=['release_year']).groupby('release_year')['rating'].agg(['mean', 'count']).reset_index()
        year_stats = year_stats[year_stats['count'] > 5]
        fig = px.scatter(year_stats, x='release_year', y='mean', size='count',
                        labels={'release_year': 'Release Year', 'mean': 'Avg Rating'},
                        color='mean', color_continuous_scale=['#7B2CBF', '#E0AAFF'])
        fig.update_layout(height=400, template='plotly_dark', paper_bgcolor='#0D0221',
                         plot_bgcolor='#1A0033', font=dict(color='#FFFFFF'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Popularity Distribution")
        fig = px.histogram(movie_stats, x='num_ratings', nbins=50,
                          labels={'num_ratings': 'Ratings per Movie', 'count': 'Count'})
        fig.update_traces(marker_color='#9D4EDD')
        fig.update_xaxes(type='log')
        fig.update_layout(height=400, template='plotly_dark', paper_bgcolor='#0D0221',
                         plot_bgcolor='#1A0033', font=dict(color='#FFFFFF'), yaxis_type='log')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: HIDDEN GEMS
# ============================================================================

elif page == "Hidden Gems":
    st.title("Hidden Gems Finder")
    
    movie_stats = filtered_df.groupby('movieId').agg({
        'rating': ['count', 'mean'],
        'title': 'first'
    }).reset_index()
    movie_stats.columns = ['movieId', 'num_ratings', 'avg_rating', 'title']
    
    st.subheader("Criteria")
    col1, col2, col3 = st.columns(3)
    with col1:
        min_rating = st.slider("Min Rating", 3.0, 5.0, 4.0, 0.1, key="gem_min")
    with col2:
        max_visibility = st.slider("Max Ratings", 1, 100, 20, key="gem_max")
    with col3:
        min_gems = st.slider("Min Gem Ratings", 1, 10, 3, key="gem_min_ratings")
    
    gems = movie_stats[(movie_stats['avg_rating'] >= min_rating) & 
                       (movie_stats['num_ratings'] <= max_visibility) &
                       (movie_stats['num_ratings'] >= min_gems)].sort_values('avg_rating', ascending=False)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gems Found", len(gems))
    with col2:
        st.metric("Avg Rating", f"{gems['avg_rating'].mean():.2f}" if len(gems) > 0 else "N/A")
    with col3:
        st.metric("Avg Visibility", f"{gems['num_ratings'].mean():.0f}" if len(gems) > 0 else "N/A")
    
    st.markdown("---")
    st.subheader("Scatter Plot")
    
    fig = px.scatter(movie_stats, x='num_ratings', y='avg_rating', hover_data=['title'],
                    labels={'num_ratings': 'Number of Ratings', 'avg_rating': 'Avg Rating'},
                    color_continuous_scale=['#7B2CBF', '#E0AAFF'])
    fig.add_shape(type="rect", x0=0, x1=max_visibility, y0=min_rating, y1=5.0,
                 fillcolor="#E0AAFF", opacity=0.1, line=dict(color="#E0AAFF", width=2, dash="dash"))
    fig.update_layout(height=500, template='plotly_dark', paper_bgcolor='#0D0221',
                     plot_bgcolor='#1A0033', font=dict(color='#FFFFFF'))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader(f"Gems: {len(gems)}")
    if len(gems) > 0:
        for idx, (_, row) in enumerate(gems.head(20).iterrows(), 1):
            st.write(f"**{idx}. {row['title']}** | Rating: {row['avg_rating']:.2f}/5 | Ratings: {int(row['num_ratings'])}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("MovieLens Analytics Dashboard | 6.7M Ratings | 50K+ Movies")
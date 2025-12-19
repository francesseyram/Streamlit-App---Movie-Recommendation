"""
PHASE 2: STREAMLIT ANALYTICS DASHBOARD
========================================
Movie recommendation analytics with Cinematic Purple theme.

Auto-downloads data from Google Drive on first load.

Run:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="MovieLens Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cinematic Purple theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0D0221;
        color: #FFFFFF;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1A0033;
        border-right: 2px solid #7B2CBF;
    }
    
    h1, h2, h3 {
        color: #E0AAFF;
    }
    
    [data-testid="metric-container"] {
        background-color: #1A0033;
        border: 1px solid #7B2CBF;
        border-radius: 8px;
        padding: 15px;
    }
    
    .stButton > button {
        background-color: #7B2CBF;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #9D4EDD;
    }
    
    [data-testid="stExpander"] {
        border: 1px solid #7B2CBF;
        border-radius: 8px;
    }
    
    hr {
        border-color: #7B2CBF;
    }
    
    p {
        color: #D0D0D0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DOWNLOAD DATA FROM GOOGLE DRIVE
# ============================================================================

@st.cache_data
def download_from_google_drive():
    """
    Download ratings_sample_cleaned.parquet from Google Drive.
    """
    import gdown
    
    FILE_ID = "1S4Lklg1e1LpbjSnfmUdSxD3jhE0kGlBN"
    output_file = "ratings_sample_cleaned.parquet"
    
    if not os.path.exists(output_file):
        st.info("First load: Downloading data from Google Drive (30-60 seconds)...")
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={FILE_ID}",
                output_file,
                quiet=False
            )
            st.success("Data downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            st.error("Please check your Google Drive FILE_ID is correct.")
            return None
    
    return output_file

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load cleaned data from parquet file."""
    # Ensure data file exists
    data_file = download_from_google_drive()
    
    if data_file is None or not os.path.exists(data_file):
        st.error("Data file not found. Please check Google Drive setup.")
        st.stop()
    
    df = pd.read_parquet(data_file)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['genres_list'] = df['genres'].str.split('|')
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ============================================================================
# SIDEBAR: FILTERS & NAVIGATION
# ============================================================================

st.sidebar.title("MovieLens Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Dashboard Pages",
    ["Overview", "User Behavior", "Content Analysis", "Hidden Gems"],
    help="Select which analysis dashboard to view"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

year_range = st.sidebar.slider(
    "Rating Year Range",
    min_value=df['year'].min(),
    max_value=df['year'].max(),
    value=(df['year'].min(), df['year'].max()),
    help="Filter data by year of rating"
)

available_genres = set()
for genres_list in df['genres_list']:
    if isinstance(genres_list, list):
        available_genres.update(genres_list)
available_genres = sorted(list(available_genres))

selected_genres = st.sidebar.multiselect(
    "Filter by Genre",
    available_genres,
    default=None,
    help="Leave empty to see all genres"
)

rating_threshold = st.sidebar.slider(
    "Minimum Rating",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="Only show movies rated at least this high"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Summary")

filtered_df = df[
    (df['year'] >= year_range[0]) & 
    (df['year'] <= year_range[1]) &
    (df['rating'] >= rating_threshold)
]

st.sidebar.metric("Total Ratings", f"{len(filtered_df):,}")
st.sidebar.metric("Unique Users", f"{filtered_df['userId'].nunique():,}")
st.sidebar.metric("Unique Movies", f"{filtered_df['movieId'].nunique():,}")

# ============================================================================
# APPLY FILTERS FUNCTION
# ============================================================================

def apply_filters(data):
    """Apply sidebar filters to dataframe."""
    filtered = data[
        (data['year'] >= year_range[0]) & 
        (data['year'] <= year_range[1]) &
        (data['rating'] >= rating_threshold)
    ].copy()
    
    if selected_genres:
        genre_mask = filtered['genres_list'].apply(
            lambda x: isinstance(x, list) and any(g in x for g in selected_genres)
        )
        filtered = filtered[genre_mask]
    
    return filtered

filtered_df = apply_filters(df)

# ============================================================================
# PAGE 1: OVERVIEW DASHBOARD
# ============================================================================

if page == "Overview":
    st.title("Overview")
    st.markdown("High-level metrics and trends across the dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Average Rating",
            value=f"{filtered_df['rating'].mean():.2f}",
            delta=f"{filtered_df['rating'].mean() - 3.5:.2f} vs baseline",
            help="Overall average rating"
        )
    
    with col2:
        st.metric(
            label="Median Rating",
            value=f"{filtered_df['rating'].median():.1f}",
            help="Middle value (50th percentile)"
        )
    
    with col3:
        st.metric(
            label="Std Deviation",
            value=f"{filtered_df['rating'].std():.2f}",
            help="Spread of ratings"
        )
    
    with col4:
        st.metric(
            label="Data Points",
            value=f"{len(filtered_df):,}",
            help="Number of ratings"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Distribution")
        
        rating_counts = filtered_df['rating'].value_counts().sort_index()
        
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={'x': 'Rating', 'y': 'Count'},
            title="How ratings are distributed"
        )
        fig.update_traces(marker_color='#7B2CBF')
        fig.update_layout(
            height=400,
            showlegend=False,
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='#0D0221',
            plot_bgcolor='#1A0033',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Trends Over Time")
        
        ratings_by_year = filtered_df.groupby('year').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        ratings_by_year.columns = ['year', 'avg_rating', 'count']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=ratings_by_year['year'],
            y=ratings_by_year['avg_rating'],
            mode='lines+markers',
            name='Average Rating',
            line=dict(color='#E0AAFF', width=3),
            marker=dict(size=8, color='#7B2CBF')
        ))
        
        fig.update_layout(
            title="Are ratings increasing or decreasing?",
            xaxis_title="Year",
            yaxis_title="Average Rating",
            height=400,
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='#0D0221',
            plot_bgcolor='#1A0033',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Genre Performance")
    
    all_genres = []
    genre_ratings = []
    for genres_list, rating in zip(filtered_df['genres_list'], filtered_df['rating']):
        if isinstance(genres_list, list):
            for g in genres_list:
                all_genres.append(g)
                genre_ratings.append(rating)
    
    genre_df = pd.DataFrame({'genre': all_genres, 'rating': genre_ratings})
    genre_stats = genre_df.groupby('genre').agg({
        'rating': ['mean', 'count', 'std']
    }).round(2)
    genre_stats.columns = ['avg_rating', 'count', 'std_dev']
    genre_stats = genre_stats.sort_values('avg_rating', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=genre_stats.index,
        y=genre_stats['avg_rating'],
        marker=dict(
            color=genre_stats['avg_rating'],
            colorscale=[[0, '#7B2CBF'], [1, '#E0AAFF']],
            showscale=True,
            colorbar=dict(title="Avg Rating", thickness=15)
        ),
        text=genre_stats['avg_rating'].round(2),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Average Rating: %{y:.2f}<br>Count: ' + 
                      genre_stats['count'].astype(str) + '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Which genres are rated highest?",
        xaxis_title="Genre",
        yaxis_title="Average Rating",
        height=400,
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#0D0221',
        plot_bgcolor='#1A0033',
        font=dict(color='#FFFFFF')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View Genre Statistics"):
        st.dataframe(genre_stats.reset_index(), use_container_width=True)

# ============================================================================
# PAGE 2: USER BEHAVIOR ANALYSIS
# ============================================================================

elif page == "User Behavior":
    st.title("User Behavior Analysis")
    st.markdown("Understanding how different users rate movies")
    
    user_stats = filtered_df.groupby('userId').agg({
        'rating': ['count', 'mean', 'std', 'min', 'max']
    }).reset_index()
    user_stats.columns = ['userId', 'num_ratings', 'avg_rating', 'rating_std', 'min_rating', 'max_rating']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        harsh_count = (user_stats['avg_rating'] < 3).sum()
        st.metric(
            "Harsh Raters",
            harsh_count,
            f"{harsh_count/len(user_stats)*100:.1f}% of users"
        )
    
    with col2:
        moderate_count = ((user_stats['avg_rating'] >= 3) & (user_stats['avg_rating'] < 3.5)).sum()
        st.metric(
            "Moderate Raters",
            moderate_count,
            f"{moderate_count/len(user_stats)*100:.1f}% of users"
        )
    
    with col3:
        generous_count = (user_stats['avg_rating'] >= 3.5).sum()
        st.metric(
            "Generous Raters",
            generous_count,
            f"{generous_count/len(user_stats)*100:.1f}% of users"
        )
    
    with col4:
        highly_active = (user_stats['num_ratings'] > 50).sum()
        st.metric(
            "Highly Active Users",
            highly_active,
            f"{highly_active/len(user_stats)*100:.1f}% of users"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Behavior Distribution")
        
        fig = px.histogram(
            user_stats,
            x='avg_rating',
            nbins=50,
            labels={'avg_rating': 'Average Rating per User', 'count': 'Number of Users'},
            title="Are users harsh or generous?"
        )
        fig.update_traces(marker_color='#7B2CBF')
        
        mean_rating = user_stats['avg_rating'].mean()
        fig.add_vline(
            x=mean_rating,
            line_dash="dash",
            line_color="#E0AAFF",
            annotation_text=f"Mean: {mean_rating:.2f}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            height=400,
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='#0D0221',
            plot_bgcolor='#1A0033',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Activity Distribution")
        
        fig = px.histogram(
            user_stats,
            x='num_ratings',
            nbins=50,
            labels={'num_ratings': 'Number of Ratings', 'count': 'Number of Users'},
            title="How active are users?"
        )
        fig.update_traces(marker_color='#9D4EDD')
        
        median_ratings = user_stats['num_ratings'].median()
        fig.add_vline(
            x=median_ratings,
            line_dash="dash",
            line_color="#E0AAFF",
            annotation_text=f"Median: {median_ratings:.0f}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            height=400,
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='#0D0221',
            plot_bgcolor='#1A0033',
            font=dict(color='#FFFFFF'),
            yaxis_type='log'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Activity vs Rating Consistency")
    
    fig = px.scatter(
        user_stats,
        x='num_ratings',
        y='rating_std',
        hover_data=['avg_rating', 'min_rating', 'max_rating'],
        title="Do active users rate more consistently?",
        labels={
            'num_ratings': 'Number of Ratings',
            'rating_std': 'Standard Deviation'
        },
        color='avg_rating',
        color_continuous_scale=['#7B2CBF', '#E0AAFF']
    )
    
    fig.update_layout(
        height=450,
        hovermode='closest',
        template='plotly_dark',
        paper_bgcolor='#0D0221',
        plot_bgcolor='#1A0033',
        font=dict(color='#FFFFFF')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("User Segments")
    
    segments = {
        'Harsh & Inactive': (user_stats['avg_rating'] < 3) & (user_stats['num_ratings'] < 10),
        'Harsh & Active': (user_stats['avg_rating'] < 3) & (user_stats['num_ratings'] >= 10),
        'Moderate & Inactive': ((user_stats['avg_rating'] >= 3) & (user_stats['avg_rating'] < 3.5)) & (user_stats['num_ratings'] < 10),
        'Moderate & Active': ((user_stats['avg_rating'] >= 3) & (user_stats['avg_rating'] < 3.5)) & (user_stats['num_ratings'] >= 10),
        'Generous & Inactive': (user_stats['avg_rating'] >= 3.5) & (user_stats['num_ratings'] < 10),
        'Generous & Active': (user_stats['avg_rating'] >= 3.5) & (user_stats['num_ratings'] >= 10),
    }
    
    segment_data = []
    for segment_name, mask in segments.items():
        count = mask.sum()
        pct = count / len(user_stats) * 100
        segment_data.append({
            'Segment': segment_name,
            'User Count': count,
            'Percentage': f'{pct:.1f}%'
        })
    
    segment_df = pd.DataFrame(segment_data)
    st.dataframe(segment_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 3: CONTENT ANALYSIS
# ============================================================================

elif page == "Content Analysis":
    st.title("Content Analysis")
    st.markdown("Deep dive into movie characteristics and ratings")
    
    movie_stats = filtered_df.groupby('movieId').agg({
        'rating': ['count', 'mean'],
        'title': 'first',
        'release_year': 'first'
    }).reset_index()
    movie_stats.columns = ['movieId', 'num_ratings', 'avg_rating', 'title', 'release_year']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Movies",
            movie_stats['movieId'].nunique(),
            help="Number of unique movies"
        )
    
    with col2:
        st.metric(
            "Avg Ratings/Movie",
            f"{movie_stats['num_ratings'].mean():.0f}",
            help="Average ratings per movie"
        )
    
    with col3:
        st.metric(
            "Median Ratings/Movie",
            f"{movie_stats['num_ratings'].median():.0f}",
            help="Median shows long-tail distribution"
        )
    
    with col4:
        st.metric(
            "Most Rated Movie",
            movie_stats['num_ratings'].max(),
            help="Maximum ratings"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Release Year Impact")
        
        year_stats = filtered_df.dropna(subset=['release_year']).groupby('release_year').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        year_stats.columns = ['release_year', 'avg_rating', 'count']
        year_stats = year_stats[year_stats['count'] > 5]
        
        fig = px.scatter(
            year_stats,
            x='release_year',
            y='avg_rating',
            size='count',
            title="Do older/newer movies get different ratings?",
            labels={
                'release_year': 'Release Year',
                'avg_rating': 'Average Rating',
                'count': 'Number of Ratings'
            },
            color='avg_rating',
            color_continuous_scale=['#7B2CBF', '#E0AAFF'],
            hover_data=['count']
        )
        
        fig.update_layout(
            height=400,
            hovermode='closest',
            template='plotly_dark',
            paper_bgcolor='#0D0221',
            plot_bgcolor='#1A0033',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Popularity Distribution")
        
        fig = px.histogram(
            movie_stats,
            x='num_ratings',
            nbins=50,
            title="Few blockbusters, many niche movies?",
            labels={'num_ratings': 'Ratings per Movie', 'count': 'Number of Movies'}
        )
        fig.update_traces(marker_color='#9D4EDD')
        fig.update_xaxes(type='log')
        
        fig.update_layout(
            height=400,
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='#0D0221',
            plot_bgcolor='#1A0033',
            font=dict(color='#FFFFFF'),
            yaxis_type='log'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Top-Rated vs Lowest-Rated Movies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top 10 Highest-Rated (min 10 ratings)**")
        top_movies = movie_stats[movie_stats['num_ratings'] >= 10].nlargest(10, 'avg_rating')
        
        for idx, (_, row) in enumerate(top_movies.iterrows(), 1):
            st.write(
                f"{idx}. **{row['title']}** ({row['release_year']:.0f})\n"
                f"   Rating: {row['avg_rating']:.2f}/5 ({int(row['num_ratings'])} ratings)"
            )
    
    with col2:
        st.markdown("**Lowest 10 Rated (min 10 ratings)**")
        bottom_movies = movie_stats[movie_stats['num_ratings'] >= 10].nsmallest(10, 'avg_rating')
        
        for idx, (_, row) in enumerate(bottom_movies.iterrows(), 1):
            st.write(
                f"{idx}. **{row['title']}** ({row['release_year']:.0f})\n"
                f"   Rating: {row['avg_rating']:.2f}/5 ({int(row['num_ratings'])} ratings)"
            )

# ============================================================================
# PAGE 4: HIDDEN GEMS FINDER
# ============================================================================

elif page == "Hidden Gems":
    st.title("Hidden Gems Finder")
    st.markdown("Discover underrated movies with high ratings but low visibility")
    
    movie_stats = filtered_df.groupby('movieId').agg({
        'rating': ['count', 'mean'],
        'title': 'first',
        'release_year': 'first'
    }).reset_index()
    movie_stats.columns = ['movieId', 'num_ratings', 'avg_rating', 'title', 'release_year']
    
    st.subheader("Criteria for Hidden Gems")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_rating = st.slider(
            "Minimum Rating",
            min_value=3.0,
            max_value=5.0,
            value=4.0,
            step=0.1,
            key="hidden_gems_min_rating"
        )
    
    with col2:
        max_visibility = st.slider(
            "Maximum Ratings",
            min_value=1,
            max_value=100,
            value=20,
            key="hidden_gems_max_visibility"
        )
    
    with col3:
        min_gem_ratings = st.slider(
            "Minimum Gem Ratings",
            min_value=1,
            max_value=10,
            value=3,
            key="hidden_gems_min_ratings"
        )
    
    hidden_gems = movie_stats[
        (movie_stats['avg_rating'] >= min_rating) &
        (movie_stats['num_ratings'] <= max_visibility) &
        (movie_stats['num_ratings'] >= min_gem_ratings)
    ].sort_values('avg_rating', ascending=False)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Hidden Gems Found", len(hidden_gems))
    
    with col2:
        st.metric("Avg Gem Rating", f"{hidden_gems['avg_rating'].mean():.2f}")
    
    with col3:
        st.metric("Avg Visibility", f"{hidden_gems['num_ratings'].mean():.0f}")
    
    st.markdown("---")
    
    st.subheader("Hidden Gems Scatter Plot")
    
    fig = px.scatter(
        movie_stats,
        x='num_ratings',
        y='avg_rating',
        hover_data=['title', 'release_year'],
        title="Find the sweet spot: High Rating + Low Visibility",
        labels={
            'num_ratings': 'Number of Ratings (Visibility)',
            'avg_rating': 'Average Rating'
        },
        color='release_year',
        color_continuous_scale=['#7B2CBF', '#E0AAFF']
    )
    
    fig.add_shape(
        type="rect",
        x0=0, x1=max_visibility,
        y0=min_rating, y1=5.0,
        fillcolor="#E0AAFF",
        opacity=0.1,
        line=dict(color="#E0AAFF", width=2, dash="dash"),
        name="Hidden Gems Zone"
    )
    
    fig.update_layout(
        height=500,
        hovermode='closest',
        template='plotly_dark',
        paper_bgcolor='#0D0221',
        plot_bgcolor='#1A0033',
        font=dict(color='#FFFFFF')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader(f"Hidden Gems Discovered: {len(hidden_gems)}")
    
    if len(hidden_gems) > 0:
        page_size = 10
        total_pages = (len(hidden_gems) + page_size - 1) // page_size
        
        page_num = st.selectbox(
            "Page",
            range(1, total_pages + 1),
            help=f"Total {len(hidden_gems)} gems across {total_pages} pages"
        )
        
        start_idx = (page_num - 1) * page_size
        end_idx = start_idx + page_size
        
        for idx, (_, row) in enumerate(hidden_gems.iloc[start_idx:end_idx].iterrows(), start_idx + 1):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{idx}. {row['title']}** ({row['release_year']:.0f})")
            
            with col2:
                st.metric("Rating", f"{row['avg_rating']:.2f}", label_visibility="collapsed")
            
            with col3:
                st.metric("Ratings", f"{int(row['num_ratings'])}", label_visibility="collapsed")
    else:
        st.warning("No hidden gems found with current criteria. Try adjusting the filters!")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    MovieLens Analytics Dashboard - Phase 2
    
    Filters active in sidebar | Data auto-downloaded from Google Drive | Ready for Phase 3 ML Models
    """
)
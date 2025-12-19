import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import os
import gdown
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================

def download_data():
    FILE_ID = "1S4Lklg1e1LpbjSnfmUdSxD3jhE0kGlBN"
    output_file = "ratings_sample_cleaned.parquet"
    if not os.path.exists(output_file):
        try:
            print("Downloading data from Google Drive...")
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", output_file, quiet=False)
        except Exception as e:
            print(f"Download error: {e}")
            return None
    return output_file

def load_and_preprocess():
    path = download_data()
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    df['year'] = df['year'].astype(int)
    df['genres_list'] = df['genres'].str.split('|')
    return df

# Initialize Data
RAW_DF = load_and_preprocess()
if RAW_DF.empty:
    raise Exception("Data could not be loaded. Please check your connection.")

GENRES = sorted(list(set([g for sublist in RAW_DF['genres_list'] for g in sublist])))

# ============================================================================
# LOGIC FUNCTIONS
# ============================================================================

def get_filtered_df(start_year, end_year, selected_genres, min_rating):
    filtered = RAW_DF[
        (RAW_DF['year'] >= start_year) & 
        (RAW_DF['year'] <= end_year) &
        (RAW_DF['rating'] >= min_rating)
    ].copy()
    
    if selected_genres:
        filtered = filtered[filtered['genres_list'].apply(
            lambda x: isinstance(x, list) and any(g in x for g in selected_genres)
        )]
    return filtered

def update_overview(start_y, end_y, genres, min_r):
    df = get_filtered_df(start_y, end_y, genres, min_r)
    
    # Metrics
    avg_r = f"{df['rating'].mean():.2f}"
    med_r = f"{df['rating'].median():.1f}"
    std_r = f"{df['rating'].std():.2f}"
    count_r = f"{len(df):,}"
    
    # Rating Distribution Plot
    counts = df['rating'].value_counts().sort_index()
    fig1 = px.bar(x=counts.index, y=counts.values, labels={'x': 'Rating', 'y': 'Count'}, template='plotly_dark')
    fig1.update_traces(marker_color='#7B2CBF')
    fig1.update_layout(paper_bgcolor='#0D0221', plot_bgcolor='#1A0033', font_color="#FFFFFF")

    # Genre Performance Plot
    all_genres, genre_ratings = [], []
    for gs, r in zip(df['genres_list'], df['rating']):
        for g in gs:
            all_genres.append(g); genre_ratings.append(r)
    
    gdf = pd.DataFrame({'genre': all_genres, 'rating': genre_ratings})
    stats = gdf.groupby('genre')['rating'].mean().reset_index().sort_values('rating', ascending=False)
    
    fig2 = px.bar(stats, x='genre', y='rating', color='rating', 
                  color_continuous_scale=['#7B2CBF', '#E0AAFF'], template='plotly_dark')
    fig2.update_layout(paper_bgcolor='#0D0221', plot_bgcolor='#1A0033', font_color="#FFFFFF")

    return avg_r, med_r, std_r, count_r, fig1, fig2

def find_hidden_gems(start_y, end_y, genres, min_r, gem_min_score, gem_max_v):
    df = get_filtered_df(start_y, end_y, genres, min_r)
    movie_stats = df.groupby('title').agg({'rating': ['count', 'mean'], 'year': 'first'}).reset_index()
    movie_stats.columns = ['title', 'count', 'avg_rating', 'year']
    
    gems = movie_stats[
        (movie_stats['avg_rating'] >= gem_min_score) & 
        (movie_stats['count'] <= gem_max_v) & 
        (movie_stats['count'] >= 3)
    ].sort_values('avg_rating', ascending=False)
    
    return gems.head(20)

# ============================================================================
# CUSTOM THEME
# ============================================================================

cinematic_theme = gr.themes.Default(
    primary_hue="purple",
    secondary_hue="violet",
    neutral_hue="slate",
).set(
    body_background_fill="#0D0221",
    block_background_fill="#1A0033",
    block_border_width="2px",
    block_border_color="#7B2CBF",
    button_primary_background_fill="#7B2CBF",
    button_primary_background_fill_hover="#9D4EDD",
    button_primary_text_color="white",
    block_title_text_color="#E0AAFF",
    block_label_text_color="#E0AAFF"
)

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(theme=cinematic_theme, title="MovieLens Analytics") as demo:
    gr.Markdown("# 🎬 MovieLens Cinematic Analytics Dashboard")
    
    with gr.Row():
        # Left Panel (Filters)
        with gr.Column(scale=1):
            gr.Markdown("### 🛠 Filters")
            start_year = gr.Slider(
                minimum=int(RAW_DF['year'].min()), 
                maximum=int(RAW_DF['year'].max()), 
                value=int(RAW_DF['year'].min()), 
                label="Start Year", 
                step=1
            )
            end_year = gr.Slider(
                minimum=int(RAW_DF['year'].min()), 
                maximum=int(RAW_DF['year'].max()), 
                value=int(RAW_DF['year'].max()), 
                label="End Year", 
                step=1
            )
            genre_multi = gr.Dropdown(choices=GENRES, multiselect=True, label="Filter by Genre")
            rating_slider = gr.Slider(minimum=0.5, maximum=5.0, value=2.0, step=0.5, label="Min Rating (Global)")
            apply_btn = gr.Button("Apply Filters", variant="primary")
            
        # Right Panel (Tabs)
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("Overview"):
                    with gr.Row():
                        m1 = gr.Textbox(label="Avg Rating", interactive=False)
                        m2 = gr.Textbox(label="Median Rating", interactive=False)
                        m3 = gr.Textbox(label="Std Dev", interactive=False)
                        m4 = gr.Textbox(label="Total Ratings", interactive=False)
                    
                    with gr.Row():
                        plot_dist = gr.Plot(label="Rating Distribution")
                        plot_genre = gr.Plot(label="Genre Performance")

                with gr.TabItem("Hidden Gems"):
                    gr.Markdown("### Discover Underrated Masterpieces")
                    with gr.Row():
                        hg_min = gr.Slider(3.0, 5.0, value=4.0, label="Min Gem Rating")
                        hg_max_v = gr.Slider(1, 100, value=20, label="Max Visibility (Ratings Count)")
                    
                    hg_btn = gr.Button("Search for Gems", variant="secondary")
                    hg_table = gr.Dataframe(label="Discovered Gems")

    # Event Binders
    apply_btn.click(
        update_overview, 
        inputs=[start_year, end_year, genre_multi, rating_slider], 
        outputs=[m1, m2, m3, m4, plot_dist, plot_genre]
    )
    
    hg_btn.click(
        find_hidden_gems,
        inputs=[start_year, end_year, genre_multi, rating_slider, hg_min, hg_max_v],
        outputs=hg_table
    )

    # Initial Load
    demo.load(
        update_overview, 
        inputs=[start_year, end_year, genre_multi, rating_slider], 
        outputs=[m1, m2, m3, m4, plot_dist, plot_genre]
    )

if __name__ == "__main__":
    demo.launch()
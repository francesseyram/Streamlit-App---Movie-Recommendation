# Save this as streamlit_app.py and push

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="MovieLens Analytics", layout="wide")

st.markdown("""<style>
.stApp { background-color: #0D0221; color: #FFFFFF; }
[data-testid="stSidebar"] { background-color: #1A0033; }
h1, h2, h3 { color: #E0AAFF; }
</style>""", unsafe_allow_html=True)

@st.cache_data
def load():
    import gdown
    if not os.path.exists('ratings_sample_cleaned.parquet'):
        gdown.download("https://drive.google.com/uc?id=1S4Lklg1e1LpbjSnfmUdSxD3jhE0kGlBN", 
                      'ratings_sample_cleaned.parquet', quiet=False)
    return pd.read_parquet('ratings_sample_cleaned.parquet')

st.title("MovieLens Analytics")
df = load()

st.sidebar.write(f"Ratings: {len(df):,}")
st.sidebar.write(f"Users: {df['userId'].nunique():,}")
st.sidebar.write(f"Movies: {df['movieId'].nunique():,}")

st.write(f"✓ Loaded {len(df):,} ratings")
st.write(df.head(10))
# Replace your streamlit_app.py with this minimal version first:

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="MovieLens Analytics", layout="wide")

st.markdown("""<style>
.stApp { background-color: #0D0221; color: #FFFFFF; }
[data-testid="stSidebar"] { background-color: #1A0033; border-right: 2px solid #7B2CBF; }
h1, h2, h3 { color: #E0AAFF; }
</style>""", unsafe_allow_html=True)

@st.cache_data
def download_from_google_drive():
    import gdown
    FILE_ID = "1S4Lklg1e1LpbjSnfmUdSxD3jhE0kGlBN"
    output_file = "ratings_sample_cleaned.parquet"
    
    if not os.path.exists(output_file):
        st.info("Downloading data...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", output_file, quiet=False)
    return output_file

@st.cache_data
def load_data():
    data_file = download_from_google_drive()
    df = pd.read_parquet(data_file)
    return df

st.title("MovieLens Analytics Dashboard")

try:
    df = load_data()
    st.success(f"Loaded {len(df):,} ratings successfully!")
    st.write(f"Columns: {list(df.columns)}")
    st.write(df.head())
except Exception as e:
    st.error(f"ERROR: {str(e)}")
    import traceback
    st.write(traceback.format_exc())
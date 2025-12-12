"""
Chart 2: Which Cars Are Most Similar?
Euclidean distance-based KNN recommender to find similar cars.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import httpx
from selectolax.parser import HTMLParser
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_PATH = SCRIPT_DIR.parent / "tables" / "carsized_data_clean.csv"

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Hide Streamlit chrome for cleaner embedding
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# Features for similarity calculation
FEATURES = ['length', 'width', 'height', 'wheelbase', 'weight', 'cargo_volume']

# Prepare data - only keep cars with all features available
@st.cache_data
def prepare_recommender_data(df):
    df_valid = df.dropna(subset=FEATURES).copy()
    df_valid = df_valid.reset_index(drop=True)

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_valid[FEATURES])

    # Fit KNN model (6 neighbors: 1 is self, 5 are recommendations)
    nn = NearestNeighbors(n_neighbors=6, metric='euclidean')
    nn.fit(features_scaled)

    return df_valid, features_scaled, nn, scaler

df_valid, features_scaled, nn_model, scaler = prepare_recommender_data(df)

# Function to get car image from carsized.com
@st.cache_data(ttl=3600)
def get_car_image(url):
    """Scrape car thumbnail image from carsized.com page."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        response.raise_for_status()

        tree = HTMLParser(response.text)

        # Find the main car image - look for img with side-view in src
        for img in tree.css('img'):
            src = img.attributes.get('src', '')
            if 'side-view' in src or '/resources/' in src:
                if src.startswith('/'):
                    return f"https://www.carsized.com{src}"
                return src

        # Fallback: find any img in the car display area
        for img in tree.css('img'):
            src = img.attributes.get('src', '')
            if '/resources/' in src and '.png' in src:
                if src.startswith('/'):
                    return f"https://www.carsized.com{src}"
                return src

        return None
    except httpx.RequestError:
        return None

def get_similar_cars(car_label, n_recommendations=5):
    """Find n most similar cars to the selected car."""
    # Find the car in our valid dataset
    car_idx = df_valid[df_valid['car_label'] == car_label].index

    if len(car_idx) == 0:
        return None, None

    car_idx = car_idx[0]

    # Get nearest neighbors
    distances, indices = nn_model.kneighbors([features_scaled[car_idx]])

    # Skip first result (it's the car itself)
    similar_indices = indices[0][1:n_recommendations+1]
    similar_distances = distances[0][1:n_recommendations+1]

    # Convert distances to similarity percentages
    # Using formula: similarity = 1 / (1 + distance) * 100
    similarities = 1 / (1 + similar_distances) * 100

    similar_cars = df_valid.iloc[similar_indices].copy()
    similar_cars['similarity'] = similarities

    return similar_cars, similarities

# Create placeholder for image (will be filled after selection)
image_placeholder = st.empty()

# UI - dropdown
selected_car = st.selectbox(
    "Select a car to find similar vehicles:",
    options=df_valid['car_label'].tolist(),
    index=0
)

# Get selected car data and display image in the placeholder above
selected_car_df = df_valid[df_valid['car_label'] == selected_car]
if selected_car_df.empty:
    st.warning("Selected car not found in dataset.")
    st.stop()
selected_car_data = selected_car_df.iloc[0]
selected_car_img = get_car_image(selected_car_data['url'])

# Display selected car image in placeholder (left-aligned, 1/2 size of recommendation images)
if selected_car_img:
    with image_placeholder.container():
        col_img, col_empty = st.columns([1, 9])
        with col_img:
            st.image(selected_car_img, width='stretch')

# Get similar cars
similar_cars, similarities = get_similar_cars(selected_car, n_recommendations=5)

if similar_cars is not None and len(similar_cars) > 0:
    # Convert to list for easier indexing
    cars_list = list(similar_cars.iterrows())

    # Row 1: 3 cars (with padding columns to make images smaller)
    cols_row1 = st.columns([0.5, 1, 0.5, 1, 0.5, 1, 0.5])
    for i, col_idx in enumerate([1, 3, 5]):
        idx, car = cars_list[i]
        with cols_row1[col_idx]:
            img_url = get_car_image(car['url'])
            if img_url:
                st.image(img_url, width='stretch')
            else:
                st.markdown(
                    """<div style="background-color: #f0f0f0; height: 80px;
                    display: flex; align-items: center; justify-content: center;
                    border-radius: 8px; color: #888;">No image</div>""",
                    unsafe_allow_html=True
                )

    # Row 1: Car names (centered, manufacturer on one line, model on next)
    cols_row1 = st.columns([0.5, 1, 0.5, 1, 0.5, 1, 0.5])
    for i, col_idx in enumerate([1, 3, 5]):
        idx, car = cars_list[i]
        with cols_row1[col_idx]:
            st.markdown(f"<div style='text-align: center;'><b>{car['manufacturer']}</b><br><b>{car['car_name']}</b></div>", unsafe_allow_html=True)

    # Row 1: Production dates (centered)
    cols_row1 = st.columns([0.5, 1, 0.5, 1, 0.5, 1, 0.5])
    for i, col_idx in enumerate([1, 3, 5]):
        idx, car = cars_list[i]
        with cols_row1[col_idx]:
            years = f"({int(car['production_start'])}-{car['production_end_display']})"
            st.markdown(f"<div style='text-align: center; color: #666; font-size: 0.9em;'>{years}</div>", unsafe_allow_html=True)

    # Row 1: Similarity scores (centered)
    cols_row1 = st.columns([0.5, 1, 0.5, 1, 0.5, 1, 0.5])
    for i, col_idx in enumerate([1, 3, 5]):
        idx, car = cars_list[i]
        with cols_row1[col_idx]:
            st.markdown(
                f"<div style='text-align: center; font-size: 1.2em; color: #2e7d32; font-weight: bold;'>"
                f"{car['similarity']:.1f}% similar</div>",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: 2 cars (centered with padding)
    cols_row2 = st.columns([1, 1, 0.5, 1, 1])
    for i, col_idx in enumerate([1, 3]):
        idx, car = cars_list[3 + i]
        with cols_row2[col_idx]:
            img_url = get_car_image(car['url'])
            if img_url:
                st.image(img_url, width='stretch')
            else:
                st.markdown(
                    """<div style="background-color: #f0f0f0; height: 80px;
                    display: flex; align-items: center; justify-content: center;
                    border-radius: 8px; color: #888;">No image</div>""",
                    unsafe_allow_html=True
                )

    # Row 2: Car names (centered, manufacturer on one line, model on next)
    cols_row2 = st.columns([1, 1, 0.5, 1, 1])
    for i, col_idx in enumerate([1, 3]):
        idx, car = cars_list[3 + i]
        with cols_row2[col_idx]:
            st.markdown(f"<div style='text-align: center;'><b>{car['manufacturer']}</b><br><b>{car['car_name']}</b></div>", unsafe_allow_html=True)

    # Row 2: Production dates (centered)
    cols_row2 = st.columns([1, 1, 0.5, 1, 1])
    for i, col_idx in enumerate([1, 3]):
        idx, car = cars_list[3 + i]
        with cols_row2[col_idx]:
            years = f"({int(car['production_start'])}-{car['production_end_display']})"
            st.markdown(f"<div style='text-align: center; color: #666; font-size: 0.9em;'>{years}</div>", unsafe_allow_html=True)

    # Row 2: Similarity scores (centered)
    cols_row2 = st.columns([1, 1, 0.5, 1, 1])
    for i, col_idx in enumerate([1, 3]):
        idx, car = cars_list[3 + i]
        with cols_row2[col_idx]:
            st.markdown(
                f"<div style='text-align: center; font-size: 1.2em; color: #2e7d32; font-weight: bold;'>"
                f"{car['similarity']:.1f}% similar</div>",
                unsafe_allow_html=True
            )
else:
    st.warning("Could not find similar cars. The selected car may be missing dimensional data.")

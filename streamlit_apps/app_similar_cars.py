"""
Chart 2: Which Cars Are Most Similar?
Euclidean distance-based KNN recommender to find similar cars.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_PATH = SCRIPT_DIR.parent / "tables" / "carsized_data_clean.csv"
IMAGE_URLS_PATH = SCRIPT_DIR.parent / "tables" / "car_image_urls.json"

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Hide Streamlit chrome and set dark theme
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
.stApp {background-color: #000000;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load data and pre-scraped image URLs
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    with open(IMAGE_URLS_PATH, 'r') as f:
        image_urls = json.load(f)
    return df, image_urls

df, IMAGE_URLS = load_data()

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

# Look up car image URL from pre-scraped data
def get_car_image(page_url):
    """Get car image URL from pre-scraped data."""
    return IMAGE_URLS.get(page_url)


def display_car_image(img_url, fallback_text="No image"):
    """Display car image using client-side loading (bypasses cloud IP blocks)."""
    if img_url:
        st.markdown(
            f'<img src="{img_url}" style="width: 100%; border-radius: 8px;">',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""<div style="background-color: #1a1a1a; height: 80px;
            display: flex; align-items: center; justify-content: center;
            border-radius: 8px; color: #888;">{fallback_text}</div>""",
            unsafe_allow_html=True
        )

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
with image_placeholder.container():
    col_img, col_empty = st.columns([1, 9])
    with col_img:
        display_car_image(selected_car_img)

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
            display_car_image(img_url)

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
            st.markdown(f"<div style='text-align: center; color: #999; font-size: 0.9em;'>{years}</div>", unsafe_allow_html=True)

    # Row 1: Similarity scores (centered)
    cols_row1 = st.columns([0.5, 1, 0.5, 1, 0.5, 1, 0.5])
    for i, col_idx in enumerate([1, 3, 5]):
        idx, car = cars_list[i]
        with cols_row1[col_idx]:
            st.markdown(
                f"<div style='text-align: center; font-size: 1.2em; color: #4caf50; font-weight: bold;'>"
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
            display_car_image(img_url)

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
            st.markdown(f"<div style='text-align: center; color: #999; font-size: 0.9em;'>{years}</div>", unsafe_allow_html=True)

    # Row 2: Similarity scores (centered)
    cols_row2 = st.columns([1, 1, 0.5, 1, 1])
    for i, col_idx in enumerate([1, 3]):
        idx, car = cars_list[3 + i]
        with cols_row2[col_idx]:
            st.markdown(
                f"<div style='text-align: center; font-size: 1.2em; color: #4caf50; font-weight: bold;'>"
                f"{car['similarity']:.1f}% similar</div>",
                unsafe_allow_html=True
            )
else:
    st.warning("Could not find similar cars. The selected car may be missing dimensional data.")

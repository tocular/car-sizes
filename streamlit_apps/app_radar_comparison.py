"""
Chart 3: Radar Plot - How do specific cars compare to their relative benchmarks?
Interactive radar chart with car selection and benchmark filters.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
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

# Attributes for radar chart
attributes = ['length', 'width', 'height', 'wheelbase', 'weight', 'cargo_volume']
attribute_labels = ['Length', 'Width', 'Height', 'Wheelbase', 'Weight', 'Cargo Vol.']

# Calculate global min/max for normalization
global_stats = {}
for attr in attributes:
    valid_data = df[attr].dropna()
    global_stats[attr] = {
        'min': valid_data.min(),
        'max': valid_data.max()
    }

def normalize_value(value, attr):
    """Normalize value to 0-1 scale based on global min/max."""
    if pd.isna(value):
        return None
    stats = global_stats[attr]
    return (value - stats['min']) / (stats['max'] - stats['min'])

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

# Create placeholder for image (will be filled after selection)
image_placeholder = st.empty()

# Controls container above chart
car_options = df['car_label'].tolist()
selected_car_label = st.selectbox(
    "Select Car",
    options=car_options,
    index=0
)

# Get selected car data
selected_car_df = df[df['car_label'] == selected_car_label]
if selected_car_df.empty:
    st.warning("Selected car not found in dataset.")
    st.stop()
selected_car = selected_car_df.iloc[0]

# Display selected car image in placeholder (left-aligned, 1/2 size of recommendation images)
selected_car_img = get_car_image(selected_car['url'])
if selected_car_img:
    with image_placeholder.container():
        col_img, col_empty = st.columns([1, 9])
        with col_img:
            st.image(selected_car_img, width='stretch')

# Benchmark filters in expandable dropdown
with st.expander("Benchmark Filters"):
    # Decade filter
    available_decades = sorted(df['decade'].unique().tolist())
    default_decades = available_decades

    selected_decades = st.multiselect(
        "Decades",
        options=available_decades,
        default=default_decades
    )

    # Body style filter - default to selected car's body style
    available_body_styles = sorted(df['body_style_general'].dropna().unique().tolist())
    car_body_style = selected_car['body_style_general']
    default_body_styles = [car_body_style] if car_body_style in available_body_styles else available_body_styles

    selected_body_styles = st.multiselect(
        "Body Styles",
        options=available_body_styles,
        default=default_body_styles
    )

    # Segment filter
    available_segments = sorted(df['segment'].dropna().unique().tolist())
    default_segments = available_segments

    selected_segments = st.multiselect(
        "Segments",
        options=available_segments,
        default=default_segments
    )

# Filter data for benchmark calculation
benchmark_df = df[
    (df['decade'].isin(selected_decades)) &
    (df['body_style_general'].isin(selected_body_styles)) &
    (df['segment'].isin(selected_segments))
]

# Calculate benchmark averages
benchmark_values = {}
for attr in attributes:
    benchmark_values[attr] = benchmark_df[attr].mean()

# Prepare radar data
car_normalized = [normalize_value(selected_car[attr], attr) for attr in attributes]
benchmark_normalized = [normalize_value(benchmark_values[attr], attr) for attr in attributes]

# Handle None values for display
car_display = [v if v is not None else 0 for v in car_normalized]
benchmark_display = [v if v is not None else 0 for v in benchmark_normalized]

# Close the polygon
car_display.append(car_display[0])
benchmark_display.append(benchmark_display[0])
labels_display = attribute_labels + [attribute_labels[0]]

# Create radar chart
fig = go.Figure()

# Benchmark trace (filled area, lighter)
fig.add_trace(go.Scatterpolar(
    r=benchmark_display,
    theta=labels_display,
    fill='toself',
    fillcolor='rgba(99, 110, 250, 0.2)',
    line=dict(color='rgba(99, 110, 250, 0.8)', width=2, dash='dash'),
    name=f'Benchmark Avg (n={len(benchmark_df)})',
    hovertemplate='<b>Benchmark</b><br>%{theta}: %{r:.2f}<extra></extra>'
))

# Selected car trace (solid line)
fig.add_trace(go.Scatterpolar(
    r=car_display,
    theta=labels_display,
    fill='toself',
    fillcolor='rgba(239, 85, 59, 0.3)',
    line=dict(color='rgba(239, 85, 59, 1)', width=3),
    name=selected_car_label,
    hovertemplate='<b>Selected Car</b><br>%{theta}: %{r:.2f}<extra></extra>'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            tickvals=[0.25, 0.5, 0.75, 1],
            ticktext=['25%', '50%', '75%', '100%'],
            gridcolor='lightgray'
        ),
        angularaxis=dict(
            gridcolor='lightgray'
        ),
        bgcolor='rgba(250,250,250,0.8)'
    ),
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.15,
        xanchor='center',
        x=0.5
    ),
    height=500,
    margin=dict(l=80, r=80, t=40, b=80)
)

st.plotly_chart(fig, width='stretch')

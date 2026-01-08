"""
Chart 1: Have Car Sizes Trended Upwards Since the 1990s?
Faceted bubble chart (2x2 grid) showing length, width, height, and weight trends.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_PATH = SCRIPT_DIR.parent / "tables" / "carsized_data_clean.csv"

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Hide Streamlit chrome and set dark theme
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
.stApp {background-color: #000000;}

/* Make labels more visible */
.stSlider label, .stMultiSelect label {
    color: #ffffff !important;
    font-weight: 500 !important;
}

</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# Filter controls in expandable dropdown
with st.expander("Configure chart"):
    min_year = st.slider(
        "Start Year",
        min_value=1990,
        max_value=2024,
        value=1990,
        step=1
    )

    body_styles = st.multiselect(
        "Body Styles",
        options=sorted(df['body_style_general'].dropna().unique().tolist()),
        default=sorted(df['body_style_general'].dropna().unique().tolist())
    )

# Check if at least one body style is selected
if not body_styles:
    st.warning("Please select at least one body style to display the chart.")
    st.stop()

# Filter data
df_filtered = df[
    (df['production_start'] >= min_year) &
    (df['body_style_general'].isin(body_styles))
].copy()

# Aggregate by year for cleaner visualization
df_agg = df_filtered.groupby('production_start').agg({
    'length': 'mean',
    'width': 'mean',
    'height': 'mean',
    'weight': 'mean',
    'car_name': 'count'
}).reset_index()
df_agg.columns = ['year', 'length', 'width', 'height', 'weight', 'count']

# Create 2x2 subplot with more vertical spacing for titles
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Length (cm)',
        'Width (cm)',
        'Height (cm)',
        'Weight (kg)'
    ),
    horizontal_spacing=0.1,
    vertical_spacing=0.18
)

dimensions = [
    ('length', 1, 1, '#636EFA'),
    ('width', 1, 2, '#EF553B'),
    ('height', 2, 1, '#00CC96'),
    ('weight', 2, 2, '#AB63FA')
]

for dim, row, col, color in dimensions:
    # Get data for this dimension
    x = df_agg['year'].values
    y = df_agg[dim].values
    sizes = df_agg['count'].values

    # Remove NaN values for trendline calculation
    mask = ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    # Calculate trendline
    if len(x_clean) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        trendline_y = slope * x_clean + intercept

        # Add trendline
        fig.add_trace(
            go.Scatter(
                x=x_clean,
                y=trendline_y,
                mode='lines',
                line=dict(dash='dot', color=color, width=2),
                name=f'{dim.capitalize()} trend',
                showlegend=False,
                hovertemplate=f'Trend: %{{y:.1f}}<extra></extra>'
            ),
            row=row, col=col
        )

    # Add bubble chart
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=sizes,
                sizemode='area',
                sizeref=2.*max(sizes)/(40.**2) if len(sizes) > 0 else 1,
                sizemin=4,
                color=color,
                opacity=0.6,
                line=dict(width=1, color='white')
            ),
            name=dim.capitalize(),
            showlegend=False,
            hovertemplate=(
                f'<b>{dim.capitalize()}</b><br>'
                'Year: %{x}<br>'
                'Avg: %{y:.1f}<br>'
                'Cars: %{marker.size}<extra></extra>'
            )
        ),
        row=row, col=col
    )

# Update layout - dark theme, no title, more top margin for subplot titles
fig.update_layout(
    height=550,
    margin=dict(l=60, r=30, t=40, b=50),
    font=dict(size=13, color='#ffffff'),
    plot_bgcolor='#000000',
    paper_bgcolor='#000000'
)

# Update subplot title positions to add space below them and make them bold
for annotation in fig['layout']['annotations']:
    annotation['y'] = annotation['y'] + 0.02
    annotation['font'] = dict(size=14, color='#ffffff', weight=700)

# Update axes with more visible fonts
fig.update_xaxes(title_text='Production Year', title_font=dict(size=12, color='#ffffff'), row=2, col=1)
fig.update_xaxes(title_text='Production Year', title_font=dict(size=12, color='#ffffff'), row=2, col=2)
fig.update_yaxes(gridcolor='#333333', gridwidth=0.5, tickfont=dict(size=11, color='#ffffff'))
fig.update_xaxes(gridcolor='#333333', gridwidth=0.5, tickfont=dict(size=11, color='#ffffff'))

st.plotly_chart(fig, width='stretch')

# Quantifying Car Size

An analysis of automotive dimensional trends using data from carsized.com, covering 2,300+ vehicle models.

## Overview

This project examines how car dimensions have changed over time. The analysis includes:

- **Dimension trends**: Tracking length, width, height, and weight from 1990 to present
- **Benchmark comparisons**: Comparing individual vehicles against segment averages via radar charts
- **Similarity analysis**: Finding dimensionally similar vehicles using KNN-based Euclidean distance

## Data

The dataset contains measurements for:
- Length, width, height (cm)
- Weight (kg)
- Wheelbase (cm)
- Cargo volume (liters)

Vehicles are categorized by body style, market segment, and production decade.

## Project Structure

```
car-sizes/
├── report/              # Quarto website
│   └── index.qmd
├── streamlit_apps/      # Interactive visualizations
│   ├── app_dimension_trends.py
│   ├── app_radar_comparison.py
│   └── app_similar_cars.py
├── tables/              # Processed data
└── requirements.txt
```

## Tech Stack

- **Analysis**: pandas, scikit-learn, scipy
- **Visualization**: Plotly, Streamlit
- **Report**: Quarto

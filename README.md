# 🏚️ Housing Crash Course

Is that house a steal or a rip-off? A simple Streamlit app for 46-887 (Machine Learning for Business Applications).

**Live app:** https://housing-crash-course.streamlit.app

## What it does

1. Upload a CSV and see a preview
2. Pick a column and a chart type (histogram, scatter, or line)
3. Pick two inputs and a target, fit a linear regression, and get a prediction for values you type in

## Data

`Housing.csv` is the **Housing Prices Dataset** from Kaggle (545 houses: price, area, bedrooms, bathrooms, stories, parking, and a few yes/no features). Download it from this repo and upload it in the app.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

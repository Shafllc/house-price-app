# House Price Explorer

A simple Streamlit app built for class. It lets you:

1. Upload a CSV file and see a preview
2. Pick a numeric column and a chart type (line, scatter, or histogram)
3. Pick two inputs and a target, fit a linear regression, and get a prediction for values you type in

## Data

`Housing.csv` is the **Housing Prices Dataset** from Kaggle (545 houses; price, area, bedrooms, bathrooms, stories, parking, and a few yes/no features). Upload it in the app to try it out.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("House Price Explorer")
st.write("Upload a CSV (for example Housing.csv), explore a column, and predict a number with linear regression.")

# 1. Upload the CSV and show a preview
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Data preview")
st.dataframe(df.head())

numeric_cols = df.select_dtypes(include="number").columns.tolist()

if len(numeric_cols) < 3:
    st.error("This app needs a CSV with at least three numeric columns.")
    st.stop()

# 2. Choose a column and a chart type
st.subheader("Chart")
column = st.selectbox("Choose a column", numeric_cols)
chart_type = st.radio("Choose a chart type", ["Line", "Scatter", "Histogram"], horizontal=True)

fig, ax = plt.subplots()
if chart_type == "Line":
    ax.plot(df[column])
    ax.set_xlabel("Row")
    ax.set_ylabel(column)
elif chart_type == "Scatter":
    ax.scatter(df.index, df[column], s=10)
    ax.set_xlabel("Row")
    ax.set_ylabel(column)
else:
    ax.hist(df[column], bins=30)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
ax.set_title(f"{chart_type} of {column}")
st.pyplot(fig)

# 3. Linear regression with two inputs and a target
st.subheader("Linear regression")
target = st.selectbox("Choose the target (what to predict)", numeric_cols)
input_options = [c for c in numeric_cols if c != target]
input_1 = st.selectbox("Choose input 1", input_options, index=0)
input_2 = st.selectbox("Choose input 2", [c for c in input_options if c != input_1], index=0)

features = [input_1, input_2]
data = df[features + [target]].dropna()

model = LinearRegression()
model.fit(data[features], data[target])

st.write(f"Model: {target} = {model.intercept_:,.2f} "
         f"+ {model.coef_[0]:,.2f} x {input_1} + {model.coef_[1]:,.2f} x {input_2}")
st.write(f"R-squared on the data: {model.score(data[features], data[target]):.3f}")

st.write("Type in values to get a prediction:")
value_1 = st.number_input(input_1, value=float(data[input_1].mean()))
value_2 = st.number_input(input_2, value=float(data[input_2].mean()))

# Use the same column names and order as in training
new_row = pd.DataFrame([[value_1, value_2]], columns=features)
prediction = model.predict(new_row)[0]
st.success(f"Predicted {target}: {prediction:,.2f}")

# 4. What the model is for and what happens when it is wrong
st.write(
    "**About this model:** It estimates a house's sale price from simple facts like its area and number of bedrooms "
    "to help buyers and sellers set a fair asking price, and when it is wrong a seller may underprice and lose money "
    "or overprice and the house sits unsold, while a buyer may overpay."
)

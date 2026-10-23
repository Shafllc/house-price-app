import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Housing Crash Course", page_icon="🏚️", layout="wide")
st.title("🏚️ Housing Crash Course")
st.caption("Is that house a steal or a rip-off? Upload a CSV, explore it, and let a straight line guess the price.")

# Same shape as the class apps:
# the user uploads a file, and everything else sits inside an if.
uploaded_file = st.file_uploader("Upload a CSV file (try Housing.csv from the repo)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 3:
        st.error("This file needs at least three numeric columns.")
        st.stop()

    # 1. Preview
    st.subheader("1. Preview")
    st.write(f"{len(df):,} rows and {len(df.columns)} columns. Here are the first few:")
    st.dataframe(df.head(), hide_index=True, width="stretch")

    # 2. Chart
    # The user picks a column and a chart type.
    # Line and scatter need a second column to compare against, so we default to price.
    st.subheader("2. Explore a column")
    c1, c2 = st.columns(2)
    column = c1.selectbox("Column", numeric_cols, index=1)
    chart_type = c2.radio("Chart type", ["Histogram", "Scatter", "Line"], horizontal=True)

    if chart_type == "Histogram":
        fig = px.histogram(df, x=column, nbins=30, title=f"How {column} is spread out")
    else:
        others = [c for c in numeric_cols if c != column]
        default = others.index("price") if "price" in others else 0
        against = st.selectbox("Compare against", others, index=default)
        if chart_type == "Scatter":
            fig = px.scatter(df, x=column, y=against, title=f"{against} versus {column}")
        else:
            avg = df.groupby(column, as_index=False)[against].mean()
            fig = px.line(avg, x=column, y=against, markers=True, title=f"Average {against} by {column}")
    st.plotly_chart(fig, width="stretch")

    # 3. Model
    # Two inputs and one target, then a linear regression.
    st.subheader("3. Predict with a linear regression")
    m1, m2, m3 = st.columns(3)
    target = m1.selectbox("Target: what you want to predict", numeric_cols)
    input_options = [c for c in numeric_cols if c != target]
    input_1 = m2.selectbox("Input 1", input_options, index=0)
    input_2 = m3.selectbox("Input 2", [c for c in input_options if c != input_1], index=0)

    features = [input_1, input_2]
    data = df[features + [target]].dropna()
    if len(data) < 5:
        st.error("Need at least 5 rows with values in all three columns.")
        st.stop()

    X = data[features]
    y = data[target]
    model = LinearRegression()
    model.fit(X, y)

    k1, k2, k3 = st.columns(3)
    k1.metric(f"Each extra {input_1} adds", f"{model.coef_[0]:,.0f}")
    k2.metric(f"Each extra {input_2} adds", f"{model.coef_[1]:,.0f}")
    k3.metric("R2 on this data", f"{model.score(X, y):.2f}")

    # Try it: the user types values and the model gives an answer.
    # The table we pass to predict has the same column names in the same order as training.
    st.subheader("Try it")
    t1, t2 = st.columns(2)
    values = []
    for col, box in zip(features, [t1, t2]):
        if pd.api.types.is_integer_dtype(data[col]):
            values.append(box.number_input(f"Enter {col}", value=int(data[col].median()), step=1))
        else:
            values.append(box.number_input(f"Enter {col}", value=float(data[col].median())))

    new_row = pd.DataFrame([values], columns=features)
    prediction = model.predict(new_row)[0]
    st.metric(f"Predicted {target}", f"{prediction:,.0f}")

    # A line does not know where its data stopped, so warn when a value is outside the range.
    for col, value in zip(features, values):
        low, high = data[col].min(), data[col].max()
        if value < low or value > high:
            st.warning(f"{col} = {value:,} is outside the range the model learned from ({low:,} to {high:,}). Treat the number with caution.")

    # What the model is for, and what happens when it is wrong.
    st.info(
        "**What is this model for?** It gives a quick estimate of a house's price from its size and number of bedrooms, "
        "so a buyer or seller can tell if an asking price is reasonable; when it is wrong, a seller can underprice and "
        "lose money or overprice and the house sits unsold, and a buyer can overpay."
    )
else:
    st.info("Upload a CSV file to get started. It needs at least three numeric columns.")

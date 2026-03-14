import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("📊 Sales & Demand Forecasting Dashboard")

st.write(
"""
This dashboard predicts **future sales demand** using a Machine Learning model 
trained on historical Superstore sales data.
"""
)

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("models/sales_forecast_results.pkl","rb"))

# -----------------------------
# Load Data
# -----------------------------
data = pd.read_csv("data/Sample - Superstore.csv", encoding="latin1")

data['Order Date'] = pd.to_datetime(data['Order Date'], format='mixed')

daily_sales = data.groupby('Order Date')['Sales'].sum().reset_index()

# -----------------------------
# KPI Metrics
# -----------------------------
st.subheader("📈 Key Business Metrics")

col1, col2, col3 = st.columns(3)

total_sales = round(daily_sales['Sales'].sum(),2)
avg_sales = round(daily_sales['Sales'].mean(),2)
max_sales = round(daily_sales['Sales'].max(),2)

col1.metric("Total Sales", f"${total_sales}")
col2.metric("Average Daily Sales", f"${avg_sales}")
col3.metric("Highest Daily Sales", f"${max_sales}")

# -----------------------------
# Sales Trend Chart
# -----------------------------
st.subheader("📉 Historical Sales Trend")

fig, ax = plt.subplots()

ax.plot(daily_sales['Order Date'], daily_sales['Sales'])
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Sales Trend Over Time")

st.pyplot(fig)

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🔮 Predict Future Sales")

col1, col2 = st.columns(2)

with col1:
    input_date = st.date_input("Select Date")

# with col2:
#     st.write("Click the button below to predict sales")

year = input_date.year
month = input_date.month
day = input_date.day
day_of_week = input_date.weekday()

input_data = pd.DataFrame({
    "year":[year],
    "month":[month],
    "day":[day],
    "day_of_week":[day_of_week]
})

if st.button("Predict Sales"):

    prediction = model.predict(input_data)

    st.success(f"Predicted Sales: ${prediction[0]:.2f}")

# -----------------------------
# Forecast Table
# -----------------------------
st.subheader("📋 Forecast Data")

forecast_data = pd.read_csv("data/sales_forecast_results.csv")

st.dataframe(forecast_data.head(15))
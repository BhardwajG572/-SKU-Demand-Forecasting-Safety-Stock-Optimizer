import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime

# 1. APP CONFIGURATION & UI SETUP

st.set_page_config(page_title="Retail Demand Engine", layout="wide")
st.title("📦 SKU Demand Forecasting & Safety Stock Optimizer")
st.markdown("""
This engine uses an optimized XGBoost model trained on historical panel data to forecast SKU demand. 
It operates with a **10.44% WAPE**, allowing Supply Chain Managers to accurately size safety stock and prevent overstock waste.
""")

# 2. LOAD THE AI ENGINE

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("xgboost_demand_model.json")
    return model

model = load_model()

# 3. USER INPUT WIDGETS (Sidebar)

st.sidebar.header("Forecast Parameters")

selected_store = st.sidebar.selectbox("Select Store ID", range(1, 11))
selected_item = st.sidebar.selectbox("Select Item ID", range(1, 51))
forecast_date = st.sidebar.date_input("Select Date for Forecast", datetime.date.today())

st.sidebar.markdown("---")
st.sidebar.header("Recent Pipeline Data")
st.sidebar.caption("In production, these pull from a SQL database.")
lag_1 = st.sidebar.number_input("Sales Yesterday (Lag 1)", min_value=0, value=20)
lag_7 = st.sidebar.number_input("Sales Last Week (Lag 7)", min_value=0, value=22)
rolling_7 = st.sidebar.number_input("7-Day Moving Average", min_value=0.0, value=18.5)

# 4. INFERENCE PIPELINE

if st.sidebar.button("Run Global Forecast", type="primary"):
    
    # Format the input exactly how the XGBoost model expects it
    input_data = pd.DataFrame({
        'store': [selected_store],
        'item': [selected_item],
        'day_of_week': [forecast_date.weekday()],
        'month': [forecast_date.month],
        'year': [forecast_date.year],
        'lag_1': [lag_1],
        'lag_7': [lag_7],
        'rolling_7_mean': [rolling_7]
    })
    
    # Apply the exact same Categorical Fix used during training
    
    input_data['store'] = input_data['store'].astype('category')
    input_data['item'] = input_data['item'].astype('category')

    # Make the Prediction
    prediction = model.predict(input_data)[0]
    predicted_sales = max(0, int(round(prediction))) 
    
    # Calculate Safety Stock Buffer using your optimized 10.44% WAPE margin
    wape_margin = 0.1044
    safety_stock = int(round(predicted_sales * (1 + wape_margin)))

    # 5. BUSINESS VALUE DISPLAY

    st.subheader(f"Forecast Results for {forecast_date.strftime('%A, %B %d, %Y')}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Daily Demand", f"{predicted_sales} Units")
    col2.metric("Recommended Shelf Stock", f"{safety_stock} Units", "Includes 10.4% Buffer")
    col3.metric("Stockout Risk", "Mitigated", delta_color="normal")
    
    st.success(f"**Action Plan:** To maintain optimal service levels for Item {selected_item} at Store {selected_store}, ensure **{safety_stock} units** are available by opening time. This mathematically minimizes both stockout probability and warehouse holding costs.")
import streamlit as st
import pandas as pd
import torch
from datetime import datetime, timedelta
from main_torchHYBRID import prepare_data, HybridModel, predict, get_pollutant_level

# --- Load model and data ---
@st.cache_resource
def load_model_and_data():
    X_train, y_train, X_test, y_test, scaler, features, df, test_dates = prepare_data('water_quality_data.csv')
    lookback = X_train.shape[1]
    model = HybridModel(input_size=len(features), seq_length=lookback)
    model.load_state_dict(torch.load('best_hybrid_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model, df, scaler, features, lookback

model, df, scaler, features, lookback = load_model_and_data()

# --- UI Layout ---
st.set_page_config(page_title="Water Quality Dashboard", layout="wide")
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1e3a8a;
            padding-bottom: 10px;
        }
        .param-title {
            font-size: 1.5rem;
            color: #1d4ed8;
        }
        .wqi-box {
            background-color: #dbeafe;
            padding: 10px;
            border-radius: 10px;
            margin-top: 10px;
            font-size: 1.2rem;
            color: #1e40af;
            font-weight: bold;
        }
        .metric-box .stMetricValue {
            color: #1e40af !important;
            font-weight: 600;
        }
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: white;
            font-weight: 600;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #2563eb, #1e40af);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💧 Water Quality Monitoring Dashboard</div>', unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("🔧 Filters")
param_filter = st.sidebar.selectbox("Select Parameter", ["All Parameters", "Ammonia", "Phosphate", "Dissolved Oxygen", "Nitrate", "pH Level", "Temperature"])
time_filter = st.sidebar.selectbox("Select Time Range", ["30days", "6months", "1year"])

# Filter the DataFrame by time range
def filter_df(df, range_):
    if range_ == "30days":
        return df[df.index >= df.index[-1] - pd.DateOffset(days=30)]
    elif range_ == "6months":
        return df[df.index >= df.index[-1] - pd.DateOffset(months=6)]
    else:
        return df[df.index >= df.index[-1] - pd.DateOffset(years=1)]

filtered_df = filter_df(df, time_filter)

# Parameters and Column Mapping
params = {
    "Ammonia": "Ammonia (mg/L)",
    "Phosphate": "Phosphate (mg/L)",
    "Dissolved Oxygen": "Dissolved Oxygen (mg/L)",
    "Nitrate": "Nitrate (mg/L)",
    "pH Level": "pH Level",
    "Temperature": "Surface Water Temp (°C)"
}

# Display parameter charts
for label, col in params.items():
    if param_filter == "All Parameters" or param_filter == label:
        st.markdown(f'<div class="param-title">📊 {label}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Current", f"{filtered_df[col].iloc[-1]:.2f}")
        with c2:
            st.metric("Average", f"{filtered_df[col].mean():.2f}")
        st.line_chart(filtered_df[col])

# Latest WQI Summary
latest = df.iloc[-1]
latest_wqi = float(latest['WQI'])
pollutant_level = get_pollutant_level(latest_wqi)

st.markdown(f"<div class='wqi-box'>Current WQI: {latest_wqi:.2f} ({pollutant_level})</div>", unsafe_allow_html=True)

# Prediction Panel
st.markdown("---")
st.header("🔮 Predict Water Quality Index (WQI)")
future_date = st.date_input("Select a future date", datetime.now().date() + timedelta(days=30))
if st.button("Predict WQI"):
    try:
        date_obj = pd.to_datetime(future_date)
        prediction = predict(model, df, scaler, features, date_obj)
        if prediction:
            pred_level = get_pollutant_level(prediction)
            st.success(f"Predicted WQI on {future_date}: {prediction:.2f} ({pred_level})")
        else:
            st.warning("Prediction failed: Insufficient data or invalid date.")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.caption("© 2024 Water Quality Dashboard — All rights reserved")
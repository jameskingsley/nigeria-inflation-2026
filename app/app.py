import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Nigeria Inflation 2026", page_icon="🇳🇬", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🇳🇬 Nigeria Inflation Forecasting Dashboard")
st.subheader("Automated 2026 Projections via MLOps Pipeline")

# --- SIDEBAR CONFIG ---
st.sidebar.header("Deployment Settings")
# Replace with your actual Render URL once deployed
API_URL = st.sidebar.text_input("FastAPI URL", "https://your-api-name.onrender.com")
horizon = st.sidebar.slider("Forecast Months", 1, 24, 12)

# --- DATA FETCHING ---
if st.button("Generate 2026 Forecast"):
    try:
        with st.spinner('📡 Fetching latest production model predictions...'):
            response = requests.get(f"{API_URL}/predict?months={horizon}", timeout=15)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'])
            
            # --- TOP METRICS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Model", data['model_type'])
            col2.metric("Avg. Forecast Rate", f"{df['rate'].mean():.2f}%")
            col3.metric("Peak Inflation", f"{df['rate'].max()}%")

            # --- PLOTLY CHART ---
            fig = go.Figure()
            # Confidence Interval (Shaded)
            fig.add_trace(go.Scatter(
                x=pd.concat([df['date'], df['date'][::-1]]),
                y=pd.concat([df['high'], df['low'][::-1]]),
                fill='toself',
                fillcolor='rgba(0,176,246,0.2)',
                line_color='rgba(255,255,255,0)',
                name='Confidence Interval',
            ))
            # Main Forecast Line
            fig.add_trace(go.Scatter(x=df['date'], y=df['rate'], line=dict(color='#00b0f6', width=4), name='Forecasted Rate'))
            
            fig.update_layout(title="Inflation Rate Projection (2026)", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # --- DATA TABLE ---
            with st.expander("View Raw Forecast Data"):
                st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Could not connect to API. Verify the URL is correct and the Render service is awake. Error: {e}")
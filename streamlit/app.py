
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import pyodbc
import os
import json
from datetime import datetime, timedelta

# Page Config
st.set_page_config(
    page_title="Viral Predictor Dashboard",
    page_icon="🚀",
    layout="wide"
)

# Configuration
API_URL = "https://helk01-viral-post-predictor.hf.space/predict"
SQL_SERVER = "viral-sql-server12.database.windows.net"
SQL_DATABASE = "viral_posts_db"
SQL_USER = "sqladmin"
# Ideally load this from st.secrets, but for demo we can ask user or env var
SQL_PASSWORD = os.environ.get("SQL_PASSWORD") or st.sidebar.text_input("SQL Password", type="password")

def get_sql_connection():
    if not SQL_PASSWORD:
        return None
    try:
        conn_str = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};UID={SQL_USER};PWD={SQL_PASSWORD};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
        return pyodbc.connect(conn_str)
    except Exception as e:
        st.error(f"SQL Connection Failed: {e}")
        return None

def fetch_recent_logs():
    conn = get_sql_connection()
    if conn:
        try:
            query = "SELECT TOP 100 * FROM predictions ORDER BY prediction_timestamp DESC"
            df = pd.read_sql(query, conn)
            return df
        finally:
            conn.close()
    return pd.DataFrame()

def check_drift():
    conn = get_sql_connection()
    if conn:
        try:
            # Compare last 24h vs All Time
            query_recent = """
            SELECT AVG(viral_probability) as avg_prob 
            FROM predictions 
            WHERE prediction_timestamp > DATEADD(day, -1, GETDATE())
            """
            
            query_all = """
            SELECT AVG(viral_probability) as avg_prob 
            FROM predictions
            """
            
            recent = pd.read_sql(query_recent, conn).iloc[0]['avg_prob'] or 0
            history = pd.read_sql(query_all, conn).iloc[0]['avg_prob'] or 0
            
            return recent, history
        finally:
            conn.close()
    return 0, 0

# --- UI ---
st.title("🚀 Viral Post Predictor Workspace")
st.markdown("Hybrid Architecture: Azure ML (Training) + Hugging Face (Inference) + Azure SQL (Logging)")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Live Monitor", "📉 Drift Detection"])

with tab1:
    st.header("New Post Prediction")
    text_input = st.text_area("Enter your tweet text:", height=150, placeholder="Type something viral...")
    
    if st.button("Predict Viral Score"):
        if not text_input:
            st.warning("Please enter text!")
        else:
            with st.spinner("Calling Hugging Face Endpoint..."):
                try:
                    response = requests.post(API_URL, json={"text": text_input})
                    if response.status_code == 200:
                        result = response.json()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Is Viral?", "YES 🔥" if result['is_viral'] else "NO ❄️")
                        with col2:
                            st.metric("Viral Probability", f"{result['viral_probability']:.2%}")
                        with col3:
                            st.metric("Est. Engagement", f"{result['predicted_engagement']:,}")
                            
                        st.success("Prediction logged to Azure SQL! ✅")
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

with tab2:
    st.header("Live Azure SQL Logs")
    if st.button("Refresh Logs"):
        with st.spinner("Fetching from Azure SQL..."):
            df = fetch_recent_logs()
            if not df.empty:
                st.dataframe(df)
                
                # Simple timeline
                fig = px.line(df, x='prediction_timestamp', y='viral_probability', title="Recent Predictions Trend")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No logs found or connection failed. Check Password.")

with tab3:
    st.header("Data Drift Monitor")
    st.markdown("Comparing **Last 24 Hours** vs **Historical Baseline**")
    
    if st.button("Analyze Drift"):
        recent_avg, history_avg = check_drift()
        
        drift_delta = recent_avg - history_avg
        
        col1, col2 = st.columns(2)
        with col1:
             st.metric("24h Avg Probability", f"{recent_avg:.2%}", delta=f"{drift_delta:.2%}")
        with col2:
             st.metric("Historical Avg", f"{history_avg:.2%}")
             
        if abs(drift_delta) > 0.10:
             st.error("⚠️ DATA DRIFT DETECTED! Significant deviation from baseline.")
        else:
             st.success("✅ No significant drift detected.")


import argparse
import pandas as pd
import numpy as np
import pyodbc
import os
import mlflow
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sql_connection():
    server = os.environ.get('SQL_SERVER', 'viral-sql-server12.database.windows.net')
    database = os.environ.get('SQL_DATABASE', 'viral_posts_db')
    username = os.environ.get('SQL_USER', 'sqladmin') # Use sqladmin for Jobs
    password = os.environ.get('SQL_PASSWORD')
    
    if not password:
        raise ValueError("SQL_PASSWORD environment variable not set")
    
    conn_str = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
    return pyodbc.connect(conn_str)

def main():
    logger.info("Starting Data Drift Check...")
    
    mlflow.start_run()
    
    try:
        # 1. Fetch Recent Data (Last 24 hours) from SQL
        conn = get_sql_connection()
        query = """
        SELECT viral_probability, predicted_engagement 
        FROM predictions 
        WHERE prediction_timestamp > DATEADD(day, -1, GETDATE())
        """
        df_recent = pd.read_sql(query, conn)
        conn.close()
        
        if df_recent.empty:
            logger.warning("No new data in last 24h. Skipping drift check.")
            mlflow.log_metric("drift_detected", 0)
            return

        # 2. Define Baseline Statistics (e.g. from Training Data)
        # In a real scenario, load this from a 'baseline.json' artifact
        BASELINE = {
            'viral_prob_mean': 0.14,  # Hypothetical baseline stats
            'viral_prob_std': 0.05,
            'engagement_mean': 250,
            'engagement_std': 100
        }
        
        # 3. Calculate Current Statistics
        current_stats = {
            'viral_prob_mean': df_recent['viral_probability'].mean(),
            'viral_prob_std': df_recent['viral_probability'].std(),
            'engagement_mean': df_recent['predicted_engagement'].mean(),
            'engagement_std': df_recent['predicted_engagement'].std()
        }
        
        # 4. Check for Drift (Simple Z-Score or Threshold)
        drift_detected = False
        threshold = 2.0 # Standard deviations
        
        logger.info(f"Baseline Mean: {BASELINE['viral_prob_mean']}, Current Mean: {current_stats['viral_prob_mean']}")
        
        # Check Viral Probability Drift
        z_score_prob = abs(current_stats['viral_prob_mean'] - BASELINE['viral_prob_mean']) / BASELINE['viral_prob_std']
        if z_score_prob > threshold:
            logger.warning(f"DRIFT DETECTED: Viral Probability Z-Score {z_score_prob:.2f}")
            drift_detected = True
            
        # Log Metrics
        mlflow.log_metrics(current_stats)
        mlflow.log_metric("drift_score_prob", z_score_prob)
        mlflow.log_metric("drift_detected", int(drift_detected))
        
        if drift_detected:
            # You could trigger a retraining pipeline here using Azure ML SDK
            logger.warning("❌ DATA DRIFT CONFIRMED - Retraining recommended")
        else:
            logger.info("✅ No meaningful drift detected")
            
    except Exception as e:
        logger.error(f"Drift Check Failed: {e}")
        mlflow.end_run()
        raise
    
    mlflow.end_run()

if __name__ == "__main__":
    main()

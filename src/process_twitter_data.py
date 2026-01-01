"""
Process Twitter dataset for viral post prediction.
Converts raw tweets into ML-ready features.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_twitter_data(input_file: str = "tweets.csv", output_dir: str = "data/processed"):
    """
    Process Twitter dataset for viral prediction.
    
    Steps:
    1. Load raw Twitter data
    2. Clean and prepare data
    3. Define virality based on engagement
    4. Save processed dataset
    
    Args:
        input_file: Path to tweets.csv
        output_dir: Directory to save processed data
    """
    print("=" * 60)
    print("🐦 Twitter Data Processing")
    print("=" * 60)
    
    # Check for Azure ML paths - try multiple possible variable names
    azure_input = (os.environ.get('AZURE_ML_INPUT_raw_data') or 
                   os.environ.get('AZURE_ML_INPUT_RAW_DATA'))
    azure_output = (os.environ.get('AZURE_ML_OUTPUT_processed_data') or 
                    os.environ.get('AZURE_ML_OUTPUT_PROCESSED_DATA'))
    
    # Debug: Print all Azure-related environment variables
    logger.info("Azure ML Environment Variables:")
    for key, value in os.environ.items():
        if 'AZURE' in key.upper() or 'OUTPUT' in key.upper():
            logger.info(f"  {key} = {value}")
    
    if azure_input:
        # Running on Azure ML
        input_file = Path(azure_input) / "tweets.csv"
        logger.info(f"✅ Running on Azure ML - input from: {input_file}")
    else:
        logger.info(f"ℹ️ Running locally - using: {input_file}")
    
    if azure_output:
        output_dir = azure_output
        logger.info(f"✅ Running on Azure ML - output to: {output_dir}")
    else:
        logger.info(f"ℹ️ Running locally - output to: {output_dir}")
    
    # Load data
    logger.info(f"Loading Twitter data from {input_file}...")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} tweets")
    
    # Initial data inspection
    logger.info("Inspecting data...")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Data types:\n{df.dtypes}")
    
    # Clean data
    logger.info("Cleaning data...")
    
    # Remove rows with missing content
    df = df.dropna(subset=['content'])
    logger.info(f"After removing empty content: {len(df)} tweets")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['content'])
    logger.info(f"After removing duplicates: {len(df)} tweets")
    
    # Handle missing engagement metrics
    df['number_of_likes'] = df['number_of_likes'].fillna(0)
    df['number_of_shares'] = df['number_of_shares'].fillna(0)
    
    # Create combined engagement score
    df['total_engagement'] = df['number_of_likes'] + (df['number_of_shares'] * 2)  # Shares weighted more
    
    # Calculate engagement percentile
    df['engagement_percentile'] = df['total_engagement'].rank(pct=True)
    
    # Define viral: top 10% of engagement
    viral_threshold_percentile = 0.90
    df['is_viral'] = (df['engagement_percentile'] >= viral_threshold_percentile).astype(int)
    
    viral_count = df['is_viral'].sum()
    viral_pct = (viral_count / len(df)) * 100
    
    logger.info(f"Viral posts: {viral_count} ({viral_pct:.1f}%)")
    logger.info(f"Engagement threshold for viral: {df[df['is_viral']==1]['total_engagement'].min():.0f}")
    
    # Parse datetime
    logger.info("Parsing datetime...")
    df['posted_at'] = pd.to_datetime(df['date_time'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['posted_at'])
    
    # Extract temporal features
    df['hour'] = df['posted_at'].dt.hour
    df['day_of_week'] = df['posted_at'].dt.dayofweek
    df['month'] = df['posted_at'].dt.month
    
    # Text features
    df['text_length'] = df['content'].str.len()
    df['word_count'] = df['content'].str.split().str.len()
    df['has_url'] = df['content'].str.contains('http', case=False, na=False).astype(int)
    df['has_hashtag'] = df['content'].str.contains('#', na=False).astype(int)
    df['has_mention'] = df['content'].str.contains('@', na=False).astype(int)
    df['question_count'] = df['content'].str.count('\?')
    df['exclamation_count'] = df['content'].str.count('!')
    
    # Rename columns to match expected schema
    df_processed = pd.DataFrame({
        'id': df.index,
        'text': df['content'],
        'posted_at': df['posted_at'],
        'likes': df['number_of_likes'].astype(int),
        'shares': df['number_of_shares'].astype(int),
        'total_engagement': df['total_engagement'].astype(int),
        'is_viral': df['is_viral'],
        'hour': df['hour'],
        'day_of_week': df['day_of_week'],
        'month': df['month'],
        'text_length': df['text_length'],
        'word_count': df['word_count'],
        'has_url': df['has_url'],
        'has_hashtag': df['has_hashtag'],
        'has_mention': df['has_mention'],
        'question_count': df['question_count'],
        'exclamation_count': df['exclamation_count']
    })
    
    # Statistics
    logger.info("\n" + "=" * 60)
    logger.info("📊 Dataset Statistics")
    logger.info("=" * 60)
    logger.info(f"Total tweets: {len(df_processed)}")
    logger.info(f"Viral tweets: {df_processed['is_viral'].sum()} ({df_processed['is_viral'].mean()*100:.1f}%)")
    logger.info(f"\nEngagement stats:")
    logger.info(df_processed['total_engagement'].describe())
    logger.info(f"\nText length stats:")
    logger.info(df_processed['text_length'].describe())
    
    # Save processed data
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"twitter_processed_{timestamp}.csv"
    
    df_processed.to_csv(output_file, index=False)
    logger.info(f"\n💾 Saved processed data to: {output_file}")
    
    print("\n" + "=" * 60)
    print("✅ Processing complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Extract NLP features: python src/extract_features_twitter.py")
    print(f"2. Train models: python src/train_model.py")
    print(f"3. Compare with synthetic data results!")
    
    return df_processed


if __name__ == "__main__":
    process_twitter_data()

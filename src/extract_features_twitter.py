"""
Extract NLP features from Twitter data.
Adapted from extract_features.py for Twitter-specific processing.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TwitterNLPFeatureExtractor:
    """Extract NLP features from Twitter data."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize feature extractor.
        
        Args:
            model_name: Sentence transformer model
        """
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        logger.info("✅ Model loaded successfully")
    
    def extract_sentiment(self, text: str) -> Tuple[float, float, str]:
        """Extract sentiment from text."""
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            label = 'positive'
        elif polarity < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return polarity, subjectivity, label
    
    def extract_text_stats(self, text: str) -> dict:
        """Extract basic text statistics."""
        text = str(text)
        words = text.split()
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': max(text.count('.') + text.count('!') + text.count('?'), 1),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'uppercase_count': sum(1 for c in text if c.isupper()),
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),
            'emoji_count': sum(1 for c in text if ord(c) > 127000)  # Rough emoji detection
        }
    
    def process_dataset(self, input_file: str, output_dir: str = "data/processed"):
        """
        Process Twitter dataset and extract all features.
        
        Args:
            input_file: Path to processed Twitter CSV
            output_dir: Directory to save features
        """
        print("=" * 60)
        print("🧠 Twitter NLP Feature Extraction")
        print("=" * 60)
        
        # Check for Azure ML output path - try multiple possible variable names
        azure_output = (os.environ.get('AZURE_ML_OUTPUT_processed_data') or 
                        os.environ.get('AZURE_ML_OUTPUT_PROCESSED_DATA'))
        
        # Debug: Print all Azure-related environment variables
        logger.info("Azure ML Environment Variables:")
        for key, value in os.environ.items():
            if 'AZURE' in key.upper() or 'OUTPUT' in key.upper():
                logger.info(f"  {key} = {value}")
        
        if azure_output:
            output_dir = azure_output
            logger.info(f"✅ Using Azure ML output path: {output_dir}")
        else:
            logger.info(f"ℹ️ Using local output path: {output_dir}")
        
        print(f"📁 Using: {Path(input_file).name}\n")
        
        # Load processed Twitter data
        logger.info(f"Loading dataset from {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"Dataset loaded: {len(df)} tweets")
        
        # Extract text features
        logger.info("Extracting text features...")
        features_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing tweets"):
            text = row['text']
            
            # Sentiment
            polarity, subjectivity, label = self.extract_sentiment(text)
            
            # Text stats
            stats = self.extract_text_stats(text)
            
            features = {
                'sentiment_polarity': polarity,
                'sentiment_subjectivity': subjectivity,
                'sentiment_label': label,
                **stats
            }
            
            features_list.append(features)
        
        # Combine features
        features_df = pd.DataFrame(features_list)
        df = pd.concat([df, features_df], axis=1)
        
        # Generate embeddings
        logger.info("Preparing text for embeddings...")
        texts = df['text'].tolist()
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create embedding features
        logger.info("Creating embedding features...")
        embedding_cols = [f'embedding_{i}' for i in range(embeddings.shape[1])]
        embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)
        
        # Combine all features
        logger.info("Combining all features...")
        df_final = pd.concat([df.reset_index(drop=True), embeddings_df], axis=1)
        
        # Save
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"twitter_features_{timestamp}.csv"
        
        df_final.to_csv(output_file, index=False)
        logger.info(f"✅ Processed dataset saved to {output_file}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Feature Extraction Summary")
        print("=" * 60)
        print(f"Total tweets: {len(df_final)}")
        print(f"Total features: {len(df_final.columns)}")
        print(f"\nFeature categories:")
        print(f"  - Original: {len(df.columns) - len(features_df.columns)}")
        print(f"  - Text stats: {len(stats)}")
        print(f"  - Sentiment: 3")
        print(f"  - Embeddings: {len(embedding_cols)}")
        
        print(f"\nSentiment distribution:")
        print(df_final['sentiment_label'].value_counts())
        
        print(f"\nText statistics:")
        print(f"  Avg words per tweet: {float(df_final['word_count'].mean()):.1f}")
        print(f"  Avg sentiment polarity: {float(df_final['sentiment_polarity'].mean()):.3f}")
        print(f"  Tweets with questions: {int((df_final['question_marks'] > 0).sum())}")
        print(f"  Tweets with exclamations: {int((df_final['exclamation_marks'] > 0).sum())}")
        print("=" * 60)
        
        print("\n✅ Feature extraction complete!")
        print(f"\nNext steps:")
        print(f"1. Train the model: python src/train_model.py")
        print(f"2. Make predictions: python src/predict.py")
        
        return df_final


def main():
    """Main execution function."""
    # Check for Azure ML input path
    azure_input = os.environ.get('AZURE_ML_OUTPUT_processed_data')
    
    if azure_input:
        # Running on Azure ML - use the output from previous script
        data_dir = Path(azure_input)
        logger.info(f"Running on Azure ML - loading from: {data_dir}")
    else:
        # Running locally
        data_dir = Path("data/processed")
    
    # Find most recent processed Twitter file
    csv_files = list(data_dir.glob("twitter_processed_*.csv"))
    
    if not csv_files:
        print(f"❌ No processed Twitter files found in {data_dir}")
        print("Please run: python src/process_twitter_data.py first")
        return
    
    # Use most recent file
    input_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    
    # Extract features
    extractor = TwitterNLPFeatureExtractor()
    extractor.process_dataset(str(input_file))


if __name__ == "__main__":
    main()

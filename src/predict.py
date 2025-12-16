"""
Test trained BERT model with custom text.
Predict if a tweet/post will go viral.
"""

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path
import json
import joblib


class ViralPredictor:
    """Make predictions with trained models."""
    
    def __init__(self):
        """Initialize predictor."""
        self.bert_model = None
        self.bert_tokenizer = None
        self.xgboost_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_bert_model(self, model_path: str):
        """Load trained BERT model."""
        print(f"Loading BERT model from {model_path}...")
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.bert_model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        print("✅ BERT model loaded")
    
    def load_xgboost_model(self, model_path: str):
        """Load trained XGBoost model."""
        print(f"Loading XGBoost model from {model_path}...")
        self.xgboost_model = joblib.load(model_path)
        print("✅ XGBoost model loaded")
    
    def predict_bert(self, text: str) -> dict:
        """
        Predict with BERT model.
        
        Args:
            text: Tweet/post text
            
        Returns:
            Dictionary with prediction and probability
        """
        # Tokenize
        encoding = self.bert_tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
        
        viral_prob = probabilities[0][1].item()
        
        return {
            'prediction': 'VIRAL' if prediction == 1 else 'NOT VIRAL',
            'viral_probability': viral_prob,
            'confidence': max(probabilities[0]).item()
        }
    
    def predict_interactive(self):
        """Interactive prediction mode."""
        print("\n" + "=" * 60)
        print("🔮 Viral Post Predictor - Interactive Mode")
        print("=" * 60)
        print("\nEnter text to predict if it will go viral!")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            text = input("📝 Enter your tweet/post: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not text:
                print("❌ Please enter some text!\n")
                continue
            
            # Predict with BERT
            if self.bert_model:
                result = self.predict_bert(text)
                
                print("\n" + "-" * 60)
                print(f"📊 Prediction: {result['prediction']}")
                print(f"🎯 Viral Probability: {result['viral_probability']*100:.1f}%")
                print(f"✅ Confidence: {result['confidence']*100:.1f}%")
                
                # Recommendation
                if result['viral_probability'] > 0.7:
                    print("💥 High chance of going viral!")
                elif result['viral_probability'] > 0.4:
                    print("⚡ Moderate viral potential")
                else:
                    print("💤 Low viral potential")
                
                print("-" * 60 + "\n")
            else:
                print("❌ No model loaded!")


def main():
    """Main execution."""
    print("=" * 60)
    print("🤖 Viral Post Predictor")
    print("=" * 60)
    
    # Find latest BERT model
    models_dir = Path("models")
    bert_dirs = list(models_dir.glob("bert_viral_classifier_*"))
    
    if not bert_dirs:
        print("\n❌ No BERT models found in models/")
        print("Please train a model first: python src/train_bert.py")
        return
    
    # Use most recent BERT model
    latest_bert = max(bert_dirs, key=lambda f: f.stat().st_mtime)
    
    # Initialize predictor
    predictor = ViralPredictor()
    predictor.load_bert_model(str(latest_bert))
    
    # Start interactive mode
    predictor.predict_interactive()


if __name__ == "__main__":
    main()

"""
Batch Prediction Script for Viral Post Predictor
Loads trained model from Azure ML and generates predictions for new data.
"""

import os
import argparse
import logging
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from pathlib import Path
import mlflow

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiTaskBERT(nn.Module):
    """BERT model with classification and regression heads."""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        
        class_logits = self.classifier(pooled_output)
        reg_output = self.regressor(pooled_output).squeeze(-1)
        
        return class_logits, reg_output

def load_model(model_dir):
    """Load model artifacts from the specified directory."""
    logger.info(f"Loading model from {model_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Locate model files - handle potentially nested Azure ML structures
    # Recursively find task_heads.pt
    model_files = list(Path(model_dir).rglob('task_heads.pt'))
    if not model_files:
        raise FileNotFoundError(f"Could not find task_heads.pt in {model_dir}")
    
    task_heads_path = model_files[0]
    base_path = task_heads_path.parent
    
    logger.info(f"Found model artifacs at: {base_path}")

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(base_path / 'bert')
    
    # Load model architecture and weights
    model = MultiTaskBERT()
    model.load_state_dict(torch.load(task_heads_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to input CSV file")
    parser.add_argument("--model_dir", type=str, help="Path to model directory")
    parser.add_argument("--output_dir", type=str, help="Path to save predictions")
    args = parser.parse_args()
    
    logger.info("🚀 Starting Batch Prediction")
    
    # Load Model
    model, tokenizer, device = load_model(args.model_dir)
    
    # Load Data
    logger.info(f"Reading input data from {args.input_data}")
    # Handle both directory (Azure ML input) and direct file
    input_path = Path(args.input_data)
    if input_path.is_dir():
        # Find csv files in directory
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
             raise FileNotFoundError(f"No CSV files found in {input_path}")
        input_file = csv_files[0]
    else:
        input_file = input_path

    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} rows")
    
    # Prepare for predictions
    predictions = []
    
    logger.info("Running inference...")
    # Process in batches
    batch_size = 32
    for i in range(0, len(df), batch_size):
        batch_texts = df['text'][i:i+batch_size].tolist()
        
        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.no_grad():
            class_logits, reg_outputs = model(input_ids, attention_mask)
            
            # Classification: Argmax for class, Softmax for probability
            probs = torch.softmax(class_logits, dim=1)
            pred_classes = torch.argmax(probs, dim=1).cpu().numpy()
            viral_probs = probs[:, 1].cpu().numpy() # Probability of class 1 (Viral)
            
            # Regression: Expm1 to reverse log1p if used during training (assuming yes based on previous code)
            # Or just raw output if no log transform. Let's assume raw or log1p based on standard practice?
            # Previous code used expm1, so we keep that consistency.
            pred_engagement = torch.expm1(reg_outputs).cpu().numpy()
            
        
        # Collect results
        for j, text in enumerate(batch_texts):
             predictions.append({
                 'text': text,
                 'is_viral_pred': bool(pred_classes[j]),
                 'viral_probability': float(viral_probs[j]),
                 'predicted_engagement': max(0, int(pred_engagement[j])) # Ensure non-negative
             })
             
    # Create result DataFrame
    result_df = pd.DataFrame(predictions)
    
    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "predictions.csv")
    result_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    main()

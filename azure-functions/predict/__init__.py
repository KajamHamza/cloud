"""
Azure Function - Viral Post Prediction API
Serverless endpoint for BERT model inference
"""

import logging
import json
import azure.functions as func
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from pathlib import Path

# Global variables for model caching
model = None
tokenizer = None
device = None

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


def load_model():
    """Load model (called on cold start)"""
    global model, tokenizer, device
    
    if model is not None:
        return  # Already loaded
    
    logging.info("Loading model (cold start)...")
    
    device = torch.device('cpu')  # Functions use CPU only
    
    # Model path
    model_path = Path(__file__).parent.parent / 'model' / 'bert-viral-classifier' / 'bert_multitask_20251221_174041'
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_path / 'bert')
    
    # Load model
    model = MultiTaskBERT()
    task_heads_path = model_path / 'task_heads.pt'
    model.load_state_dict(torch.load(task_heads_path, map_location=device))
    model.eval()
    
    logging.info("Model loaded successfully!")


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Main function handler"""
    logging.info('Prediction request received')
    
    try:
        # Load model on first request
        load_model()
        
        # Parse request
        try:
            req_body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON"}),
                status_code=400,
                mimetype="application/json"
            )
        
        text = req_body.get('text')
        if not text:
            return func.HttpResponse(
                json.dumps({"error": "No text provided"}),
                status_code=400,
                mimetype="application/json"
            )
        
        logging.info(f"Processing: {text[:50]}...")
        
        # Tokenize
        encoding = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            class_logits, reg_output = model(input_ids, attention_mask)
            
            prediction = torch.argmax(class_logits, dim=1).item()
            viral_prob = torch.softmax(class_logits, dim=1)[0][1].item()
            engagement = torch.expm1(reg_output).item()
            engagement = max(0, int(engagement))
        
        result = {
            'is_viral': bool(prediction),
            'viral_probability': round(float(viral_prob), 4),
            'predicted_engagement': engagement,
            'model_version': 'v1.0'
        }
        
        logging.info(f"Prediction: {result}")
        
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

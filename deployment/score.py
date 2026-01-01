"""
Scoring script for BERT Viral Post Predictor
Handles inference requests for the deployed Azure ML endpoint
"""

import json
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTaskBERT(nn.Module):
    """BERT model with classification and regression heads."""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Classification head (viral yes/no)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )
        
        # Regression head (engagement score)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Classification output
        class_logits = self.classifier(pooled_output)
        
        # Regression output
        reg_output = self.regressor(pooled_output).squeeze(-1)
        
        return class_logits, reg_output


def init():
    """
    Initialize the model and tokenizer.
    Called once when the endpoint starts.
    """
    global model, tokenizer, device
    
    try:
        logger.info("Initializing model...")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Get model path from environment
        model_path = os.getenv('AZUREML_MODEL_DIR')
        logger.info(f"Model path: {model_path}")
        
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(
            os.path.join(model_path, 'bert')
        )
        logger.info("Tokenizer loaded successfully")
        
        # Load model architecture
        model = MultiTaskBERT()
        
        # Load trained weights
        task_heads_path = os.path.join(model_path, 'task_heads.pt')
        model.load_state_dict(torch.load(task_heads_path, map_location=device))
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise


def run(raw_data):
    """
    Process inference requests.
    
    Input JSON format:
    {
        "text": "Your tweet text here"
    }
    
    Output JSON format:
    {
        "is_viral": true/false,
        "viral_probability": 0.85,
        "predicted_engagement": 1234
    }
    """
    try:
        # Parse input
        data = json.loads(raw_data)
        text = data.get('text', '')
        
        if not text:
            return json.dumps({"error": "No text provided"})
        
        logger.info(f"Processing text: {text[:50]}...")
        
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
            
            # Get classification prediction
            prediction = torch.argmax(class_logits, dim=1).item()
            viral_prob = torch.softmax(class_logits, dim=1)[0][1].item()
            
            # Get engagement prediction (reverse log1p transform)
            engagement = torch.expm1(reg_output).item()
            engagement = max(0, int(engagement))  # Ensure non-negative integer
        
        # Prepare response
        result = {
            'is_viral': bool(prediction),
            'viral_probability': round(float(viral_prob), 4),
            'predicted_engagement': engagement,
            'model_version': 'v1.0'
        }
        
        logger.info(f"Prediction: {result}")
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return json.dumps({"error": str(e)})

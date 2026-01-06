"""
Flask API for Viral Post Prediction
Serves the trained BERT model as a REST API on Azure App Service
"""

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import os
import logging
import json
from pathlib import Path
import pyodbc
from datetime import datetime
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None


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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        
        class_logits = self.classifier(pooled_output)
        reg_output = self.regressor(pooled_output).squeeze(-1)
        
        return class_logits, reg_output


def load_model():
    """Load the model on startup"""
    global model, tokenizer, device
    
    try:
        logger.info("Loading model...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Model path (adjust for Azure ML download structure)
        model_path = Path('model/bert-viral-classifier/bert_multitask_20251221_174041')
        bert_path = model_path / 'bert'
        
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(bert_path)
        logger.info("Tokenizer loaded")
        
        # Load model (Initialize BERT from fine-tuned path)
        model = MultiTaskBERT(model_name=str(bert_path))
        
        # Load task heads (strict=False because pt file only has heads, not full BERT)
        task_heads_path = model_path / 'task_heads.pt'
        model.load_state_dict(torch.load(task_heads_path, map_location=device), strict=False)
        
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def log_prediction_to_sql(text, is_viral, probability, working_engagement, model_version='v1.0'):
    """Log prediction to Azure SQL Database (Fire and Forget)"""
    try:
        # Get secrets from environment variables
        server = os.environ.get('SQL_SERVER', 'viral-sql-server12.database.windows.net')
        database = os.environ.get('SQL_DATABASE', 'viral_posts_db')
        username = os.environ.get('SQL_USER', 'sqladmin')
        password = os.environ.get('SQL_PASSWORD')
        
        if not password:
            logger.warning("SQL_PASSWORD not set. Skipping logging.")
            return

        # Connect
        conn_str = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
        
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            query = """
                INSERT INTO predictions (
                    input_text, 
                    is_viral_prediction, 
                    viral_probability, 
                    predicted_engagement,
                    model_version,
                    prediction_timestamp
                ) VALUES (?, ?, ?, ?, ?, GETDATE())
            """
            cursor.execute(query, (text[:4000], is_viral, probability, working_engagement, model_version))
            conn.commit()
            logger.info("✅ Prediction logged to SQL")
            
    except Exception as e:
        logger.error(f"❌ Failed to log to SQL: {str(e)}")


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Viral Post Predictor API',
        'version': 'v1.0'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Request JSON:
    {
        "text": "Your tweet text here"
    }
    
    Response JSON:
    {
        "is_viral": true,
        "viral_probability": 0.85,
        "predicted_engagement": 1234,
        "model_version": "v1.0"
    }
    """
    global model, tokenizer, device
    
    # Lazy load model if not properly initialized (e.g. running via Gunicorn)
    if model is None:
        try:
            logger.info("Model not loaded (Gunicorn worker start). Loading now...")
            load_model()
        except Exception as e:
            return jsonify({'error': f'Model load failed: {str(e)}'}), 500

    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        logger.info(f"Processing prediction for: {text[:50]}...")
        
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
            
            # Debug: Log raw regression output
            logger.info(f"Raw regression output: {reg_output}")
            logger.info(f"Regression output shape: {reg_output.shape}")
            
            engagement = torch.expm1(reg_output).item()
            logger.info(f"After expm1: {engagement}")
            engagement = max(0, int(engagement))
            logger.info(f"Final engagement: {engagement}")
        
        result = {
            'is_viral': bool(prediction),
            'viral_probability': round(float(viral_prob), 4),
            'predicted_engagement': engagement,
            'model_version': 'v1.0'
        }
        
        # Log to SQL in background thread (don't block response)
        threading.Thread(
            target=log_prediction_to_sql,
            args=(text, bool(prediction), float(viral_prob), engagement)
        ).start()
        
        logger.info(f"Prediction: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)

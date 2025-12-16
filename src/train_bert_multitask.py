"""
Multi-Task BERT Fine-Tuning for Viral Post Prediction
Trains BOTH classification (viral yes/no) AND regression (engagement prediction)
"""

import os
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
    mean_squared_error, mean_absolute_error, r2_score
)
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ViralPostDataset(Dataset):
    """Dataset for multi-task viral post prediction."""
    
    def __init__(self, texts, class_labels, reg_labels, tokenizer, max_length=128):
        """
        Args:
            texts: List of post texts
            class_labels: Binary labels (0=not viral, 1=viral)
            reg_labels: Engagement scores
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.class_labels = class_labels
        self.reg_labels = reg_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        class_label = self.class_labels[idx]
        reg_label = self.reg_labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'class_label': torch.tensor(class_label, dtype=torch.long),
            'reg_label': torch.tensor(reg_label, dtype=torch.float)
        }


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


class MultiTaskBERTTrainer:
    """Train multi-task BERT model."""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = MultiTaskBERT(model_name)
        self.model.to(self.device)
        
        self.metrics = {}
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, batch_size: int = 16):
        """Prepare data loaders."""
        logger.info("Preparing data...")
        
        # Handle both Twitter and synthetic data formats
        if 'text' in df.columns:
            texts = df['text'].tolist()
        elif 'title' in df.columns and 'body' in df.columns:
            texts = (df['title'] + " " + df['body']).tolist()
        else:
            raise ValueError("Dataset must have 'text' or 'title'+'body' columns")
        
        class_labels = df['is_viral'].astype(int).tolist()
        
        # Regression target
        if 'total_engagement' in df.columns:
            reg_labels = df['total_engagement'].tolist()
        elif 'score' in df.columns:
            reg_labels = df['score'].tolist()
        else:
            reg_labels = df.get('likes', df['is_viral']).tolist()
        
        # Normalize regression labels (log transform for better training)
        reg_labels = np.log1p(reg_labels).tolist()
        
        # Train-test split
        train_texts, test_texts, train_class, test_class, train_reg, test_reg = train_test_split(
            texts, class_labels, reg_labels, test_size=test_size, random_state=42, stratify=class_labels
        )
        
        logger.info(f"Training samples: {len(train_texts)}")
        logger.info(f"Test samples: {len(test_texts)}")
        logger.info(f"Viral posts in training: {sum(train_class)} ({sum(train_class)/len(train_class)*100:.1f}%)")
        
        # Create datasets
        train_dataset = ViralPostDataset(train_texts, train_class, train_reg, self.tokenizer)
        test_dataset = ViralPostDataset(test_texts, test_class, test_reg, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train(self, train_loader, test_loader, epochs=3, learning_rate=2e-5):
        """Train the model."""
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # Loss functions
        class_criterion = nn.CrossEntropyLoss()
        reg_criterion = nn.MSELoss()
        
        logger.info(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                class_labels = batch['class_label'].to(self.device)
                reg_labels = batch['reg_label'].to(self.device)
                
                # Forward pass
                class_logits, reg_output = self.model(input_ids, attention_mask)
                
                # Calculate losses
                class_loss = class_criterion(class_logits, class_labels)
                reg_loss = reg_criterion(reg_output, reg_labels)
                
                # Combined loss (weighted)
                loss = class_loss + 0.1 * reg_loss  # Weight regression loss less
                
                train_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'class_loss': class_loss.item(),
                    'reg_loss': reg_loss.item()
                })
            
            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")
            
            # Evaluation after each epoch
            self.evaluate(test_loader)
    
    def evaluate(self, test_loader):
        """Evaluate the model."""
        self.model.eval()
        
        class_preds = []
        class_true = []
        reg_preds = []
        reg_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                class_labels = batch['class_label'].to(self.device)
                reg_labels = batch['reg_label'].to(self.device)
                
                class_logits, reg_output = self.model(input_ids, attention_mask)
                
                class_pred = torch.argmax(class_logits, dim=1)
                
                class_preds.extend(class_pred.cpu().numpy())
                class_true.extend(class_labels.cpu().numpy())
                reg_preds.extend(reg_output.cpu().numpy())
                reg_true.extend(reg_labels.cpu().numpy())
        
        # Classification metrics
        accuracy = accuracy_score(class_true, class_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            class_true, class_preds, average='binary'
        )
        
        # Regression metrics (convert back from log space)
        reg_preds_exp = np.expm1(reg_preds)
        reg_true_exp = np.expm1(reg_true)
        
        mse = mean_squared_error(reg_true_exp, reg_preds_exp)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(reg_true_exp, reg_preds_exp)
        r2 = r2_score(reg_true_exp, reg_preds_exp)
        
        self.metrics = {
            'classification': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': confusion_matrix(class_true, class_preds).tolist()
            },
            'regression': {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2_score': float(r2),
                'mse': float(mse)
            }
        }
        
        logger.info(f"Classification - Acc: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        logger.info(f"Regression - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
    
    def save_model(self, output_dir="models"):
        """Save the trained model."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = output_dir / f"bert_multitask_{timestamp}"
        
        # Save model
        model_path.mkdir(exist_ok=True)
        self.model.bert.save_pretrained(model_path / "bert")
        self.tokenizer.save_pretrained(model_path / "bert")
        
        # Save task heads
        torch.save({
            'classifier': self.model.classifier.state_dict(),
            'regressor': self.model.regressor.state_dict()
        }, model_path / "task_heads.pt")
        
        # Save metrics
        with open(model_path / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"💾 Saved model to {model_path}")
        return timestamp
    
    def print_summary(self):
        """Print training summary."""
        print("\n" + "=" * 60)
        print("📊 Multi-Task BERT Training Summary")
        print("=" * 60)
        
        if 'classification' in self.metrics:
            print("\n🎯 Classification (Viral Yes/No)")
            print(f"  Accuracy:  {self.metrics['classification']['accuracy']:.3f}")
            print(f"  Precision: {self.metrics['classification']['precision']:.3f}")
            print(f"  Recall:    {self.metrics['classification']['recall']:.3f}")
            print(f"  F1-Score:  {self.metrics['classification']['f1_score']:.3f}")
            
            cm = np.array(self.metrics['classification']['confusion_matrix'])
            print(f"\n  Confusion Matrix:")
            print(f"                Predicted")
            print(f"              Not  Viral")
            print(f"  Actual Not  {cm[0][0]:4d} {cm[0][1]:4d}")
            print(f"         Viral {cm[1][0]:4d} {cm[1][1]:4d}")
        
        if 'regression' in self.metrics:
            print("\n📈 Regression (Engagement Prediction)")
            print(f"  RMSE:      {self.metrics['regression']['rmse']:.2f}")
            print(f"  MAE:       {self.metrics['regression']['mae']:.2f}")
            print(f"  R² Score:  {self.metrics['regression']['r2_score']:.3f}")
        
        print("=" * 60 + "\n")


def main():
    """Main execution function."""
    print("=" * 60)
    print("🤖 Multi-Task BERT Training (Classification + Regression)")
    print("=" * 60)
    
    # Find most recent features file
    data_dir = Path("data/processed")
    twitter_files = list(data_dir.glob("twitter_features_*.csv"))
    synthetic_files = list(data_dir.glob("*_features.csv"))
    
    if twitter_files:
        input_file = max(twitter_files, key=lambda f: f.stat().st_mtime)
        data_source = "Twitter"
    elif synthetic_files:
        input_file = max(synthetic_files, key=lambda f: f.stat().st_mtime)
        data_source = "Synthetic"
    else:
        print("❌ No feature files found")
        return
    
    print(f"📁 Using {data_source} data: {input_file.name}\n")
    
    # Load data
    logger.info("Loading dataset...")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} posts")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("⚠️  No GPU - training will be slower\n")
    
    # Train
    trainer = MultiTaskBERTTrainer()
    train_loader, test_loader = trainer.prepare_data(df, batch_size=16)
    trainer.train(train_loader, test_loader, epochs=3)
    
    # Save
    timestamp = trainer.save_model()
    trainer.print_summary()
    
    print("\n✅ Multi-task BERT training complete!")
    print(f"\n📦 Model: models/bert_multitask_{timestamp}/")
    print("\nYou now have:")
    print("  • Viral classification (yes/no)")
    print("  • Engagement prediction (likes/score)")


if __name__ == "__main__":
    main()

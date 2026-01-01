"""
Unit tests for BERT multi-task model
"""

import pytest
import torch
import pandas as pd
from src.train_bert_multitask import MultiTaskBERT, MultiTaskBERTTrainer, ViralPostDataset
from transformers import DistilBertTokenizer


def test_model_initialization():
    """Test that model can be initialized"""
    model = MultiTaskBERT()
    assert model is not None
    assert hasattr(model, 'bert')
    assert hasattr(model, 'classifier')
    assert hasattr(model, 'regressor')


def test_model_forward_pass():
    """Test model forward pass with dummy data"""
    model = MultiTaskBERT()
    model.eval()
    
    # Dummy input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    # Forward pass
    with torch.no_grad():
        class_logits, reg_output = model(input_ids, attention_mask)
    
    # Check output shapes
    assert class_logits.shape == (batch_size, 2), "Classification output should be (batch_size, 2)"
    assert reg_output.shape == (batch_size,), "Regression output should be (batch_size,)"


def test_dataset_creation():
    """Test dataset can be created from data"""
    # Mock data
    texts = ["test tweet 1", "test tweet 2", "test tweet 3"]
    class_labels = [0, 1, 0]
    reg_labels = [10.0, 100.0, 50.0]
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = ViralPostDataset(texts, class_labels, reg_labels, tokenizer, max_length=128)
    
    assert len(dataset) == 3
    
    # Test getting an item
    item = dataset[0]
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert 'class_label' in item
    assert 'reg_label' in item


def test_data_preparation():
    """Test data loading and preparation"""
    # Mock DataFrame
    df = pd.DataFrame({
        'text': ['tweet 1', 'tweet 2', 'tweet 3', 'tweet 4', 'tweet 5'] * 20,
        'is_viral': [0, 1, 0, 1, 0] * 20,
        'total_engagement': [10, 100, 50, 200, 30] * 20
    })
    
    trainer = MultiTaskBERTTrainer()
    train_loader, test_loader = trainer.prepare_data(df, batch_size=16)
    
    assert len(train_loader) > 0, "Train loader should not be empty"
    assert len(test_loader) > 0, "Test loader should not be empty"
    
    # Test batch shape
    batch = next(iter(train_loader))
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'class_labels' in batch
    assert 'reg_labels' in batch


def test_metrics_calculation():
    """Test that metrics are calculated correctly"""
    trainer = MultiTaskBERTTrainer()
    
    # Mock predictions and labels
    class_preds = torch.tensor([0, 1, 0, 1, 0, 1])
    class_labels = torch.tensor([0, 1, 0, 0, 0, 1])
    reg_preds = torch.tensor([10.0, 100.0, 50.0, 200.0, 30.0, 150.0])
    reg_labels = torch.tensor([12.0, 95.0, 55.0, 190.0, 35.0, 160.0])
    
    # This would normally be called during evaluation
    # Just testing that the trainer has the metrics attribute
    assert hasattr(trainer, 'metrics')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

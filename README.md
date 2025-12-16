# Viral Post Predictor - Synthetic Data Approach

## 🚀 Quick Start

### 1. Setup (one time)
```bash
cd c:\Users\hamza\Desktop\cloud
.\setup.bat
```

### 2. Generate Data (~2 minutes)
```bash
venv\Scripts\activate
python src\generate_data.py
```

### 3. Extract NLP Features (~5 minutes)
```bash
python src\extract_features.py
```

### 4. Train Model (coming next)
```bash
python src\train_model.py
```

---

## 📊 What Each Script Does

**generate_data.py**: Creates 20K synthetic posts
- Realistic titles & bodies
- Engagement metrics (score, comments, awards)
- 15 topics (tech, gaming, AI, etc.)
- Viral labels (top 10%)

**extract_features.py**: NLP feature extraction
- Sentiment analysis (positive/negative/neutral)
- Text embeddings (384-dim vectors)
- Text statistics (length, punctuation, etc.)
- Combined features ready for ML

**train_model.py**: Train viral predictor (next step)
- Classification: viral yes/no
- Regression: upvote prediction
- Model evaluation metrics

---

## 🎯 Project Status

✅ Data generation - DONE  
🔄 NLP features - IN PROGRESS  
⏳ Model training - NEXT  
⏳ Prediction API - TODO  
⏳ Power BI dashboard - TODO (main goal)

---

## 🔧 Customize

Edit `config/generator_config.json`:
- Number of posts
- Topics
- Virality threshold
- Engagement patterns

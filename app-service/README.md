---
title: Viral Post Predictor
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Viral Post Predictor API 🚀

This is a REST API for predicting if a social media post will go viral using a fine-tuned BERT model.

## API Usage

**Endpoint:** `/predict`  
**Method:** `POST`

### Request Format
```json
{
  "text": "Your tweet text here"
}
```

### Response Format
```json
{
  "is_viral": true,
  "viral_probability": 0.85,
  "predicted_engagement": 1234,
  "model_version": "v1.0"
}
```

## Local Testing
```bash
docker build -t viral-predictor .
docker run -p 7860:7860 viral-predictor
```

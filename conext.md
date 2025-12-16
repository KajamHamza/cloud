# **Viral Predictor – AI that Predicts Reddit/X Post Virality Before Publishing + already existing posts potential**

Team :

Ranya EL Amrani

El Rhirhayi Taha

El Kajam Hamza

### **Goal**

And  (Main)

A live dashboard showing:

- Top predicted viral posts
- Model accuracy over time
- Real vs predicted upvotes
- Trending subreddits and posting times
- System health and usage metrics

Create a web tool where anyone can type a Reddit post (title + body) or a X post and instantly get: (Optional)

- **How likely the post is to go viral**
- **An estimate of how many upvotes it might get**
- **Three smart tips to improve it** (when to post, wording, keywords, tone, etc.)

**Data Sources (100% free)**

- Historical dataset: Reddit posts 2015–2023 from Kaggle
- Real-time enrichment: Reddit PRAW (free) or X/Twitter Basic v2 API (free tier)

X: https://docs.x.com/x-api/introduction

If data seems not that Great or harder to obtain we would generate our own data and expose it via an API

Steps :

**Data Ingestion**

Download Reddit dataset → store raw files in **Azure Blob Storage**. 

Lightweight Azure ML notebook script that pulls 50–100 new posts daily and saves them to the same Blob.

1. **Storage**
    - Raw data → Azure Blob Storage
    - Cleaned data, NLP embeddings, predictions → **Azure Cosmos DB for NoSQL (Serverless mode)**
2. **Data Processing – Batch**
    
    **Azure Data Factory** daily lightweight pipeline: cleans data, extracts basic NLP features, writes everything to Cosmos DB.
    
3. **Streaming**
    
    **Azure Event Hubs** (free tier) + **Azure Stream Analytics**: continuously computes best posting times and hottest subreddits → outputs written to Cosmos DB every 5 minutes.
    
4. **Data Balancing**
    
    Target “viral” = top 5% of upvotes → SMOTE + undersampling applied during training.
    
5. **Model Training & NLP**
    
    We might use 2 models (NLP +Regression)
    
6. Deployment 
7. **Inference – User Interface**
    
    Fully built with **Microsoft Power Pages** (professional public website).
    
8. **CI/CD**
    
    **Azure DevOps Pipelines** (free): on every commit → automatic retraining + endpoint redeployment if performance improves.
    
9. **Monitoring & Alerting**
    
    **Application Insights** + Azure Monitor alert (email if prediction latency > 3 seconds).
    
10. **Security & Governance**
    - Single Resource Group
    - RBAC restricted
    - API keys stored in **Azure Key Vault**
    - Access logs enabled
11. **Dashboard – Power BI ( Main Goal)**
    - Top 10 predicted viral posts (last 24 h)
    - Model accuracy over time
    - Real vs predicted upvotes (scatter plot)
    - Live metrics from Stream Analytics (best times/subreddits)
    - Global KPIs (data volume, predictions served, error rate)
    
12. **Power Pages – App (Optional)**
    
    Public web page where the user:
    
    - Types title + body of their future post
    - Selects subreddit and planned posting time
    - Clicks “Predict Virality”
    - Instantly sees: probability gauge, estimated upvotes, 3 personalized tips
    Direct call to the Azure ML endpoint from Power Pages.
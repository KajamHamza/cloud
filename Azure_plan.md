# 🚀 Azure Deployment Plan - Viral Post Predictor
## Step-by-Step Guide (Budget: $115/month)

---

## ✅ **Pre-Deployment Checklist**

- [ ] Azure account with $200 credits
- [ ] Azure CLI installed (`winget install Microsoft.AzureCLI`)
- [ ] Logged in: `az login`
- [ ] tweets.csv file ready (52K tweets)
- [ ] Trained models in `models/` directory

---

## 📝 **Setup Variables**

Run these commands first (PowerShell):

```powershell
# Core variables
$RESOURCE_GROUP = "viral-predictor-rg"
$LOCATION = "eastus"  # Cheapest region
$STORAGE_ACCOUNT = "viralpredictstorage7767"
$SQL_SERVER = "viral-sql-server12"
$SQL_DB = "viral_posts_db"
$ML_WORKSPACE = "viral-ml-workspace"
$APP_NAME = "viral-predictor-app$(Get-Random -Max 9999)"

# SQL Password (change this!)
$SQL_PASSWORD = "aszd12@!EFRG12@!"

echo "✅ Variables set!"
```

---

## 🔧 **PHASE 1: Foundation (Day 1)**

### Step 1: Create Resource Group
```powershell
az group create --name $RESOURCE_GROUP --location $LOCATION
```
- [x] Resource group created

---

### Step 2: Create Storage Account
```powershell
# Create storage account
az storage account create `
  --name $STORAGE_ACCOUNT `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --sku Standard_LRS

# Create containers
az storage container create --name raw-data --account-name $STORAGE_ACCOUNT --auth-mode login
az storage container create --name processed-data --account-name $STORAGE_ACCOUNT --auth-mode login
az storage container create --name models --account-name $STORAGE_ACCOUNT --auth-mode login
```
- [x] Storage account created
- [x] Containers created

---

### Step 3: Upload Raw tweets.csv
```powershell
# Upload RAW tweets.csv (unprocessed)
az storage blob upload `
  --account-name $STORAGE_ACCOUNT `
  --container-name raw-data `
  --name tweets.csv `
  --file .\tweets.csv `
  --auth-mode login

# Verify
az storage blob list --account-name $STORAGE_ACCOUNT --container-name raw-data --output table

# Upload processing scripts
az storage blob upload `
  --account-name $STORAGE_ACCOUNT `
  --container-name raw-data `
  --name process_twitter_data.py `
  --file .\src\process_twitter_data.py `
  --auth-mode key

az storage blob upload `
  --account-name $STORAGE_ACCOUNT `
  --container-name raw-data `
  --name extract_features_twitter.py `
  --file .\src\extract_features_twitter.py `
  --auth-mode key
```
- [x] tweets.csv uploaded (RAW data)
- [x] Processing scripts uploaded
- [x] Upload verified

---

### Step 4: Create Azure SQL Database
```powershell
# Create SQL Server
az sql server create `
  --name $SQL_SERVER `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --admin-user sqladmin `
  --admin-password $SQL_PASSWORD

# Allow Azure services
az sql server firewall-rule create `
  --resource-group $RESOURCE_GROUP `
  --server $SQL_SERVER `
  --name AllowAzureServices `
  --start-ip-address 0.0.0.0 `
  --end-ip-address 0.0.0.0

# Create database
az sql db create `
  --resource-group $RESOURCE_GROUP `
  --server $SQL_SERVER `
  --name $SQL_DB `
  --service-objective Basic `
  --max-size 2GB
```
- [x] SQL Server created
- [x] Database created

---

### Step 5: Create SQL Tables

**⚠️ MANUAL STEP:** Run this SQL in Azure Portal

1. Go to Azure Portal → SQL databases → `viral_posts_db`
2. Click "Query editor" (login with sqladmin + your password)
3. Copy and paste this SQL:

```sql
CREATE TABLE predictions (
    id INT PRIMARY KEY IDENTITY(1,1),
    tweet_text NVARCHAR(MAX),
    created_at DATETIME DEFAULT GETDATE(),
    is_viral BIT,
    viral_probability FLOAT,
    predicted_engagement INT,
    actual_likes INT,
    actual_shares INT
);

CREATE INDEX idx_created ON predictions(created_at);
```
- [ ] Tables created

**Cost so far:** $10/month (Storage + SQL)

---

## 🤖 **PHASE 2: Machine Learning (Day 2)**

### Step 6: Create ML Workspace
```powershell
# Install ML extension
az extension add -n ml -y

# Create workspace
az ml workspace create `
  --name $ML_WORKSPACE `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION
```
- [x] ML workspace created

---

### Step 7: Process Data on Azure ML (GPU)

**Create data processing job** to run on Azure GPU:

Create `process-data-job.yml`:
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command: |
  python process_twitter_data.py
  python extract_features_twitter.py
code: ./src
environment: azureml:viral-env:1
compute: gpu-cluster
experiment_name: data_processing
display_name: process-twitter-data
inputs:
  raw_data:
    type: uri_file
    path: azureml://datastores/workspaceblobstore/paths/raw-data/tweets.csv
outputs:
  processed_data:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/processed-data
```

```powershell
# Create GPU compute (used for BOTH processing and training)
az ml compute create `
  --name gpu-cluster `
  --type AmlCompute `
  --resource-group $RESOURCE_GROUP `
  --workspace-name $ML_WORKSPACE `
  --size Standard_DS3_v2 `
  --min-instances 0 `
  --max-instances 1 `
  --idle-time-before-scale-down 1800
  
# to keep using your storage account, you need to register it as a datastore first

az ml datastore create --file datastore.yml --resource-group $RESOURCE_GROUP --workspace-name $ML_WORKSPACE
# Submit processing job
az ml job create --file process-data-job.yml -g $RESOURCE_GROUP -w $ML_WORKSPACE

# Monitor job
az ml job list -g $RESOURCE_GROUP -w $ML_WORKSPACE --output table

# Get job status (replace JOB_NAME with actual name from list)
az ml job show -n JOB_NAME -g $RESOURCE_GROUP -w $ML_WORKSPACE
```

**This will:**
1. Download tweets.csv from Blob Storage
2. Run `process_twitter_data.py` (clean, define viral)
3. Run `extract_features_twitter.py` (NLP features, GPU-accelerated embeddings)
4. Save processed features back to Blob Storage

**Time:** ~10 minutes (GPU-accelerated)  
**Cost:** ~$0.15 (one-time)

- [x] GPU compute created
- [x] Processing job submitted
- [x] Processing completed (~10 mins)
- [x] Processed data in Blob Storage

---

### Step 8: Submit Training Job (Same GPU)
**Training uses the same GPU cluster created in Step 7**

---

### Step 9: Create Training Environment
Create `environment.yml`:
```yaml
name: viral-env
channels:
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
    - torch
    - transformers
    - xgboost
    - scikit-learn
    - pandas
    - numpy
    - azure-storage-blob
```

```powershell
# Create environment
az ml environment create `
  --name viral-env `
  --conda-file environment.yml `
  --image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04 `
  --resource-group $RESOURCE_GROUP `
  --workspace-name $ML_WORKSPACE
```
- [x] Environment created

---

### Step 10: Submit Training Job

Create `training-job.yml`:
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command: >
  python train_bert_multitask.py
code: ./src
environment: azureml:viral-env:1
compute: gpu-cluster
experiment_name: viral_prediction
display_name: bert-multitask-training
inputs:
  processed_data:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/processed-data
outputs:
  model_output:
    type: uri_folder
```

**⚠️ MANUAL STEP:** Update `train_bert_multitask.py` to load from Azure:
```python
# Edit src/train_bert_multitask.py - modify main() function:
def main():
    # Load data from Azure ML input
    input_path = os.environ.get('AZURE_ML_INPUT_processed_data', './data/processed')
    csv_files = list(Path(input_path).glob("twitter_features_*.csv"))
    # ... rest of code unchanged
```

```powershell
# Upload training code
mkdir azure-ml-training
cp -r src azure-ml-training/
cp training-job.yml azure-ml-training/

# Submit job
cd azure-ml-training
az ml job create --file training-job.yml -g $RESOURCE_GROUP -w $ML_WORKSPACE
cd ..
```

**Monitor training:**
```powershell
# List jobs
az ml job list -g $RESOURCE_GROUP -w $ML_WORKSPACE --output table

# Stream logs (replace JOB_NAME)
az ml job stream -n JOB_NAME -g $RESOURCE_GROUP -w $ML_WORKSPACE
```
- [x] Training job submitted
- [x] Training completed (~15-20 mins)

**Phase 2 Cost:**
- GPU processing: $0.15 (one-time)
- GPU training: $0.40 (one-time)
- **Total:** $0.55 one-time

---

## 🚀 **PHASE 3: Deployment (Day 3)**

### Step 11: Register Trained Model
```powershell
# Register BERT model
az ml model create `
  --name bert-viral-classifier `
  --version 1 `
  --path ./models/bert_multitask_* `
  --type custom_model `
  --resource-group $RESOURCE_GROUP `
  --workspace-name $ML_WORKSPACE
```
- [x] Model registered

---

### Step 12: Download Model and Create Flask API

**⚠️ Note:** Using Azure App Service instead of managed online endpoints due to ACI quota limitations.

```powershell
# Download model from Azure ML
az ml model download `
  --name bert-viral-classifier `
  --version 1 `
  --download-path ./app-service/model `
  --resource-group $RESOURCE_GROUP `
  --workspace-name $ML_WORKSPACE
```

Flask API already created in `app-service/` folder:
- `app.py` - Flask API with prediction endpoint
- `requirements.txt` - Dependencies
- `DEPLOYMENT.md` - Full deployment guide

- [ ] Model downloaded
- [ ] Flask API files created

---

### Step 13: Deploy to Azure App Service

```powershell
# Create App Service plan (B1 tier - $13/month)
az appservice plan create `
  --name viral-api-plan `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --sku B1 `
  --is-linux

# Create web app
$APP_NAME = "viral-predictor-api-$((Get-Random -Max 9999))"
az webapp create `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --plan viral-api-plan `
  --runtime "PYTHON:3.11"

# Configure startup command
az webapp config set `
  --resource-group $RESOURCE_GROUP `
  --name $APP_NAME `
  --startup-file "gunicorn --bind=0.0.0.0:8000 --timeout 600 app:app"

# Package and deploy
cd app-service
Compress-Archive -Path * -DestinationPath ../api.zip -Force
cd ..

az webapp deployment source config-zip `
  --resource-group $RESOURCE_GROUP `
  --name $APP_NAME `
  --src api.zip
```

- [ ] App Service plan created
- [ ] Web app created
- [ ] Application deployed

**Deployment time:** ~5-10 minutes  
**Cost:** ~$13/month (B1 tier)

---

### Step 14: Test API Endpoint

```powershell
# Get app URL
$APP_URL = az webapp show `
  --resource-group $RESOURCE_GROUP `
  --name $APP_NAME `
  --query defaultHostName -o tsv

# Test health check
curl "https://$APP_URL/"

# Test prediction
curl -X POST "https://$APP_URL/predict" `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"Breaking: Major AI breakthrough announced! 🚀\"}'
```

**Expected response:**
```json
{
  "is_viral": true,
  "viral_probability": 0.8521,
  "predicted_engagement": 1234,
  "model_version": "v1.0"
}
```

- [ ] Health check successful
- [ ] Prediction endpoint working


**Cost so far:** $85/month (Storage + SQL + ML Endpoint)

---

## 🌐 **PHASE 4: Streamlit Interface (Day 4)**

### Step 15: Create Streamlit App
Create `app.py`:
```python
import streamlit as st
import requests
import json

st.set_page_config(page_title="Viral Post Predictor", page_icon="🔮")

st.title("🔮 Viral Post Predictor")
st.write("Predict if your tweet/post will go viral!")

# Input
text = st.text_area("Enter your tweet/post:", height=100, 
                    placeholder="What's on your mind?")

if st.button("🚀 Predict", type="primary"):
    if text:
        # Call Azure ML endpoint
        endpoint_url = st.secrets["ENDPOINT_URL"]
        api_key = st.secrets["API_KEY"]
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {"text": text}
        
        with st.spinner("Analyzing..."):
            response = requests.post(endpoint_url, headers=headers, json=data)
            result = response.json()
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Viral?", "YES ✅" if result['is_viral'] else "NO ❌")
        
        with col2:
            st.metric("Viral Probability", f"{result['viral_probability']*100:.1f}%")
        
        with col3:
            st.metric("Expected Engagement", f"{result['predicted_engagement']:,}")
        
        # Recommendation
        if result['viral_probability'] > 0.7:
            st.success("💥 High viral potential! This could go viral!")
        elif result['viral_probability'] > 0.4:
            st.info("⚡ Moderate viral potential")
        else:
            st.warning("💤 Low viral potential - consider revising")
    else:
        st.error("Please enter some text!")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("This app predicts viral potential using BERT + XGBoost models trained on 52K tweets.")
    st.metric("Model F1-Score", "59.3%")
    st.metric("Model Accuracy", "92.8%")
```

Create `.streamlit/secrets.toml`:
```toml
ENDPOINT_URL = "https://viral-prediction-endpoint.eastus.inference.ml.azure.com/score"
API_KEY = "your-api-key-here"
```

Create `requirements.txt`:
```
streamlit
requests
```
- [ ] Streamlit app created

---

### Step 16: Deploy to Azure App Service
```powershell
# Create App Service plan
az appservice plan create `
  --name viral-app-plan `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --sku B1 `
  --is-linux

# Create web app
az webapp create `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --plan viral-app-plan `
  --runtime "PYTHON:3.11"

# Deploy app
zip -r app.zip app.py .streamlit requirements.txt

az webapp deployment source config-zip `
  --resource-group $RESOURCE_GROUP `
  --name $APP_NAME `
  --src app.zip

# Set startup command
az webapp config set `
  --resource-group $RESOURCE_GROUP `
  --name $APP_NAME `
  --startup-file "python -m streamlit run app.py --server.port=8000 --server.address=0.0.0.0"
```

**Access app:**
```
https://$APP_NAME.azurewebsites.net
```
- [ ] Streamlit deployed
- [ ] App accessible

**Cost:** $100/month (+ $15 for App Service)

---

## 📊 **PHASE 5: Power BI Dashboard (Day 5)**

### Step 17: Create Power BI Embedded
```powershell
az powerbi embedded-capacity create `
  --resource-group $RESOURCE_GROUP `
  --name viraldashboard `
  --location $LOCATION `
  --sku-name A1 `
  --sku-tier PBIE_Azure
```
- [ ] Power BI capacity created

---

### Step 18: Build Dashboard

**⚠️ MANUAL STEP:** Create Power BI Dashboard (Requires Power BI Desktop)

**Download:** https://powerbi.microsoft.com/desktop/

**In Power BI Desktop:**

1. **Connect to Azure SQL:**
   - Get Data → Azure SQL Database
   - Server: `$SQL_SERVER.database.windows.net`
   - Database: `viral_posts_db`
   - Credentials: sqladmin / your password

2. **Create Visualizations:**
   - **Card:** Total Predictions
   - **Gauge:** Average Viral Probability
   - **Line Chart:** Predictions Over Time
   - **Bar Chart:** Viral vs Non-Viral Count
   - **Table:** Recent Predictions

3. **Publish:**
   - File → Publish → Publish to Power BI
   - Sign in with Azure account
   - Select workspace

- [ ] Dashboard created
- [ ] Dashboard published

**Cost:** $115/month (+ $10 for Power BI)

---

## 🔒 **PHASE 6: Security & Monitoring (Day 6)**

### Step 19: Create Key Vault
```powershell
# Create Key Vault
az keyvault create `
  --name viral-keyvault$(Get-Random -Max 999) `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION

# Store secrets
az keyvault secret set --vault-name viral-keyvault --name MLEndpointKey --value "YOUR_API_KEY"
az keyvault secret set --vault-name viral-keyvault --name SQLPassword --value $SQL_PASSWORD
```
- [ ] Key Vault created
- [ ] Secrets stored

---

### Step 20: Setup Monitoring
```powershell
# Create Application Insights
az monitor app-insights component create `
  --app viral-insights `
  --location $LOCATION `
  --resource-group $RESOURCE_GROUP `
  --application-type web

# Connect to ML endpoint
az ml online-endpoint update `
  --name viral-prediction-endpoint `
  --resource-group $RESOURCE_GROUP `
  --workspace-name $ML_WORKSPACE `
  --app-insights viral-insights

# Create alert (high latency)
az monitor metrics alert create `
  --name HighLatencyAlert `
  --resource-group $RESOURCE_GROUP `
  --scopes $(az monitor app-insights component show -g $RESOURCE_GROUP --app viral-insights --query id -o tsv) `
  --condition "avg response_time > 2000" `
  --description "Alert when API latency > 2s"
```
- [ ] Application Insights configured
- [ ] Alerts created

---

## 🔄 **PHASE 7: CI/CD (Day 7)**

### Step 21: Setup GitHub Actions

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Azure

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Deploy Streamlit App
        run: |
          zip -r app.zip app.py .streamlit requirements.txt
          az webapp deployment source config-zip \
            --resource-group viral-predictor-rg \
            --name ${{ secrets.APP_NAME }} \
            --src app.zip
```

**Setup:**
```powershell
# Create service principal
az ad sp create-for-rbac `
  --name "viral-predictor-sp" `
  --role contributor `
  --scopes /subscriptions/YOUR_SUB_ID/resourceGroups/$RESOURCE_GROUP `
  --sdk-auth
```

**Copy output to GitHub Secrets:**
- Secret name: `AZURE_CREDENTIALS`
- Value: Full JSON output

- [ ] GitHub Actions configured
- [ ] Auto-deployment working

---

## 🤖 **PHASE 8: MLOps Enhancements (Day 8)** 

### Step 22: Add MLflow Experiment Tracking

**Update training script to track experiments** (FREE):

Add to `src/train_bert_multitask.py`:
```python
import mlflow
import mlflow.pytorch

def main():
    # Start MLflow run
    mlflow.set_tracking_uri("azureml://")
    mlflow.set_experiment("viral_prediction")
    
    with mlflow.start_run(run_name="bert_multitask_training"):
        # Log parameters
        mlflow.log_params({
            "model": "distilbert-base-uncased",
            "epochs": 3,
            "learning_rate": 2e-5,
            "batch_size": 16
        })
        
        # ... existing training code ...
        
        # Log metrics after evaluation
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "rmse": rmse,
            "r2_score": r2
        })
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        print(f"✅ MLflow tracking: {mlflow.active_run().info.run_id}")
```

**View experiments in Azure ML Studio:**
- Go to portal.azure.com → ML Workspace → Experiments
- Compare runs, visualize metrics

- [ ] MLflow tracking added
- [ ] Experiments visible in Azure ML

**Cost:** $0 (included in ML workspace)

---

### Step 23: Setup Data Drift Detection

**Monitor data quality changes** (FREE):

Create `monitoring/data_drift.py`:
```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import DataDriftMonitor
from azure.identity import DefaultAzureCredential

# Initialize client
ml_client = MLClient.from_config(DefaultAzureCredential())

# Create data drift monitor
monitor = DataDriftMonitor(
    name="viral-data-drift-monitor",
    compute="gpu-cluster",
    monitoring_signals={
        "data_drift_signal": {
            "type": "data_drift",
            "production_data": "azureml:viralstorage_processed:1",
            "reference_data": "azureml:viralstorage_processed:1",
            "features": ["text_length", "word_count", "sentiment_polarity"],
            "alert_enabled": True
        }
    },
    schedule={
        "frequency": "week",
        "interval": 1
    }
)

# Create monitor
ml_client.schedules.create_or_update(monitor)
print("✅ Data drift monitoring enabled")
```

```powershell
# Run setup
python monitoring/data_drift.py
```

- [ ] Data drift monitor created
- [ ] Weekly drift checks enabled

**Cost:** $0 (uses existing compute, runs 5 min/week)

---

### Step 24: Automated Retraining Pipeline

**Schedule weekly model retraining** (~$2/month):

Create `pipelines/retrain_pipeline.py`:
```python
from azure.ai.ml import dsl, Input, Output
from azure.ai.ml.entities import PipelineJob

@dsl.pipeline(name="automated_retraining")
def retrain_pipeline():
    # Step 1: Process new data
    process_step = process_data_component(
        raw_data=Input(path="azureml://datastores/viralstorage/paths/tweets.csv")
    )
    
    # Step 2: Train model
    train_step = train_model_component(
        processed_data=process_step.outputs.processed_data
    )
    
    # Step 3: Evaluate model
    eval_step = evaluate_model_component(
        model=train_step.outputs.model,
        test_data=process_step.outputs.test_data
    )
    
    # Step 4: Deploy if better
    deploy_step = conditional_deploy_component(
        new_model=train_step.outputs.model,
        metrics=eval_step.outputs.metrics,
        threshold=0.85  # Deploy if accuracy > 85%
    )
    
    return {
        "trained_model": train_step.outputs.model,
        "metrics": eval_step.outputs.metrics
    }

# Create pipeline job
pipeline = retrain_pipeline()

# Schedule weekly execution
from azure.ai.ml.entities import CronSchedule

schedule = CronSchedule(
    expression="0 2 * * 0",  # Sunday 2 AM
    name="weekly_retrain"
)

ml_client.schedules.create_or_update(
    schedule=schedule,
    pipeline_job=pipeline
)
```

- [ ] Retraining pipeline created
- [ ] Weekly schedule configured

**Cost:** ~$2/month (30 min CPU time/week)

---

### Step 25: Model Performance Monitoring

**Track prediction accuracy in SQL** (FREE):

Update SQL schema:
```sql
-- Add performance tracking columns
ALTER TABLE predictions 
ADD model_version VARCHAR(50) DEFAULT 'v1.0',
    prediction_timestamp DATETIME DEFAULT GETDATE(),
    feedback_correct BIT NULL,
    feedback_timestamp DATETIME NULL;

-- Create performance view
CREATE VIEW model_performance AS
SELECT 
    model_version,
    DATE(prediction_timestamp) as prediction_date,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN feedback_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
    AVG(CASE WHEN feedback_correct IS NOT NULL 
        THEN CAST(feedback_correct AS FLOAT) 
        ELSE NULL END) as accuracy
FROM predictions
WHERE feedback_correct IS NOT NULL
GROUP BY model_version, DATE(prediction_timestamp);

-- Create monitoring alert trigger
CREATE TRIGGER alert_low_accuracy
ON predictions
AFTER INSERT
AS
BEGIN
    DECLARE @recent_accuracy FLOAT;
    
    SELECT @recent_accuracy = AVG(CAST(feedback_correct AS FLOAT))
    FROM predictions
    WHERE feedback_timestamp > DATEADD(day, -7, GETDATE())
    AND feedback_correct IS NOT NULL;
    
    IF @recent_accuracy < 0.80
    BEGIN
        -- Log alert (implement notification if needed)
        PRINT 'WARNING: Model accuracy below 80%';
    END
END;
```

Update `score.py` to log predictions:
```python
def run(raw_data):
    # ... existing prediction code ...
    
    # Log to SQL
    import pyodbc
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO predictions 
        (tweet_text, is_viral, viral_probability, predicted_engagement, model_version)
        VALUES (?, ?, ?, ?, ?)
    """, (text, prediction, viral_prob, engagement, "v1.0"))
    
    conn.commit()
    return result
```

- [ ] Performance tracking schema created
- [ ] Prediction logging enabled
- [ ] Performance dashboard view created

**Cost:** $0 (uses existing SQL database)

---

### Step 26: Enhanced GitHub Actions with Testing

**Add automated testing** (FREE on GitHub):

Update `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Azure with Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install pytest pytest-cov
          pip install -r requirements.txt
      
      - name: Run unit tests
        run: pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Deploy Streamlit App
        run: |
          zip -r app.zip app.py .streamlit requirements.txt
          az webapp deployment source config-zip \
            --resource-group viral-predictor-rg \
            --name ${{ secrets.APP_NAME }} \
            --src app.zip
      
      - name: Run integration tests
        run: |
          python tests/test_endpoint.py

  retrain-check:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v3
      
      - name: Check if retraining needed
        run: |
          python monitoring/check_performance.py
          # Trigger retraining pipeline if accuracy drops
```

Create `tests/test_model.py`:
```python
import pytest
from src.train_bert_multitask import MultiTaskBERT, MultiTaskBERTTrainer

def test_model_initialization():
    """Test model can be initialized"""
    model = MultiTaskBERT()
    assert model is not None

def test_model_forward_pass():
    """Test model forward pass"""
    import torch
    model = MultiTaskBERT()
    
    # Dummy input
    input_ids = torch.randint(0, 1000, (1, 128))
    attention_mask = torch.ones((1, 128))
    
    class_logits, reg_output = model(input_ids, attention_mask)
    
    assert class_logits.shape == (1, 2)
    assert reg_output.shape == (1,)

def test_trainer_data_preparation():
    """Test data loading"""
    import pandas as pd
    
    # Mock data
    df = pd.DataFrame({
        'text': ['test tweet'] * 100,
        'is_viral': [0, 1] * 50,
        'total_engagement': range(100)
    })
    
    trainer = MultiTaskBERTTrainer()
    train_loader, test_loader = trainer.prepare_data(df, batch_size=16)
    
    assert len(train_loader) > 0
    assert len(test_loader) > 0
```

- [ ] Unit tests created
- [ ] GitHub Actions testing enabled
- [ ] Code coverage tracking setup

**Cost:** $0 (GitHub Actions free tier)

---

### Step 27: MLOps Dashboard

**Create monitoring dashboard** (Uses existing Power BI):

Add to Power BI dashboard:

1. **Model Performance Tile**:
   - Data source: SQL `model_performance` view
   - Visual: Line chart of accuracy over time
   - Alert: Red if accuracy < 80%

2. **Data Drift Tile**:
   - Data source: Azure ML data drift metrics
   - Visual: Gauge showing drift percentage
   - Alert: Warning if drift > 15%

3. **Pipeline Health Tile**:
   - Data source: Azure ML pipeline runs
   - Visual: Success/Failure count
   - Alert: Red if last 3 runs failed

4. **Prediction Volume Tile**:
   - Data source: SQL predictions table
   - Visual: Daily prediction count
   - Trend: 7-day moving average

- [ ] MLOps dashboard created
- [ ] Monitoring tiles configured
- [ ] Alerts enabled

**Cost:** $0 (uses existing Power BI capacity)

---

## ✅ **Final Verification Checklist**

**Core Infrastructure:**
- [ ] tweets.csv uploaded to Blob Storage
- [ ] SQL Database tables created
- [ ] ML model trained successfully (F1: 59.3%)
- [ ] Model endpoint responding
- [ ] Streamlit app accessible
- [ ] Power BI dashboard published
- [ ] Secrets in Key Vault
- [ ] Monitoring active
- [ ] CI/CD pipeline working

**MLOps Components:**
- [ ] MLflow experiment tracking enabled
- [ ] Data drift monitoring configured
- [ ] Automated retraining pipeline scheduled
- [ ] Model performance tracking in SQL
- [ ] GitHub Actions tests passing
- [ ] MLOps dashboard tiles configured

---

## 💰 **Final Cost Summary**

| Service | Monthly Cost | Notes |
|---------|--------------|-------|
| Blob Storage | $5 | 100GB data |
| Azure SQL (Basic) | $5 | Processed predictions + MLOps metrics |
| **App Service (B1)** | **$13** | **Flask API (replaced ML Endpoint)** |
| App Service (B1) | $15 | Streamlit app |
| Power BI (A1) | $10 | Dashboard + MLOps monitoring |
| Key Vault | $3 | Secrets |
| Application Insights | $0 | Free tier |
| **MLOps Components:** | | |
| MLflow Tracking | $0 | Included in ML workspace |
| Data Drift Detection | $0 | <5 min/week on existing compute |
| Automated Retraining | $2 | 30 min CPU/week |
| Performance Monitoring | $0 | Uses existing SQL |
| GitHub Actions Testing | $0 | Free tier (2000 min/month) |
| **Monthly Total** | **$53** | |
| | | |
| **One-Time Costs:** | | |
| GPU Processing | $0.15 | Data processing (~10 mins) |
| GPU Training | $0.40 | BERT training (~15 mins) |
| **One-Time Total** | **$0.55** | |
| | | |
| **Month 1 Grand Total** | **$53.55** | **$146 under budget!** ✅ |

**Cost per month after setup:** Only $53/month (with full MLOps capabilities!)

---

## 🎯 **Access Your Deployment**

**Streamlit App:**
```
https://$APP_NAME.azurewebsites.net
```

**ML Endpoint:**
```
https://viral-prediction-endpoint.eastus.inference.ml.azure.com/score
```

**Power BI:**
- Login to app.powerbi.com
- View published dashboard

**Azure Portal:**
```
https://portal.azure.com → Resource Group: viral-predictor-rg
```

---

## 🚨 **Troubleshooting**

### ML Endpoint not responding
```powershell
# Check logs
az ml online-deployment get-logs `
  --name blue `
  --endpoint-name viral-prediction-endpoint `
  -g $RESOURCE_GROUP `
  -w $ML_WORKSPACE
```

### Streamlit app not loading
```powershell
# Check app logs
az webapp log tail --name $APP_NAME -g $RESOURCE_GROUP
```

### High costs
```powershell
# Stop ML endpoint when not in use
az ml online-endpoint delete --name viral-prediction-endpoint -g $RESOURCE_GROUP -w $ML_WORKSPACE

# Scale down app service
az appservice plan update --name viral-app-plan -g $RESOURCE_GROUP --sku FREE
```

---

## 🎉 **Project Complete!**

You now have a **full production ML pipeline**:
✅ Data storage and processing
✅ BERT model (F1: 59.3%) 
✅ REST API endpoint
✅ Web interface (Streamlit)
✅ Business dashboard (Power BI)
✅ CI/CD automation
✅ Monitoring & alerts
✅ Security (Key Vault)

**All 14 requirements met!** Under $115/month budget! 🚀

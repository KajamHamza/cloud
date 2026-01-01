# Azure App Service Deployment Guide

## Prerequisites
- Trained model downloaded from Azure ML
- Azure CLI installed and logged in

## Step 1: Download Model from Azure ML

```powershell
# Download the trained model
az ml model download `
  --name bert-viral-classifier `
  --version 1 `
  --download-path ./app-service/model `
  --resource-group $RESOURCE_GROUP `
  --workspace-name $ML_WORKSPACE
```

## Step 2: Create App Service Plan

```powershell
# Create App Service plan (B1 tier - $13/month)
az appservice plan create `
  --name viral-api-plan `
  --resource-group $RESOURCE_GROUP `
  --location eastus `
  --sku B1 `
  --is-linux
```

## Step 3: Create Web App

```powershell
# Create the web app
az webapp create `
  --name viral-predictor-api-$((Get-Random -Max 9999)) `
  --resource-group $RESOURCE_GROUP `
  --plan viral-api-plan `
  --runtime "PYTHON:3.11"
```

## Step 4: Configure Startup Command

```powershell
# Set the startup command  
az webapp config set `
  --resource-group $RESOURCE_GROUP `
  --name <your-app-name> `
  --startup-file "gunicorn --bind=0.0.0.0:8000 --timeout 600 app:app"
```

## Step 5: Deploy Application

```powershell
# Zip the application
cd app-service
Compress-Archive -Path * -DestinationPath ../api.zip -Force
cd ..

# Deploy to App Service
az webapp deployment source config-zip `
  --resource-group $RESOURCE_GROUP `
  --name <your-app-name> `
  --src api.zip
```

## Step 6: Test the API

```powershell
# Get the app URL
$APP_URL = az webapp show `
  --resource-group $RESOURCE_GROUP `
  --name <your-app-name> `
  --query defaultHostName -o tsv

# Test health check
curl "https://$APP_URL/"

# Test prediction
curl -X POST "https://$APP_URL/predict" `
  -H "Content-Type: application/json" `
  -d '{"text": "Breaking: Major AI breakthrough announced! 🚀"}'
```

## Expected Response

```json
{
  "is_viral": true,
  "viral_probability": 0.8521,
  "predicted_engagement": 1234,
  "model_version": "v1.0"
}
```

## Cost

- **B1 App Service:** ~$13/month
- **Storage + SQL:** ~$10/month
- **Total:** ~$23/month

## Monitoring

View logs:
```powershell
az webapp log tail `
  --resource-group $RESOURCE_GROUP `
  --name <your-app-name>
```

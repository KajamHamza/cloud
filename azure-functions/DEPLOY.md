# Azure Functions Deployment Guide

## Step 1: Copy Model

```powershell
# Copy downloaded model to functions folder
Copy-Item -Path "./app-service/model" -Destination "./azure-functions/model" -Recurse
```

## Step 2: Create Function App

```powershell
# Create storage account for function app
az storage account create `
  --name viralfunctionstorage `
  --resource-group viral-predictor-rg `
  --location eastus `
  --sku Standard_LRS

# Create Functions app (Consumption plan - no quota needed!)
az functionapp create `
  --name viral-predictor-func-$((Get-Random -Max 9999)) `
  --resource-group viral-predictor-rg `
  --storage-account viralfunctionstorage `
  --consumption-plan-location eastus `
  --runtime python `
  --runtime-version 3.11 `
  --functions-version 4 `
  --os-type Linux
```

## Step 3: Deploy Function

```powershell
cd azure-functions

# Package and deploy
func azure functionapp publish <your-function-app-name>
```

## Step 4: Test

```powershell
# Get function URL
$FUNC_URL = az functionapp show `
  --name <your-function-app-name> `
  --resource-group viral-predictor-rg `
  --query defaultHostName -o tsv

# Test prediction
curl -X POST "https://$FUNC_URL/api/predict" `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"Breaking: AI breakthrough! 🚀\"}'
```

## Important Notes

**Cold Start:** First request after idle takes 30-60 seconds (loading model)

**Memory:** Consumption plan = 1.5 GB (might struggle with BERT)

**Timeout:** 230 seconds max

**Cost:** ~$0.20 per million requests (nearly free!)

## Alternative: Premium Plan

If memory/cold start is an issue:

```powershell
az functionapp plan create `
  --name viral-func-premium `
  --resource-group viral-predictor-rg `
  --location eastus `
  --sku EP1 `
  --is-linux
```

Cost: ~$150/month but pre-warmed (no cold starts)

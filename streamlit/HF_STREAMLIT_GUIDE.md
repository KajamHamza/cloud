# 🚀 Deploying Dashboard to Hugging Face

We will create a **second Space** for your UI (just like we did for the API).

## Step 1: Create New Space
1.  Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2.  **Name**: `viral-predictor-dashboard`
3.  **SDK**: Select **Docker** (Crucial for SQL drivers!).
4.  **Hardware**: Free (2 vCPU).
5.  **Create Space**.

## Step 2: Set Secrets (Important!)
Go to **Settings** -> **Variables and secrets** -> **New secret**:
- Name: `SQL_PASSWORD`
- Value: [Your Azure SQL Password]

## Step 3: Push Code
Open terminal in `c:\Users\hamza\Desktop\cloud\streamlit`:

```powershell
# 1. Initialize git
git init -b main

# 2. Add remote (Replace USERNAME)
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/viral-predictor-dashboard

# 3. Push
git add .
git commit -m "Deploy Dashboard"
git push origin main --force
```

## Step 4: Visit Live App
Your dashboard will be at: `https://YOUR_USERNAME-viral-predictor-dashboard.hf.space`

# 🚀 Deploying to Hugging Face Spaces

This `app-service` folder is designed to be a **standalone repository**.

## Step 1: Create a Space
1.  Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2.  **Name**: `viral-post-predictor` (or similar).
3.  **SDK**: Select **Docker**.
4.  **Hardware**: Select **Default (Free)** (2 vCPU, 16GB RAM).
5.  Click **Create Space**.

## Step 2: Initialize Git and Push
Open your terminal in this folder (`c:\Users\hamza\Desktop\cloud\app-service`):

```powershell
# 1. Initialize a new git repo for just this folder
git init -b main

# 2. Add the remote (Copy from your HF Space page!)
# It will look like: https://huggingface.co/spaces/YOUR_USERNAME/viral-post-predictor
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/viral-post-predictor

# 3. Setup Large File Storage (Critical for the 250MB+ model!)
git lfs install
git lfs track "model/**"
git add .gitattributes

# 4. Add files
git add .

# 5. Commit
git commit -m "Initial deploy of Viral Predictor API"

# 6. Push to deploy!
git push origin main --force
```

## Step 3: Wait for Build
1.  Go to your Space URL.
2.  Click **Logs** to watch the build.
3.  Once "Running", your API is live!

---

## 🔗 Using the API
Your API URL will be: `https://YOUR_USERNAME-viral-post-predictor.hf.space/predict`

**Test it:**
```bash
curl -X POST "https://YOUR_USERNAME-viral-post-predictor.hf.space/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Breaking news: AI takes over the cloud! 🚀"}'
```

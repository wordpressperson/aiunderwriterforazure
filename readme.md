```markdown
# AI Underwriter Pro: Azure Deployment Guide 🛡️

This repository contains an AI-powered insurance underwriting application built with Streamlit, LangChain, and Hugging Face. This guide provides the definitive workflow for deploying the application to Azure App Service, specifically addressing the challenges of handling multi-gigabyte model weights in a cloud environment.

## 📋 Project Overview

The AI Underwriter Pro analyzes insurance applications against underwriting guidelines using a hybrid approach:

- **Local Processing**: Uses a local BART model for document summarization to reduce API costs.
- **LLM Reasoning**: Uses OpenAI GPT-3.5 for the final risk cross-referencing and analysis.

## 🛠️ Prerequisites

- Python 3.11+
- OpenAI API Key (for risk analysis)
- Hugging Face Access Token (Read-only, for model downloads)
- Azure Subscription (Basic B3 or Standard/Premium tier required for RAM/Disk requirements)

## 🚀 Azure Deployment: The "Golden Path"

Standard deployments often fail due to Azure's 230-second load balancer timeout and Hugging Face IP rate limits. Follow these steps to ensure a successful deployment.

### 1. Resource Provisioning

- **Publish**: Code
- **Runtime stack**: Python 3.11
- **Operating System**: Linux
- **Plan Tier**: Start with **Premium V3 (P2v3)**

> **Note**: Heavy models like BART require significant temporary disk space for extraction. Basic tiers (10 GB) will fail during the initial download.

### 2. Configuration & Environment Variables

In the Azure Portal, navigate to **Settings > Environment variables** and add the following:

| Variable        | Value                    | Purpose                                      |
|-----------------|--------------------------|----------------------------------------------|
| `OPENAI_API_KEY` | `your_key`              | Authenticates GPT-3.5 Analysis               |
| `HF_TOKEN`       | `your_token`            | Bypasses anonymous rate limits               |
| `HF_HOME`        | `/home/huggingface_cache` | Redirects cache to writable storage        |
| `HF_ENDPOINT`    | `https://hf-mirror.com` | Bypasses Azure IP blocks via mirror          |

**Startup Command**  
In **Configuration > General settings**, set the Startup Command to:

```bash
python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0
```

### 3. The SSH Manual Download (Critical Step)

To avoid the 230-second web timeout, you **must** download the models via the SSH backend.

1. Open **Development Tools > SSH > Go** in the Azure Portal.
2. Run the following commands:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/huggingface_cache
export HF_TOKEN="your_huggingface_token"

python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/bart-large-cnn', token=os.environ['HF_TOKEN'])"
```

3. Wait for the progress bars to reach 100%.
4. **Restart the App Service**.

### 4. Post-Deployment Scaling

Once the models are successfully cached in `/home`, you can safely scale down to a **Standard (S2 or S3)** tier to save costs.  
Enable **"Always On"** in the settings to keep the models loaded in memory.

## 🔍 Troubleshooting & Post-Mortem

### Issue: 429 Client Error (Too Many Requests)
**Cause**: Azure shared NAT IPs are often rate-limited by Hugging Face's security layer (CloudFront).  
**Solution**: Use the `https://hf-mirror.com` endpoint and ensure you are authenticated with a `HF_TOKEN`.

### Issue: OSError: We couldn't connect...
**Cause**: Azure's load balancer cuts any HTTP request longer than 230 seconds. Large model downloads (1.6 GB+) exceed this limit.  
**Solution**: Use the SSH Terminal for the initial download (not subject to HTTP timeout rules).

### Issue: No space left on device
**Cause**: The 10 GB storage on Basic/B2 tiers is insufficient for downloading, unzipping, and caching large model weights simultaneously.  
**Solution**: Provision a Premium tier (250 GB+ storage) for the initial setup, then scale down once the persistent cache is established.

## 📦 Dependencies (requirements.txt)

```plaintext
streamlit
pandas
openai
transformers==4.38.1
huggingface-hub
langchain
langchain-community
langchain-text-splitters
pypdf
torch
torchvision
```

---

**✅ Ready to deploy!**  
Follow the "Golden Path" above and your AI Underwriter Pro will be live on Azure App Service with full model caching and cost-optimized scaling.
```

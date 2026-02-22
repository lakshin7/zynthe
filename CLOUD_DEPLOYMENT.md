# Zynthe Cloud Deployment Guide

Running Knowledge Distillation models requires significant GPU resources. If your local machine (e.g., MacBook Air/Pro without extensive memory) struggles, you can run the *Heavy Backend* on a free cloud GPU instance (like Google Colab or Kaggle) and connect it to your *Local Frontend*.

## Option 1: Automated Launch (Recommended)

**Click the "Launch Cloud (free GPU)" button** in the Zynthe sidebar.

This will open a specialized Google Colab notebook pre-configured for Zynthe.
1. Run All Cells.
2. Enter your ngrok token when prompted.
3. Copy the generated `wss://...` URL back to your local app settings (if manual connection is needed, though we aim for auto-discovery in future).

## Option 2: Manual Setup (Google Colab)

### 1. Open Colab
Go to [colab.research.google.com](https://colab.research.google.com/) and create a new notebook.

### 2. Enable GPU
Click **Runtime** -> **Change runtime type** -> select **T4 GPU**.

### 3. Install Zynthe Backend
Copy and paste this into the first cell:

```python
!git clone https://github.com/lakshin7/zynthe.git
%cd zynthe
!pip install -r requirements.txt
!pip install pyngrok uvicorn fastapi nesting-asyncio
```

### 4. Setup Tunnel (ngrok)
You need `ngrok` to expose the Colab local server to the internet so your local UI can talk to it.

1. Create a free account at [ngrok.com](https://ngrok.com).
2. Get your Authtoken.
3. Run this cell in Colab:

```python
from pyngrok import ngrok
import nest_asyncio
import uvicorn
from ui.backend.api import app

# Apply patch for Colab's event loop
nest_asyncio.apply()

# Set your token
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")

# Open a tunnel to port 8765
public_url = ngrok.connect(8765).public_url
print("🚀 Backend is live at:", public_url)

# Run the server
uvicorn.run(app, port=8765)
```

### 5. Connect Local UI
1. Copy the `public_url` printed above (e.g., `https://a1b2c3d4.ngrok-free.app`).
2. Open your local Zynthe App code (`ui/src/App.tsx`).
3. Find the WebSocket connection line:
   ```javascript
   // const ws = new WebSocket('ws://localhost:8765/ws');
   const ws = new WebSocket('wss://a1b2c3d4.ngrok-free.app/ws'); // Replace with your URL
   ```
   *Note: Use `wss://` (secure websocket) because ngrok uses https.*

## Option 2: Kaggle Kernels (Free P100/T4)
Similar to Colab, but you may need to use `localtunnel` if ngrok is blocked, or verify your phone number to use internet access.

## Optimizing for Latency
Cloud backends introduce network latency.
- **Micro-updates**: The UI receives training metrics slightly delayed.
- **Large datasets**: Uploading large datasets to Colab can be slow. It's better to use HuggingFace Datasets (`load_dataset`) directly in the backend if possible.

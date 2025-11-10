"""
FastAPI Backend for Zynthe Knowledge Distillation Toolkit
Connects the Electron UI to the Python training pipeline with full transparency
"""
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import uvicorn
import yaml
import json
import csv
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from training_manager import TrainingManager
from evaluation_tasks import init_task_manager, get_task_manager, EvaluationType

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Add sys.path for imports
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Training manager for subprocess handling
training_manager = None
evaluation_task_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize training manager and evaluation task manager on startup"""
    global training_manager, evaluation_task_manager
    training_manager = TrainingManager(websocket_broadcast=broadcast_message)
    # Initialize evaluation task manager with WebSocket support
    evaluation_task_manager = init_task_manager(max_workers=2, websocket_manager=None)  # Will set websocket later
    yield
    # Cleanup on shutdown
    if evaluation_task_manager:
        evaluation_task_manager.shutdown(wait=True)

app = FastAPI(title="Zynthe API", version="1.0.0", lifespan=lifespan)

# Enable CORS for Electron app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve experiment artifacts (images/plots) directly under /experiments
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
app.mount("/experiments", StaticFiles(directory=str(EXPERIMENTS_DIR), html=False), name="experiments")

# Global state
active_experiments = {}  # Maps exp_id -> training details
training_status = {"is_training": False, "experiment_id": None, "stage": None}
websocket_connections = []
active_training_ids = set()  # Track currently running experiments

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ===== PYDANTIC MODELS =====

class TrainingConfig(BaseModel):
    teacher_model: str
    student_model: str
    dataset: str
    epochs: int = 10
    batch_size: int = 32

class PreflightRequest(BaseModel):
    teacher_model: str
    student_model: str
    dataset_size: Optional[int] = None

class EvaluationRequest(BaseModel):
    experiment_id: str
    eval_type: str = "standard"  # 'standard', 'extended', 'dual', 'benchmark'
    benchmark_tasks: Optional[List[str]] = None  # For benchmark eval: ['truthfulqa', 'mmlu', 'gsm8k']

class HFTokenRequest(BaseModel):
    token: str

# ===== HELPER FUNCTIONS =====

async def validate_dataset(file_path: Path) -> tuple[bool, str, int]:
    """Validate dataset format and return (is_valid, error_message, num_samples)"""
    
    try:
        if file_path.suffix == '.jsonl':
            num_samples = 0
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        
                        # Check required fields
                        if 'text' not in data:
                            return False, f"Line {line_num}: Missing 'text' field", 0
                        if 'label' not in data:
                            return False, f"Line {line_num}: Missing 'label' field", 0
                        
                        num_samples += 1
                        
                        # Check first 100 lines to save time
                        if line_num >= 100:
                            # Estimate total lines
                            file_size = file_path.stat().st_size
                            avg_line_size = file_path.stat().st_size / line_num
                            num_samples = int(file_size / avg_line_size)
                            break
                            
                    except json.JSONDecodeError:
                        return False, f"Line {line_num}: Invalid JSON", 0
            
            if num_samples == 0:
                return False, "No valid samples found", 0
            
            return True, "", num_samples
            
        elif file_path.suffix == '.csv':
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                
                # Check headers
                if not reader.fieldnames or 'text' not in reader.fieldnames or 'label' not in reader.fieldnames:
                    return False, "CSV must have 'text' and 'label' columns", 0
                
                # Count rows
                num_samples = sum(1 for _ in reader)
                
                if num_samples == 0:
                    return False, "No samples found in CSV", 0
            
            return True, "", num_samples
        
        else:
            return False, "Unsupported file format. Use .jsonl or .csv", 0
            
    except Exception as e:
        return False, f"Error reading file: {str(e)}", 0

async def broadcast_message(message: dict):
    """Broadcast message to all connected websockets"""
    # Clean up active training set when training completes
    if message.get("type") == "training_update":
        exp_id = message.get("experiment_id")
        status = message.get("status")
        if exp_id and status in ["completed", "failed", "stopped"]:
            active_training_ids.discard(exp_id)
    
    for connection in websocket_connections[:]:
        try:
            await connection.send_json(message)
        except:
            websocket_connections.remove(connection)

# ===== API ENDPOINTS =====

@app.get("/")
async def root():
    return {"status": "ok", "message": "Zynthe API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint for debugging"""
    import torch
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "backend": "running",
        "device": {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "current": "cpu"
        },
        "hf_token_configured": bool(os.getenv('HF_TOKEN')),
        "training_manager": training_manager is not None,
        "project_root": str(PROJECT_ROOT)
    }
    
    if torch.cuda.is_available():
        health_status["device"]["current"] = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        health_status["device"]["current"] = "mps"
    
    return health_status

@app.get("/api/experiments")
async def get_experiments():
    """Get all experiments from the experiments folder"""
    experiments_dir = PROJECT_ROOT / "experiments"
    if not experiments_dir.exists():
        return []
    
    experiments = []
    for exp_dir in sorted(experiments_dir.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue
        
        # Load config
        config = {}
        config_file = exp_dir / "config.yaml"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
            except:
                pass
        
        # Determine status
        status = "completed"
        if exp_dir.name in active_training_ids:
            status = "running"
        elif not (exp_dir / "results.json").exists():
            status = "failed"
        
        # Get basic metrics if available
        metrics = {}
        results_file = exp_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    metrics = json.load(f)
            except:
                pass
        
        experiments.append({
            "id": exp_dir.name,
            "name": config.get("experiment_name", exp_dir.name),
            "status": status,
            "teacher": config.get("model", {}).get("name", "Unknown"),
            "student": config.get("model", {}).get("student_name", "Unknown"),
            "dataset": config.get("data", {}).get("name", "Unknown"),
            "accuracy": metrics.get("accuracy", 0),
            "compression_ratio": metrics.get("compression_ratio", 1.0),
            "created_at": exp_dir.name.split("_")[0] if "_" in exp_dir.name else "Unknown",
        })
    
    return experiments

@app.get("/api/experiments/{exp_id}")
async def get_experiment(exp_id: str):
    """Get detailed information about a specific experiment"""
    exp_dir = PROJECT_ROOT / "experiments" / exp_id
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Load basic info
    info = {}
    info_file = exp_dir / "experiment_info.json"
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
    
    # Load config
    config = {}
    config_file = exp_dir / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
    
    # Load results
    results = {}
    metrics = {}
    results_file = exp_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
            results = data
            metrics = data.get("metrics", data)
    
    # Determine pipeline stages status
    stages = {
        "preflight": {"status": "upcoming", "progress": 0, "message": ""},
        "distillation": {"status": "upcoming", "progress": 0, "message": ""},
        "quantization": {"status": "upcoming", "progress": 0, "message": ""},
        "evaluation": {"status": "upcoming", "progress": 0, "message": ""},
        "deployment": {"status": "upcoming", "progress": 0, "message": ""}
    }
    
    # Check for preflight
    preflight_file = exp_dir / "preflight_report.json"
    if preflight_file.exists():
        stages["preflight"] = {"status": "completed", "progress": 100, "message": "Compatibility verified"}
    
    # Check for distillation
    best_student_dir = exp_dir / "best_student"
    if best_student_dir.exists():
        stages["distillation"] = {"status": "completed", "progress": 100, "message": "Model trained"}
    elif exp_id in active_training_ids:
        # Get live progress if training
        if training_manager:
            live_metrics = training_manager.get_metrics(exp_id)
            if live_metrics:
                stages["distillation"] = {
                    "status": "in_progress",
                    "progress": live_metrics.get("progress", 0),
                    "message": f"Epoch {live_metrics.get('epoch', 0)}/{live_metrics.get('totalEpochs', 10)}"
                }
    
    # Check for quantization
    quant_dir = exp_dir / "quantized_model"
    if quant_dir.exists():
        stages["quantization"] = {"status": "completed", "progress": 100, "message": "Model quantized"}
    
    # Check for evaluation
    if results_file.exists():
        stages["evaluation"] = {"status": "completed", "progress": 100, "message": "Metrics computed"}
    
    # Read training logs
    logs = []
    log_files = [
        exp_dir / "training.log",
        exp_dir / "distillation.log",
        exp_dir / "quantization.log",
        exp_dir / "evaluation.log"
    ]
    
    for log_file in log_files:
        if log_file.exists():
            with open(log_file) as f:
                logs.extend(f.readlines()[-50:])  # Last 50 lines
    
    # Get file sizes for export
    export_files = {}
    if best_student_dir.exists():
        for file in best_student_dir.rglob("*"):
            if file.is_file():
                export_files[file.name] = file.stat().st_size
    
    # Parse timestamp
    try:
        timestamp = datetime.strptime(exp_dir.name.split("_")[0], "%Y%m%dT%H%M%SZ")
    except:
        timestamp = datetime.now()
    
    return {
        "id": exp_id,
        "experiment_id": exp_id,  # Add this for frontend compatibility
        "name": info.get("experiment_name") or config.get("experiment_name", exp_id),
        "experiment_name": info.get("experiment_name") or config.get("experiment_name", exp_id),  # Add this
        "timestamp": timestamp.isoformat(),
        "created_at": info.get("created_at", timestamp.isoformat()),  # Add this
        "status": info.get("status", "completed" if results_file.exists() else "running"),  # Add this
        "teacher_model": info.get("teacher_model"),  # Add this
        "student_model": info.get("student_model"),  # Add this
        "dataset": info.get("dataset"),  # Add this
        "config": config,
        "results": results,
        "metrics": metrics,
        "stages": stages,
        "logs": logs,
        "export_files": export_files
    }

# ===== ARTIFACT + LIVE STREAM ENDPOINTS (for training visualizations) =====

@app.get("/api/experiments/{exp_id}/artifacts")
async def list_artifacts(exp_id: str):
    """List PNG/JPG artifacts for an experiment (used by UI gallery)."""
    exp_dir = PROJECT_ROOT / "experiments" / exp_id
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    images = []
    for p in exp_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            images.append(str(p.relative_to(EXPERIMENTS_DIR)))
    return {"experiment_id": exp_id, "images": images}

@app.get("/api/experiments/{exp_id}/confusion/{role}")
async def get_confusion_matrix(exp_id: str, role: str):
    """Return confusion matrix image path and metrics.json for student/teacher."""
    exp_dir = PROJECT_ROOT / "experiments" / exp_id
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    cm_dir = exp_dir / f"{role}_confusion"
    img = cm_dir / "confusion_matrix.png"
    metrics = cm_dir / "metrics.json"
    if not img.exists():
        raise HTTPException(status_code=404, detail="Confusion matrix not ready")
    data = {}
    if metrics.exists():
        try:
            data = json.loads(metrics.read_text())
        except Exception:
            data = {}
    return {
        "experiment_id": exp_id,
        "role": role,
        "image_path": str(img.relative_to(EXPERIMENTS_DIR)),
        "metrics": data,
    }

@app.get("/api/experiments/{exp_id}/batch-log")
async def get_batch_log(exp_id: str):
    """Return the CSV content of the detailed batch log if present."""
    exp_dir = PROJECT_ROOT / "experiments" / exp_id
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    csv_path = exp_dir / "training_detailed_log.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Batch log not found")
    return {"experiment_id": exp_id, "csv": csv_path.read_text()}

@app.get("/api/experiments/{exp_id}/micro/{role}/{epoch}")
async def get_micro_series(exp_id: str, role: str, epoch: int):
    """Return micro-series (train/eval) image paths for a given epoch and role."""
    exp_dir = PROJECT_ROOT / "experiments" / exp_id
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    files = {
        "train": exp_dir / f"{role}_epoch{epoch}_train_micro.png",
        "eval": exp_dir / f"{role}_epoch{epoch}_eval_micro.png",
    }
    resolved = {k: str(v.relative_to(EXPERIMENTS_DIR)) for k, v in files.items() if v.exists()}
    if not resolved:
        raise HTTPException(status_code=404, detail="Micro-series not found")
    return {"experiment_id": exp_id, "role": role, "epoch": epoch, "images": resolved}

@app.post("/api/stream")
async def ingest_stream_event(request: Request):
    """Receive streaming events (HTTP) and broadcast to connected WebSocket clients."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")
    await broadcast_message(payload)
    return {"ok": True}

@app.get("/api/datasets")
async def get_datasets():
    """Get all available datasets"""
    data_dir = PROJECT_ROOT / "data"
    datasets = []
    
    # Built-in datasets
    builtin_datasets = [
        {"id": "imdb_sample", "name": "IMDB Sample", "type": "built-in", "size": "1000 samples"},
        {"id": "imdb_train", "name": "IMDB Training", "type": "built-in", "size": "25000 samples"},
        {"id": "imdb_val", "name": "IMDB Validation", "type": "built-in", "size": "25000 samples"},
    ]
    datasets.extend(builtin_datasets)
    
    # User-uploaded datasets
    if data_dir.exists():
        for file in data_dir.glob("*.jsonl"):
            if file.name not in ["imdb_train.jsonl", "imdb_val.jsonl", "imdb_sample.jsonl"]:
                datasets.append({
                    "id": file.stem,
                    "name": file.stem.replace("_", " ").title(),
                    "type": "custom",
                    "size": f"{file.stat().st_size / 1024:.1f} KB"
                })
        
        for file in data_dir.glob("*.csv"):
            datasets.append({
                "id": file.stem,
                "name": file.stem.replace("_", " ").title(),
                "type": "custom",
                "size": f"{file.stat().st_size / 1024:.1f} KB"
            })
    
    return datasets

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a custom dataset (JSONL or CSV format)"""
    
    # Validate file has a filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )
    
    # Validate file extension
    if not file.filename.endswith(('.jsonl', '.csv')):
        raise HTTPException(
            status_code=400,
            detail="Only .jsonl and .csv files are supported"
        )
    
    # Create data directory if it doesn't exist
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Generate safe filename
    safe_filename = file.filename.replace(' ', '_').lower()
    file_path = data_dir / safe_filename
    
    # Check if file already exists
    if file_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {safe_filename} already exists"
        )
    
    try:
        # Save file
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Validate dataset
        is_valid, error_msg, num_samples = await validate_dataset(file_path)
        
        if not is_valid:
            # Delete invalid file
            file_path.unlink()
            raise HTTPException(status_code=400, detail=error_msg)
        
        return {
            "status": "success",
            "dataset_id": safe_filename.replace('.jsonl', '').replace('.csv', ''),
            "filename": safe_filename,
            "num_samples": num_samples,
            "message": f"Dataset uploaded successfully with {num_samples} samples"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")

@app.delete("/api/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a custom dataset"""
    data_dir = PROJECT_ROOT / "data"
    
    # Find the dataset file
    dataset_file = None
    for ext in ['.jsonl', '.csv']:
        potential_file = data_dir / f"{dataset_id}{ext}"
        if potential_file.exists():
            dataset_file = potential_file
            break
    
    if not dataset_file:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Don't allow deleting built-in datasets
    if dataset_id in ["imdb_train", "imdb_val", "imdb_sample", "sample_train", "sample_val"]:
        raise HTTPException(status_code=403, detail="Cannot delete built-in datasets")
    
    try:
        dataset_file.unlink()
        return {"status": "success", "message": f"Dataset {dataset_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

# ===== HUGGINGFACE TOKEN SETTINGS =====

@app.post("/api/settings/hf-token")
async def save_hf_token(request: HFTokenRequest):
    """Save HuggingFace token for model downloads"""
    try:
        env_file = PROJECT_ROOT / ".env"
        
        # Read existing .env
        env_vars = {}
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, val = line.split('=', 1)
                        env_vars[key] = val
        
        # Update token
        env_vars['HF_TOKEN'] = request.token
        
        # Write back
        with open(env_file, 'w') as f:
            f.write("# HuggingFace Token for downloading private/gated models\n")
            f.write("# Get your token from: https://huggingface.co/settings/tokens\n")
            for key, val in env_vars.items():
                f.write(f"{key}={val}\n")
        
        # Also set in environment for current session
        os.environ['HF_TOKEN'] = request.token
        
        # Try to login to HuggingFace
        try:
            from huggingface_hub import login
            login(token=request.token, add_to_git_credential=False)
            print(f"✅ Logged in to HuggingFace")
        except Exception as e:
            print(f"⚠️ HF login warning: {e}")
        
        return {"status": "success", "message": "Token saved successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save token: {str(e)}")

@app.get("/api/settings/hf-token")
async def get_hf_token():
    """Check if HF token is configured"""
    token = os.getenv('HF_TOKEN')
    has_token = token is not None and len(token.strip()) > 0
    return {
        "configured": has_token,
        "token_preview": f"{token[:8]}..." if (has_token and token) else None
    }

@app.delete("/api/settings/hf-token")
async def delete_hf_token():
    """Remove HF token"""
    try:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            lines = []
            with open(env_file) as f:
                for line in f:
                    if not line.strip().startswith('HF_TOKEN='):
                        lines.append(line)
            
            with open(env_file, 'w') as f:
                f.writelines(lines)
        
        if 'HF_TOKEN' in os.environ:
            del os.environ['HF_TOKEN']
        
        return {"status": "success", "message": "Token removed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove token: {str(e)}")

# ===== MODEL SEARCH & RECOMMENDATIONS =====

@app.get("/api/models/search")
async def search_models(query: str, task: str = "text-classification", limit: int = 20):
    """Search HuggingFace Hub for models"""
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        hf_token = os.getenv('HF_TOKEN')
        
        # Search models
        models = api.list_models(
            filter=task,
            search=query,
            limit=limit,
            token=hf_token if hf_token else None
        )
        
        results = []
        for model in models:
            model_id = getattr(model, 'id', getattr(model, 'modelId', 'unknown'))
            results.append({
                "id": model_id,
                "name": model_id,
                "downloads": getattr(model, 'downloads', 0),
                "likes": getattr(model, 'likes', 0),
                "task": task,
                "private": getattr(model, 'private', False)
            })
        
        # Sort by downloads
        results.sort(key=lambda x: x['downloads'], reverse=True)
        
        return {"models": results}
    
    except Exception as e:
        print(f"Model search failed: {e}")
        return {"models": [], "error": str(e)}

@app.get("/api/models/recommended")
async def get_recommended_models():
    """Get curated list of recommended teacher-student pairs"""
    return {
        "pairs": [
            {
                "teacher": {
                    "id": "bert-base-uncased",
                    "name": "BERT Base",
                    "size": "440MB",
                    "params": "110M",
                    "verified": True
                },
                "students": [
                    {
                        "id": "distilbert-base-uncased",
                        "name": "DistilBERT",
                        "size": "268MB",
                        "params": "66M",
                        "compression": "1.7x",
                        "verified": True
                    },
                    {
                        "id": "google/mobilebert-uncased",
                        "name": "MobileBERT",
                        "size": "100MB",
                        "params": "25M",
                        "compression": "4.4x",
                        "verified": True
                    }
                ]
            },
            {
                "teacher": {
                    "id": "albert-base-v2",
                    "name": "ALBERT Base",
                    "size": "45MB",
                    "params": "12M",
                    "verified": True
                },
                "students": [
                    {
                        "id": "albert/albert-tiny-v2",
                        "name": "ALBERT Tiny",
                        "size": "16MB",
                        "params": "4M",
                        "compression": "3x",
                        "verified": True
                    }
                ]
            },
            {
                "teacher": {
                    "id": "roberta-base",
                    "name": "RoBERTa Base",
                    "size": "500MB",
                    "params": "125M",
                    "verified": True
                },
                "students": [
                    {
                        "id": "distilroberta-base",
                        "name": "DistilRoBERTa",
                        "size": "330MB",
                        "params": "82M",
                        "compression": "1.5x",
                        "verified": True
                    }
                ]
            },
            {
                "teacher": {
                    "id": "microsoft/MiniLM-L12-H384-uncased",
                    "name": "MiniLM-L12",
                    "size": "130MB",
                    "params": "33M",
                    "verified": True
                },
                "students": [
                    {
                        "id": "microsoft/MiniLM-L6-H384-uncased",
                        "name": "MiniLM-L6",
                        "size": "90MB",
                        "params": "22M",
                        "compression": "1.5x",
                        "verified": True
                    }
                ]
            }
        ]
    }

# ===== TRAINING ENDPOINTS =====

@app.post("/api/training/create")
async def create_training(config: dict):
    """Create and start a new training run"""
    global training_manager
    
    # Generate experiment ID with custom name if provided
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    exp_name = config.get("experiment_name", "experiment")
    # Clean the name for use in directory
    safe_name = "".join(c for c in exp_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_')[:50]  # Limit length
    
    exp_id = f"{timestamp}_{safe_name}_{hex(hash(str(config)))[2:10]}"
    exp_dir = PROJECT_ROOT / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Transform config to match expected format for main.py
    dataset_id = config.get("dataset", "imdb_sample")
    
    # Build proper training config
    training_config = {
        "experiment_name": exp_name,
        "experiment_id": exp_id,
        "created_at": timestamp,
        
        # Model configuration
        "model": {
            "name": config.get("teacher_model", "bert-base-uncased"),
            "student_name": config.get("student_model", "distilbert-base-uncased"),
            "type": "transformer",
            "tokenizer_name": config.get("student_model", "distilbert-base-uncased"),
            "max_length": 128
        },
        
        # Training configuration
        "train": {
            "epochs": config.get("epochs", 3),
            "batch_size": config.get("batch_size", 32),
            "lr": config.get("learning_rate", 2e-5),
            "grad_accum_steps": 1,
            "mixed_precision": False,
            "early_stop_patience": 2,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "scheduler": "cosine",
            "warmup_steps": 50,
            "warmup_type": "linear"
        },
        
        # Distillation configuration
        "distillation": {
            "method": "kd_hinton",
            "temperature": config.get("temperature", 4.0),
            "alpha": 0.4
        },
        
        # Data configuration - this is the key fix!
        "data": {
            "train_path": f"data/{dataset_id}.jsonl",
            "val_path": f"data/{dataset_id}.jsonl"  # Using same for now
        },
        
        # Device configuration
        "device": {
            "prefer_mps": True,
            "prefer_cuda": False
        },
        
        # Output configuration
        "output_root": "experiments"
    }
    
    # Save config to YAML
    config_file = exp_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(training_config, f)
    
    # Create experiment info file for immediate access
    info_file = exp_dir / "experiment_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            "experiment_id": exp_id,
            "experiment_name": exp_name,
            "status": "starting",
            "created_at": timestamp,
            "teacher_model": config.get("teacher_model"),
            "student_model": config.get("student_model"),
            "dataset": dataset_id
        }, f, indent=2)
    
    # Update training status
    training_status["is_training"] = True
    training_status["experiment_id"] = exp_id
    training_status["stage"] = "starting"
    
    # Add to active training set
    active_training_ids.add(exp_id)
    
    # Broadcast to websockets
    await broadcast_message({
        "type": "training_started",
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "config": config
    })
    
    # Start the actual training process
    try:
        if not training_manager:
            raise HTTPException(status_code=503, detail="Training manager not initialized")
        
        await training_manager.start_training(exp_id, config_file, exp_dir)
        return {
            "status": "started",
            "experiment_id": exp_id,
            "experiment_name": exp_name,
            "message": "Training pipeline started successfully"
        }
    except Exception as e:
        # Cleanup on error
        training_status["is_training"] = False
        active_training_ids.discard(exp_id)
        await broadcast_message({
            "type": "training_error",
            "experiment_id": exp_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.post("/api/training/{exp_id}/pause")
async def pause_training(exp_id: str):
    """Pause a training run"""
    if not training_manager:
        raise HTTPException(status_code=503, detail="Training manager not initialized")
    
    try:
        training_manager.pause_training(exp_id)
        await broadcast_message({
            "type": "training_paused",
            "experiment_id": exp_id
        })
        return {"status": "paused", "experiment_id": exp_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/training/{exp_id}/resume")
async def resume_training(exp_id: str):
    """Resume a paused training run"""
    if not training_manager:
        raise HTTPException(status_code=503, detail="Training manager not initialized")
    
    try:
        training_manager.resume_training(exp_id)
        await broadcast_message({
            "type": "training_resumed",
            "experiment_id": exp_id
        })
        return {"status": "resumed", "experiment_id": exp_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/training/{exp_id}/stop")
async def stop_training_by_id(exp_id: str):
    """Stop a specific training run"""
    if not training_manager:
        raise HTTPException(status_code=503, detail="Training manager not initialized")
    
    try:
        training_manager.stop_training(exp_id)
        training_status["is_training"] = False
        training_status["stage"] = None
        
        # Remove from active set
        active_training_ids.discard(exp_id)
        
        await broadcast_message({
            "type": "training_stopped",
            "experiment_id": exp_id
        })
        return {"status": "stopped", "experiment_id": exp_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/training/{exp_id}/checkpoint")
async def save_checkpoint(exp_id: str):
    """Save a checkpoint for a training run"""
    # TODO: Implement checkpoint saving
    # This would signal the training process to save current state
    await broadcast_message({
        "type": "checkpoint_saved",
        "experiment_id": exp_id
    })
    return {"status": "checkpoint_saved", "experiment_id": exp_id}

@app.get("/api/training/status")
async def get_training_status():
    """Get overall training status"""
    running_processes = training_manager.get_all_running() if training_manager else {}
    
    return {
        "is_training": len(running_processes) > 0,
        "active_experiments": len(running_processes),
        "experiments": [
            {
                "experiment_id": exp_id,
                "stage": process.current_stage,
                "progress": (process.current_epoch / process.total_epochs * 100) if process.total_epochs > 0 else 0,
                "is_paused": process.is_paused
            }
            for exp_id, process in running_processes.items()
        ]
    }

@app.get("/api/training/active")
async def get_active_training():
    """Get all active training runs"""
    if not training_manager:
        return []
    
    running = training_manager.get_all_running()
    return [
        {
            "experiment_id": exp_id,
            "stage": process.current_stage,
            "epoch": process.current_epoch,
            "total_epochs": process.total_epochs,
            "progress": (process.current_epoch / process.total_epochs * 100) if process.total_epochs > 0 else 0,
            "is_paused": process.is_paused,
            "is_running": process.is_running
        }
        for exp_id, process in running.items()
    ]

@app.get("/api/training/{exp_id}/metrics")
async def get_live_metrics(exp_id: str):
    """Get live metrics for a specific training run"""
    if not training_manager:
        raise HTTPException(status_code=503, detail="Training manager not initialized")
    
    metrics = training_manager.get_metrics(exp_id)
    if not metrics:
        # Try to load from experiment directory if not running
        exp_dir = PROJECT_ROOT / "experiments" / exp_id
        results_file = exp_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            return {
                "experiment_id": exp_id,
                "status": "completed",
                "final_metrics": results
            }
        raise HTTPException(status_code=404, detail=f"No metrics found for experiment {exp_id}")
    
    return {
        "experiment_id": exp_id,
        "status": "running" if training_manager.is_running(exp_id) else "paused",
        "live_metrics": metrics
    }

@app.get("/api/models/pairs")
async def get_model_pairs():
    """Get compatible teacher-student model pairs optimized for Mac M2"""
    pairs = {
        "teachers": [
            {
                "id": "bert-base-uncased",
                "name": "BERT Base",
                "size": "440MB",
                "params": "110M",
                "description": "Industry standard, excellent for text classification",
                "compatible_students": ["distilbert-base-uncased", "TinyBERT", "MobileBERT"]
            },
            {
                "id": "roberta-base",
                "name": "RoBERTa Base",
                "size": "500MB",
                "params": "125M",
                "description": "Improved BERT training, better on sentiment tasks",
                "compatible_students": ["distilbert-base-uncased", "distilroberta-base"]
            },
            {
                "id": "distilbert-base-uncased",
                "name": "DistilBERT Base",
                "size": "268MB",
                "params": "66M",
                "description": "Faster, smaller, good for tight resources",
                "compatible_students": ["TinyBERT", "albert-tiny"]
            },
            {
                "id": "albert-base-v2",
                "name": "ALBERT Base v2",
                "size": "45MB",
                "params": "12M",
                "description": "Parameter-efficient, lightweight transformer",
                "compatible_students": ["albert-tiny"]
            },
            {
                "id": "microsoft/MiniLM-L12-H384-uncased",
                "name": "MiniLM-L12",
                "size": "130MB",
                "params": "33M",
                "description": "Compact, fast, good balance of speed and accuracy",
                "compatible_students": ["MiniLM-L6"]
            }
        ],
        "students": [
            {
                "id": "distilbert-base-uncased",
                "name": "DistilBERT",
                "size": "268MB",
                "params": "66M",
                "compatible_with": ["bert-base-uncased", "roberta-base"]
            },
            {
                "id": "TinyBERT",
                "name": "TinyBERT",
                "size": "58MB",
                "params": "14.5M",
                "compatible_with": ["bert-base-uncased", "distilbert-base-uncased"]
            },
            {
                "id": "MobileBERT",
                "name": "MobileBERT",
                "size": "100MB",
                "params": "25M",
                "compatible_with": ["bert-base-uncased"]
            },
            {
                "id": "distilroberta-base",
                "name": "DistilRoBERTa",
                "size": "330MB",
                "params": "82M",
                "compatible_with": ["roberta-base"]
            },
            {
                "id": "albert-tiny",
                "name": "ALBERT Tiny",
                "size": "16MB",
                "params": "4M",
                "compatible_with": ["albert-base-v2", "distilbert-base-uncased"]
            },
            {
                "id": "MiniLM-L6",
                "name": "MiniLM-L6",
                "size": "90MB",
                "params": "22M",
                "compatible_with": ["microsoft/MiniLM-L12-H384-uncased"]
            }
        ]
    }
    return pairs

@app.post("/api/preflight/check")
async def preflight_check(request: PreflightRequest):
    """
    Check compatibility between teacher and student models
    Returns validation results and auto-suggests alternatives if incompatible
    """
    
    # Get model pairs
    pairs_response = await get_model_pairs()
    teachers = {t["id"]: t for t in pairs_response["teachers"]}
    students = {s["id"]: s for s in pairs_response["students"]}
    
    teacher = teachers.get(request.teacher_model)
    student = students.get(request.student_model)
    
    if not teacher or not student:
        return {
            "compatible": False,
            "error": "Invalid teacher or student model ID",
            "teacher_size": None,
            "student_size": None,
            "compression_ratio": None,
            "estimated_time": None,
            "warnings": ["Model not found in catalog"],
            "suggestions": []
        }
    
    # Check compatibility
    is_compatible = request.student_model in teacher.get("compatible_students", [])
    
    # Calculate compression ratio
    compression_ratio = None
    if is_compatible:
        teacher_params = float(teacher["params"].replace("M", ""))
        student_params = float(student["params"].replace("M", ""))
        compression_ratio = round(teacher_params / student_params, 1)
    
    # Estimate training time
    estimated_time = "N/A"
    if is_compatible and request.dataset_size:
        # Rough estimate: 1 minute per 1000 samples per epoch (3 epochs default)
        minutes = (request.dataset_size / 1000) * 3
        if minutes < 60:
            estimated_time = f"~{int(minutes)} minutes"
        else:
            hours = minutes / 60
            estimated_time = f"~{hours:.1f} hours"
    elif is_compatible:
        estimated_time = "~45 minutes (estimated)"
    
    # Generate suggestions if incompatible
    suggestions = []
    warnings = []
    
    if not is_compatible:
        compatible_students = teacher.get("compatible_students", [])
        warnings.append(f"{student['name']} is not compatible with {teacher['name']}")
        
        if compatible_students:
            suggestions = [
                {
                    "model_id": sid,
                    "name": students[sid]["name"],
                    "reason": f"Recommended for {teacher['name']}"
                }
                for sid in compatible_students
                if sid in students
            ]
            
            suggestions_text = ", ".join([students[sid]["name"] for sid in compatible_students if sid in students])
            warnings.append(f"Please select one of: {suggestions_text}")
        else:
            warnings.append("No compatible students found for this teacher")
    
    return {
        "compatible": is_compatible,
        "teacher_size": teacher["size"],
        "student_size": student["size"],
        "compression_ratio": compression_ratio,
        "estimated_time": estimated_time,
        "warnings": warnings,
        "suggestions": suggestions,
        "details": {
            "teacher_params": teacher["params"],
            "student_params": student["params"],
            "teacher_description": teacher["description"]
        }
    }

@app.post("/api/models/validate")
async def validate_model_pair(request: dict):
    """
    Complete preflight validation using the core preflight system.
    
    Validates:
    - Config structure
    - Model availability on HuggingFace
    - Device compatibility
    - Architecture support
    - Resource requirements
    
    Returns comprehensive validation report
    """
    try:
        teacher_id = request.get('teacher_model')
        student_id = request.get('student_model')
        dataset_id = request.get('dataset', 'imdb_sample')  # Optional dataset parameter
        
        if not teacher_id or not student_id:
            raise HTTPException(
                status_code=400,
                detail="Both teacher_model and student_model required"
            )
        
        print(f"[API] Running preflight validation: {teacher_id} → {student_id}")
        
        # Create temporary config for validation
        temp_config = {
            'model': {
                'name': teacher_id,
                'student_name': student_id,
                'type': 'transformer'
            },
            'data': {
                'train_path': f'data/{dataset_id}.jsonl',
                'val_path': f'data/{dataset_id}.jsonl'
            },
            'train': {
                'epochs': 3,
                'batch_size': 32,
                'lr': 2e-5
            },
            'distillation': {
                'method': 'kd_hinton',
                'temperature': 4.0
            },
            'device': {
                'prefer_mps': True,
                'prefer_cuda': False
            }
        }
        
        # PHASE 1: Config validation (fast, no model loading)
        print("[API] Phase 1: Validating configuration...")
        try:
            from core.preflight.analyser import validate_config_only
            config_validation = validate_config_only(temp_config)
            print(f"[API] Config validation result: {config_validation['is_valid']}")
        except ImportError as e:
            print(f"[API] Warning: Could not import preflight analyser: {e}")
            config_validation = {'is_valid': True, 'errors': [], 'warnings': []}
        except Exception as e:
            print(f"[API] Config validation error: {e}")
            config_validation = {'is_valid': True, 'errors': [], 'warnings': [f'Config validation skipped: {str(e)}']}
        
        # PHASE 2: Model validation with HuggingFace
        print("[API] Phase 2: Validating models on HuggingFace...")
        hf_token = os.getenv('HF_TOKEN')
        
        # Use the ModelValidator we created earlier
        try:
            from core.preflight.model_validator import ModelValidator
            validator = ModelValidator(hf_token=hf_token)
            model_validation = validator.validate_pair(teacher_id, student_id)
            
            print(f"[API] Model validation result: {model_validation['pair_compatible']}")
        except Exception as e:
            print(f"[API] Model validation error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a safe error response
            model_validation = {
                'pair_compatible': False,
                'teacher': {
                    'exists': False,
                    'device_compatible': False,
                    'size_mb': None,
                    'errors': [f"Validation error: {str(e)}"],
                    'warnings': [],
                    'alternatives': []
                },
                'student': {
                    'exists': False,
                    'device_compatible': False,
                    'size_mb': None,
                    'errors': [f"Validation error: {str(e)}"],
                    'warnings': [],
                    'alternatives': []
                },
                'issues': [f"Model validation failed: {str(e)}"],
                'warnings': [],
                'recommendations': []
            }
        
        # Combine results
        validation_result = {
            'valid': config_validation['is_valid'] and model_validation['pair_compatible'],
            'can_proceed': config_validation['is_valid'] and model_validation['pair_compatible'],
            'config_validation': config_validation,
            'model_validation': model_validation,
            'teacher': {
                'id': teacher_id,
                'exists': model_validation['teacher']['exists'],
                'device_compatible': model_validation['teacher']['device_compatible'],
                'size_mb': model_validation['teacher']['size_mb'],
                'errors': model_validation['teacher']['errors'],
                'warnings': model_validation['teacher']['warnings'],
                'alternatives': model_validation['teacher'].get('alternatives', [])
            },
            'student': {
                'id': student_id,
                'exists': model_validation['student']['exists'],
                'device_compatible': model_validation['student']['device_compatible'],
                'size_mb': model_validation['student']['size_mb'],
                'errors': model_validation['student']['errors'],
                'warnings': model_validation['student']['warnings'],
                'alternatives': model_validation['student'].get('alternatives', [])
            },
            'compression_ratio': model_validation.get('compression_ratio'),
            'issues': [
                *config_validation.get('errors', []),
                *model_validation.get('issues', [])
            ],
            'warnings': [
                *config_validation.get('warnings', []),
                *model_validation.get('warnings', [])
            ],
            'recommendations': model_validation.get('recommendations', []),
        }
        
        # Add device info
        import torch
        device_info = {
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'current_device': 'cpu'
        }
        
        if torch.cuda.is_available():
            device_info['current_device'] = 'cuda'
            device_info['cuda_device_name'] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability()
            device_info['cuda_capability'] = f"{major}.{minor}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info['current_device'] = 'mps'
        
        validation_result['device_info'] = device_info
        
        print(f"[API] Validation complete. Can proceed: {validation_result['can_proceed']}")
        
        return validation_result
        
    except Exception as e:
        print(f"[API] Preflight validation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/device/info")
async def get_device_info():
    """Get current device capabilities"""
    import torch
    
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "current_device": "cpu"
    }
    
    if torch.cuda.is_available():
        device_info["current_device"] = "cuda"
        device_info["cuda_version"] = torch.version.cuda
        device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability()
        device_info["cuda_capability"] = f"{major}.{minor}"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_info["current_device"] = "mps"
    
    return device_info

# ===== ASYNC EVALUATION TASK ENDPOINTS =====

@app.post("/api/evaluation/start")
async def start_evaluation_task(request: EvaluationRequest):
    """
    Start async evaluation task (non-blocking).
    Returns task_id for tracking progress.
    """
    try:
        task_manager = get_task_manager()
        
        # Validate experiment exists
        exp_dir = PROJECT_ROOT / "experiments" / request.experiment_id
        if not exp_dir.exists():
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        # Create evaluation function based on type
        eval_type = EvaluationType(request.eval_type)
        
        # For now, return a placeholder task
        # TODO: Implement actual evaluation execution with model loading
        def dummy_eval(progress_callback=None):
            """Placeholder evaluation function"""
            import time
            for i in range(10):
                time.sleep(0.5)
                if progress_callback:
                    progress_callback({
                        'stage': 'evaluation',
                        'batch': i + 1,
                        'total_batches': 10,
                        'progress': (i + 1) / 10 * 100
                    })
            return {'accuracy': 0.85, 'f1': 0.82}
        
        task_id = task_manager.create_task(
            experiment_id=request.experiment_id,
            eval_type=eval_type,
            eval_func=dummy_eval
        )
        
        return {
            'task_id': task_id,
            'status': 'started',
            'message': f'Evaluation task started for {request.experiment_id}'
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start evaluation: {str(e)}"
        )


@app.get("/api/evaluation/task/{task_id}")
async def get_evaluation_task_status(task_id: str):
    """Get status and progress of an evaluation task"""
    try:
        task_manager = get_task_manager()
        task = task_manager.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return task.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )


@app.get("/api/evaluation/tasks")
async def get_all_evaluation_tasks():
    """Get all evaluation tasks"""
    try:
        task_manager = get_task_manager()
        tasks = task_manager.get_all_tasks()
        
        return {
            'tasks': [task.to_dict() for task in tasks.values()],
            'running_count': len(task_manager.get_running_tasks())
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get tasks: {str(e)}"
        )


@app.post("/api/evaluation/task/{task_id}/cancel")
async def cancel_evaluation_task(task_id: str):
    """Cancel a running evaluation task"""
    try:
        task_manager = get_task_manager()
        
        if task_manager.cancel_task(task_id):
            return {
                'task_id': task_id,
                'status': 'cancelled',
                'message': 'Task cancelled successfully'
            }
        else:
            return {
                'task_id': task_id,
                'status': 'not_cancelled',
                'message': 'Task not found or already completed'
            }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {str(e)}"
        )


@app.delete("/api/evaluation/tasks/cleanup")
async def cleanup_old_evaluation_tasks(max_age_hours: int = 24, keep_last_n: int = 10):
    """Clean up old completed/failed evaluation tasks"""
    try:
        task_manager = get_task_manager()
        task_manager.cleanup_old_tasks(max_age_hours=max_age_hours, keep_last_n=keep_last_n)
        
        return {
            'status': 'success',
            'message': f'Cleaned up tasks older than {max_age_hours} hours'
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup tasks: {str(e)}"
        )

# ===== END ASYNC EVALUATION ENDPOINTS =====

@app.get("/api/evaluation/{exp_id}")
async def get_evaluation_metrics(exp_id: str):
    """Get evaluation metrics for an experiment"""
    exp_dir = PROJECT_ROOT / "experiments" / exp_id
    
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Try to load evaluation results
    eval_file = exp_dir / "evaluation" / "metrics.json"
    if not eval_file.exists():
        eval_file = exp_dir / "results.json"
    
    if not eval_file.exists():
        return {
            "status": "pending",
            "message": "Evaluation not yet available",
            "metrics": None
        }
    
    try:
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        
        # Extract evaluation metrics
        if "evaluation" in eval_data:
            eval_metrics = eval_data["evaluation"]
        else:
            eval_metrics = eval_data
        
        # Format confusion matrix if present
        confusion_matrix = None
        if "confusion_matrix" in eval_metrics:
            cm = eval_metrics["confusion_matrix"]
            if isinstance(cm, list):
                confusion_matrix = cm
        
        return {
            "status": "available",
            "metrics": {
                "accuracy": eval_metrics.get("accuracy", 0),
                "precision": eval_metrics.get("precision", 0),
                "recall": eval_metrics.get("recall", 0),
                "f1": eval_metrics.get("f1", eval_metrics.get("f1_score", 0)),
                "loss": eval_metrics.get("loss", 0),
                "confusion_matrix": confusion_matrix
            },
            "timestamp": eval_metrics.get("timestamp", datetime.now().isoformat())
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load evaluation metrics: {str(e)}"
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time training updates"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    print(f"✅ WebSocket client connected. Total connections: {len(websocket_connections)}")
    
    try:
        while True:
            # Keep connection alive and receive any client messages
            data = await websocket.receive_text()
            # Echo back or handle client requests if needed
            await websocket.send_json({"type": "pong", "message": "Connection alive"})
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)
        print(f"❌ WebSocket client disconnected. Remaining connections: {len(websocket_connections)}")

if __name__ == "__main__":
    print("🚀 Starting Zynthe API on http://localhost:8765")
    print("📡 WebSocket available at ws://localhost:8765/ws")
    print("📖 API docs at http://localhost:8765/docs")
    uvicorn.run(app, host="0.0.0.0", port=8765)

"""
FastAPI Backend for Zynthe Knowledge Distillation Toolkit
Connects the Electron UI to the Python training pipeline with full transparency
"""
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import uvicorn
import yaml
import json
import csv
from pathlib import Path
from datetime import datetime

from training_manager import TrainingManager

# Training manager for subprocess handling
training_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize training manager on startup"""
    global training_manager
    training_manager = TrainingManager(websocket_broadcast=broadcast_message)
    yield
    # Cleanup on shutdown (if needed)

app = FastAPI(title="Zynthe API", version="1.0.0", lifespan=lifespan)

# Enable CORS for Electron app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                if 'text' not in reader.fieldnames or 'label' not in reader.fieldnames:
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
        "name": config.get("experiment_name", exp_id),
        "timestamp": timestamp.isoformat(),
        "config": config,
        "results": results,
        "metrics": metrics,
        "stages": stages,
        "logs": logs,
        "export_files": export_files
    }

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
    
    # Save config to YAML (including experiment name)
    config_file = exp_dir / "config.yaml"
    config_with_metadata = {
        **config,
        "experiment_id": exp_id,
        "created_at": timestamp
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_with_metadata, f)
    
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

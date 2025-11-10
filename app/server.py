from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Allow tests or scripts to override where experiment artifacts live. Default to
# the repo's "experiments" folder but make sure it exists so StaticFiles mounting
# never explodes during app import.
ARTIFACT_ROOT = Path(os.environ.get("ZYNTHE_ARTIFACT_ROOT", "experiments")).resolve()
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Zynthe Live API", version="0.1.0")
router = APIRouter(prefix="/api")

# ----------------------------
# WebSocket connection manager
# ----------------------------
class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        text = json.dumps(message)
        async with self._lock:
            conns = list(self.active_connections)
        to_drop: List[WebSocket] = []
        for conn in conns:
            try:
                await conn.send_text(text)
            except Exception:
                to_drop.append(conn)
        for conn in to_drop:
            await self.disconnect(conn)

manager = ConnectionManager()

# ----------------------------
# Helpers
# ----------------------------

def _exp_dir(exp_id: str) -> Path:
    d = ARTIFACT_ROOT / exp_id
    if not d.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    return d

# ----------------------------
# Artifact APIs
# ----------------------------

@router.get("/experiments/{exp_id}/artifacts")
async def list_artifacts(exp_id: str):
    exp = _exp_dir(exp_id)
    images: List[str] = []
    for p in exp.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            images.append(str(p.relative_to(ARTIFACT_ROOT)))
    return {"experiment_id": exp_id, "images": images}

@router.get("/experiments/{exp_id}/confusion/{role}")
async def get_confusion_matrix(exp_id: str, role: str):
    exp = _exp_dir(exp_id)
    cm_dir = exp / f"{role}_confusion"
    img = cm_dir / "confusion_matrix.png"
    metrics = cm_dir / "metrics.json"
    if not img.exists():
        raise HTTPException(status_code=404, detail="Confusion matrix not ready")
    data: Dict[str, Any] = {}
    if metrics.exists():
        try:
            data = json.loads(metrics.read_text())
        except Exception:
            data = {}
    return {
        "experiment_id": exp_id,
        "role": role,
        "image_path": str(img.relative_to(ARTIFACT_ROOT)),
        "metrics": data,
    }

@router.get("/experiments/{exp_id}/batch-log")
async def get_batch_log(exp_id: str):
    exp = _exp_dir(exp_id)
    csv_path = exp / "training_detailed_log.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Batch log not found")
    return {"experiment_id": exp_id, "csv": csv_path.read_text()}

@router.get("/experiments/{exp_id}/micro/{role}/{epoch}")
async def get_micro_series(exp_id: str, role: str, epoch: int):
    exp = _exp_dir(exp_id)
    files = {
        "train": exp / f"{role}_epoch{epoch}_train_micro.png",
        "eval": exp / f"{role}_epoch{epoch}_eval_micro.png",
    }
    resolved = {k: str(v.relative_to(ARTIFACT_ROOT)) for k, v in files.items() if v.exists()}
    if not resolved:
        raise HTTPException(status_code=404, detail="Micro-series not found")
    return {"experiment_id": exp_id, "role": role, "epoch": epoch, "images": resolved}

# ----------------------------
# Ingest live events (HTTP -> WS broadcast)
# ----------------------------

@router.post("/stream")
async def ingest_stream_event(req: Request):
    try:
        payload = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    # Basic validation & normalization
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")
    # Fan-out to all WS clients
    await manager.broadcast(payload)
    return JSONResponse({"ok": True})

app.include_router(router)

# Serve experiment artifacts statically for easy <img src="/experiments/..."> access
app.mount(
    "/experiments",
    StaticFiles(directory=str(ARTIFACT_ROOT), html=False),
    name="experiments",
)

# ----------------------------
# WebSocket endpoint
# ----------------------------

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Keep connection open, echo pings if needed
        while True:
            # Optional: receive ping messages
            _ = await websocket.receive_text()
            await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)

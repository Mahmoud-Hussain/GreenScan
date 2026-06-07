"""
api_backend.py — GreenScan 2.0 FastAPI server.
All endpoints for dataset management, training, FL, prediction, models,
reporting, and real-time WebSocket streaming.
"""

import asyncio
import json
import os
import threading
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import (
    init_db, get_all_runs, get_epoch_history, get_fl_rounds_for_run,
    get_all_model_versions, set_active_model as db_set_active,
    log_prediction,
)
from dataset_manager import get_stats, get_thumbnails, get_dataloaders, generate_synthetic_dataset
from waste_model import classifier
from training_engine import TRAINING_STATE, train as engine_train
from green_tracker import GreenMetricsTracker

# ── App setup ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    # Startup
    asyncio.create_task(_ws_pump())
    yield
    # Shutdown (nothing needed)

app = FastAPI(title="GreenScan 2.0 API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

init_db()
os.makedirs("datasets/uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ── WebSocket connection manager ──────────────────────────────────────────────
class WSManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active = [c for c in self.active if c is not ws]

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = WSManager()


# ── Background WS pump (drains TRAINING_STATE.queue → all WS clients) ─────────
async def _ws_pump():
    while True:
        try:
            msg = TRAINING_STATE.queue.get_nowait()
            await ws_manager.broadcast(msg)
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.2)
        except Exception:
            await asyncio.sleep(0.5)


# ── Static / UI ───────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
async def serve_ui():
    return FileResponse("index.html")


@app.get("/favicon.ico")
async def favicon():
    # Return empty 204 to suppress browser 404 noise
    from fastapi.responses import Response
    return Response(status_code=204)


# ── Dataset endpoints ─────────────────────────────────────────────────────────

@app.get("/api/dataset/stats")
async def dataset_stats():
    return get_stats()


@app.get("/api/dataset/thumbnails")
async def dataset_thumbnails(split: str = "train", max_per_class: int = 6):
    return get_thumbnails(split=split, max_per_class=max_per_class)


@app.post("/api/dataset/generate-synthetic")
async def generate_synthetic(images_per_class: int = 30):
    stats = generate_synthetic_dataset(images_per_class=images_per_class)
    return {"message": "Synthetic dataset generated", "stats": stats}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...), category: str = "Organic"):
    import shutil
    dest = Path(f"datasets/waste_dataset/train/{category}")
    dest.mkdir(parents=True, exist_ok=True)
    fpath = dest / file.filename
    with open(fpath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"info": f"Saved to {fpath}"}


# ── Training endpoints ────────────────────────────────────────────────────────

class TrainConfig(BaseModel):
    epochs: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 32
    fine_tune: bool = True
    fine_tune_epochs: int = 5


@app.post("/api/train/centralized")
async def start_centralized_training(config: TrainConfig):
    if TRAINING_STATE.is_running:
        raise HTTPException(status_code=409, detail="Training already in progress")

    loop = asyncio.get_event_loop()

    def _run():
        train_loader, val_loader, _, _ = get_dataloaders(
            batch_size=config.batch_size, num_workers=0
        )
        engine_train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            fine_tune=config.fine_tune,
            fine_tune_epochs=config.fine_tune_epochs,
            batch_size=config.batch_size,
            loop=loop,
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "started", "config": config.model_dump()}


@app.post("/api/train/stop")
async def stop_training():
    TRAINING_STATE.should_stop = True
    return {"status": "stop_requested"}


@app.get("/api/train/status")
async def training_status():
    return {
        "is_running": TRAINING_STATE.is_running,
        "run_id": TRAINING_STATE.current_run_id,
        "best_val_acc": round(TRAINING_STATE.best_val_acc * 100, 2),
        "history": TRAINING_STATE.history,
        "all_runs": get_all_runs(),
    }


@app.get("/api/train/history/{run_id}")
async def epoch_history(run_id: int):
    return get_epoch_history(run_id)


# ── Federated Learning endpoints ───────────────────────────────────────────────

_fl_process: subprocess.Popen = None
_fl_node_processes: list = []


@app.post("/api/fl/start")
async def start_fl(rounds: int = 3, num_nodes: int = 3):
    global _fl_process, _fl_node_processes

    # Start server in background process
    _fl_process = subprocess.Popen(
        [sys.executable, "fl_server.py", "--rounds", str(rounds),
         "--min-clients", str(num_nodes)],
        cwd=str(Path.cwd())
    )

    # Wait briefly then launch client nodes
    await asyncio.sleep(3)
    _fl_node_processes = []
    for i in range(num_nodes):
        p = subprocess.Popen(
            [sys.executable, "fl_client.py",
             "--node-id", str(i), "--nodes", str(num_nodes)],
            cwd=str(Path.cwd())
        )
        _fl_node_processes.append(p)

    return {
        "status": "fl_started",
        "server_pid": _fl_process.pid,
        "node_pids": [p.pid for p in _fl_node_processes],
        "rounds": rounds,
        "num_nodes": num_nodes,
    }


@app.get("/api/fl/status")
async def fl_status():
    global _fl_process, _fl_node_processes
    server_running = _fl_process is not None and _fl_process.poll() is None
    nodes = [
        {"node_id": i, "running": p.poll() is None}
        for i, p in enumerate(_fl_node_processes)
    ]
    from database import get_all_runs, get_fl_rounds_for_run
    fl_runs = [r for r in get_all_runs() if r["mode"] == "federated"]
    latest_run_id = fl_runs[0]["id"] if fl_runs else None
    rounds = get_fl_rounds_for_run(latest_run_id) if latest_run_id else []
    return {
        "server_running": server_running,
        "nodes": nodes,
        "latest_run_id": latest_run_id,
        "rounds": rounds,
    }


# ── Model management endpoints ────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    return get_all_model_versions()


@app.post("/api/models/{version}/activate")
async def activate_model(version: str):
    db_set_active(version)
    classifier._trained_model = None  # Force reload on next predict
    return {"status": "activated", "version": version}


@app.get("/api/models/{version}/download")
async def download_model(version: str):
    path = Path(f"models/model_{version}.pt")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(str(path), filename=f"greenscan_model_{version}.pt",
                        media_type="application/octet-stream")


# ── Prediction endpoint ───────────────────────────────────────────────────────

@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    import time
    image_bytes = await file.read()
    t0 = time.time()
    try:
        result = classifier.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    inference_ms = round((time.time() - t0) * 1000)
    result["inference_time_ms"] = inference_ms

    log_prediction(
        filename=file.filename or "unknown",
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        inference_ms=inference_ms,
        model_version=result.get("model_version", "imagenet"),
        top5_json=json.dumps(result.get("top5", [])),
    )
    return result


# ── Comparison endpoint ───────────────────────────────────────────────────────

@app.get("/api/compare")
async def compare_modes():
    all_runs = get_all_runs()
    centralized = [r for r in all_runs if r["mode"] == "centralized"]
    federated = [r for r in all_runs if r["mode"] == "federated"]

    def summarize(runs):
        if not runs:
            return None
        best = max(runs, key=lambda r: r.get("best_val_acc") or 0)
        return {
            "run_count": len(runs),
            "best_val_acc": round((best.get("best_val_acc") or 0) * 100, 2),
            "latest_run_id": runs[0]["id"],
        }

    return {
        "centralized": summarize(centralized),
        "federated": summarize(federated),
    }


# ── Legacy metrics endpoint ───────────────────────────────────────────────────

@app.get("/metrics")
async def get_metrics():
    stats = get_stats()
    return {
        "total_images": stats["total"],
        "dataset_stats": stats,
        "training_running": TRAINING_STATE.is_running,
        "best_val_acc": round(TRAINING_STATE.best_val_acc * 100, 2),
    }


# ── Report endpoints ──────────────────────────────────────────────────────────

@app.get("/api/report/pdf/{run_id}")
async def report_pdf(run_id: int):
    from report_generator import generate_pdf
    data = generate_pdf(run_id=run_id, dataset_stats=get_stats())
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=greenscan_report_{run_id}.pdf"})


@app.get("/api/report/csv/{run_id}")
async def report_csv(run_id: int):
    from report_generator import generate_csv
    data = generate_csv(run_id=run_id)
    return Response(content=data, media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=greenscan_metrics_{run_id}.csv"})


@app.get("/api/report/excel/{run_id}")
async def report_excel(run_id: int):
    from report_generator import generate_excel
    data = generate_excel(run_id=run_id)
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=greenscan_report_{run_id}.xlsx"}
    )


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/training")
async def ws_training(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from dotenv import load_dotenv
import cv2
import numpy as np
import os
import uuid
import imageio
import requests
from typing import Optional
import math
from datetime import datetime, timezone, timedelta
import torch
import platform
import psutil
import time

# Configurable threshold for Colony Absconding (Low Bee Count)
MIN_BEE_THRESHOLD = 20

def filter_detections(boxes_data, dist_thresh=50):
    """
    Manually filter boxes based on center distance.
    boxes_data: List/Array of [x1, y1, x2, y2, conf, cls]
    """
    if len(boxes_data) == 0:
        return []
    
    # Convert to list for easier manipulation
    boxes = sorted(boxes_data, key=lambda x: x[4], reverse=True) # Sort by confidence
    final_boxes = []
    
    while len(boxes) > 0:
        current = boxes.pop(0) # Take highest confidence
        final_boxes.append(current)
        
        # Calculate center of current
        cx1 = (current[0] + current[2]) / 2
        cy1 = (current[1] + current[3]) / 2
        
        # Filter out remaining boxes that are too close
        new_boxes = []
        for box in boxes:
            cx2 = (box[0] + box[2]) / 2
            cy2 = (box[1] + box[3]) / 2
            dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            if dist > dist_thresh:
                new_boxes.append(box)
        boxes = new_boxes
        
    return final_boxes

# Output directory
OUTPUT_DIR = "../frontend/static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load environment variables — always relative to this file, not the cwd
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── Device setup — must happen BEFORE model load ────────────────────────────
if torch.cuda.is_available():
    DEVICE   = 'cuda'
    # Ultralytics uses integer GPU index (0,1,2...) rather than the string 'cuda'
    # Using integer is the most reliable way to force GPU inference in Ultralytics
    INFER_DEVICE = 0
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM_GB  = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    print(f"[DEVICE] GPU detected : {GPU_NAME} ({VRAM_GB} GB VRAM)")
else:
    DEVICE       = 'cpu'
    INFER_DEVICE = 'cpu'
    GPU_NAME     = None
    VRAM_GB      = None
    print("[DEVICE] No GPU detected — using CPU")

# Load YOLO model
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
if not os.path.exists(_MODEL_PATH):
    raise FileNotFoundError(
        f"\n\n❌ YOLO model not found at: {_MODEL_PATH}\n"
        "Please place your trained 'best.pt' file inside the 'backend/' folder.\n"
        "(Tip: After training with Ultralytics, it's usually at runs/detect/train/weights/best.pt)\n"
    )
model = YOLO(_MODEL_PATH)
model.to(DEVICE)  # Move model weights to GPU VRAM at startup

# Verify the model is on the expected device
try:
    _actual_device = next(model.model.parameters()).device
    print(f"[DEVICE] Model parameters verified on : {_actual_device}")
except Exception:
    print("[DEVICE] Could not verify model device (non-critical)")

# GPU warm-up: run one dummy inference so the CUDA kernels are compiled
# This prevents the first real request from being slow due to JIT compilation
if DEVICE == 'cuda':
    try:
        _dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(_dummy, imgsz=640, conf=0.20, verbose=False, device=INFER_DEVICE)
        torch.cuda.synchronize()
        print("[DEVICE] GPU warm-up complete — CUDA kernels ready")
    except Exception as e:
        print(f"[DEVICE] GPU warm-up failed (non-critical): {e}")

print(f"[DEVICE] Ready — inference will use: {str(INFER_DEVICE).upper() if INFER_DEVICE == 'cpu' else f'GPU:{INFER_DEVICE} (CUDA)'}")

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="../frontend/static", html=True), name="static")

# Device info endpoint — safe for all environments
@app.get("/device_info")
async def device_info():
    try:
        ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        ram_gb = None
    try:
        os_platform = platform.system()
    except Exception:
        os_platform = None
    try:
        cpu_name  = platform.processor() or platform.machine() or "Unknown CPU"
        cpu_cores = psutil.cpu_count(logical=False)   # physical cores
        cpu_threads = psutil.cpu_count(logical=True)  # logical threads
    except Exception:
        cpu_name = None
        cpu_cores = None
        cpu_threads = None
    return JSONResponse(content={
        "device":        DEVICE,
        "gpu_available": DEVICE == 'cuda',
        "gpu_name":      GPU_NAME,
        "vram_gb":       VRAM_GB,
        "ram_gb":        ram_gb,
        "cpu_name":      cpu_name,
        "cpu_cores":     cpu_cores,
        "cpu_threads":   cpu_threads,
        "platform":      os_platform,
    })

# ─── Benchmark endpoint: GPU vs CPU comparison ────────────────────────────────
@app.get("/benchmark")
async def benchmark():
    """
    Runs 5 warm inferences on GPU (if available) and 5 on CPU,
    so you can verify whether GPU is actually faster.
    Also reports which device the model parameters are actually on.
    """
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    N = 5

    # Verify actual model device
    try:
        actual_param_device = str(next(model.model.parameters()).device)
    except Exception:
        actual_param_device = "unknown"

    def run_n(dev, n):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            model.predict(dummy, imgsz=640, conf=0.20, verbose=False, device=dev)
            if DEVICE == 'cuda' and dev != 'cpu':
                torch.cuda.synchronize()
            times.append(round((time.perf_counter() - t0) * 1000, 1))
        return times

    results = {}

    # GPU benchmark
    if DEVICE == 'cuda':
        try:
            gpu_times = run_n(INFER_DEVICE, N)
            results["gpu"] = {
                "times_ms": gpu_times,
                "avg_ms":   round(sum(gpu_times) / len(gpu_times), 1),
                "min_ms":   min(gpu_times),
            }
        except Exception as e:
            results["gpu"] = {"error": str(e)}

    # CPU benchmark
    try:
        cpu_times = run_n('cpu', N)
        results["cpu"] = {
            "times_ms": cpu_times,
            "avg_ms":   round(sum(cpu_times) / len(cpu_times), 1),
            "min_ms":   min(cpu_times),
        }
    except Exception as e:
        results["cpu"] = {"error": str(e)}

    speedup = None
    if "gpu" in results and "cpu" in results and "avg_ms" in results["gpu"] and "avg_ms" in results["cpu"]:
        if results["gpu"]["avg_ms"] > 0:
            speedup = round(results["cpu"]["avg_ms"] / results["gpu"]["avg_ms"], 2)

    print(f"[BENCHMARK] model params on: {actual_param_device}")
    if "gpu" in results and "avg_ms" in results.get("gpu", {}):
        print(f"[BENCHMARK] GPU avg: {results['gpu']['avg_ms']} ms")
    if "cpu" in results and "avg_ms" in results.get("cpu", {}):
        print(f"[BENCHMARK] CPU avg: {results['cpu']['avg_ms']} ms")
    if speedup:
        print(f"[BENCHMARK] GPU speedup: {speedup}x")

    return JSONResponse(content={
        "configured_device":  DEVICE,
        "infer_device":       str(INFER_DEVICE),
        "model_param_device": actual_param_device,
        "benchmark":          results,
        "speedup_x":          speedup,
    })

# ─── Telegram Helpers ────────────────────────────────────────────────────────

def _ist_now() -> str:
    """Returns current time formatted in IST."""
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).strftime("%d %b %Y, %I:%M:%S %p IST")


def _build_image_alert_msg(count: int, threshold: int) -> str:
    line = "-" * 32
    return (
        f"BEEVISION - IMAGE ANALYSIS\n"
        f"{line}\n"
        f"STATUS : LOW BEE ACTIVITY DETECTED\n"
        f"WARNING: Possible Colony Absconding Risk\n\n"
        f"DETECTION RESULTS\n"
        f"  Bees Detected      : {count}\n"
        f"  Alert Threshold    : {threshold}\n"
        f"  Below Threshold By : {threshold - count} bees\n\n"
        f"Time : {_ist_now()}\n"
        f"{line}\n"
        f"BeeVision AI Detection System"
    )



def _build_video_alert_msg(min_count: int, max_count: int, avg_count: float,
                           total_frames: int, fps: int, threshold: int) -> str:
    line = "-" * 32
    duration = total_frames / fps if fps > 0 else 0
    drop_pct = ((max_count - min_count) / max_count * 100) if max_count > 0 else 0
    return (
        f"BEEVISION - VIDEO ANALYSIS\n"
        f"{line}\n"
        f"STATUS  : LOW BEE ACTIVITY DETECTED\n"
        f"WARNING : Possible Colony Absconding Risk\n\n"
        f"BEE COUNT ANALYTICS\n"
        f"  Peak Count         : {max_count} bees\n"
        f"  Minimum Count      : {min_count} bees  (triggered alert)\n"
        f"  Average Count      : {avg_count:.1f} bees\n"
        f"  Alert Threshold    : {threshold} bees\n"
        f"  Below Threshold By : {threshold - min_count} bees\n\n"
        f"ACTIVITY DROP\n"
        f"  {max_count} -> {min_count} bees  ({drop_pct:.1f}% drop detected)\n\n"
        f"VIDEO INFO\n"
        f"  Frames Analyzed    : {total_frames}\n"
        f"  Duration           : ~{duration:.1f}s  @ {fps} fps\n\n"
        f"Time : {_ist_now()}\n"
        f"{line}\n"
        f"BeeVision AI Detection System"
    )



def _send_telegram_alert(body: str):
    """
    Send a Telegram message.
    Returns (sent: bool, error: str | None):
      (True,  None)   → sent successfully
      (False, None)   → Telegram not configured in .env
      (False, "...")  → send attempt failed
    """
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        print("[TELEGRAM] Not configured — TELEGRAM_TOKEN or TELEGRAM_CHAT_ID missing in .env")
        return False, None
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {"chat_id": chat_id, "text": body}
        response = requests.post(url, data=data, timeout=10)
        if response.status_code != 200:
            err = f"Telegram send failed ({response.status_code}): {response.text}"
            print(f"[TELEGRAM] {err}")
            return False, err
        print(f"[TELEGRAM] Message sent successfully.")
        return True, None
    except Exception as e:
        err = f"Telegram error: {e}"
        print(f"[TELEGRAM] {err}")
        return False, err

# Root endpoint
@app.get("/")
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "../frontend/index.html")
    return FileResponse(index_path, media_type="text/html")

@app.get("/style.css")
async def style():
    resp = FileResponse(os.path.join(os.path.dirname(__file__), "../frontend/style.css"), media_type="text/css")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.get("/script.js")
async def script():
    resp = FileResponse(os.path.join(os.path.dirname(__file__), "../frontend/script.js"), media_type="application/javascript")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

# Image prediction
@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...), threshold: int = Form(20)):
    try:
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        t0 = time.perf_counter()
        results = model.predict(img, imgsz=640, conf=0.20, verbose=False, device=INFER_DEVICE)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()  # wait for GPU to finish before timing
        infer_ms = round((time.perf_counter() - t0) * 1000, 1)
        print(f"[INFER] Image inference time: {infer_ms} ms on {DEVICE.upper()}")
        
        # Custom Filtering
        raw_boxes = results[0].boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]
        filtered_boxes = filter_detections(raw_boxes, dist_thresh=40)

        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cls = int(box[5])
            label = model.names[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        count = len(filtered_boxes)
        filename = f"{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, img)

        # Send Telegram alert only when count is below threshold
        sms_error = None
        sms_sent = False
        if count < threshold:
            print(f"[ALERT] Image: count={count} < threshold={threshold} — sending alert")
            sms_sent, sms_error = _send_telegram_alert(_build_image_alert_msg(count, threshold))
        else:
            print(f"[INFO] Image: count={count} >= threshold={threshold} — no alert")

        base_url = str(request.base_url).rstrip('/')
        return JSONResponse(content={
            "image_url": f"{base_url}/static/outputs/{filename}?v={uuid.uuid4().hex}",
            "count": count,
            "sms_error": sms_error,
            "sms_sent": sms_sent,
            "threshold": threshold,
            "infer_ms": infer_ms,
            "infer_device": DEVICE,
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Video prediction
@app.post("/predict_video/")
async def predict_video(request: Request, file: UploadFile = File(...), threshold: int = Form(20)):
    try:
        input_path = os.path.join(OUTPUT_DIR, f"input_{uuid.uuid4().hex}.mp4")
        with open(input_path, "wb") as f:
            f.write(await file.read())

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return JSONResponse(content={"error": "Cannot open uploaded video"}, status_code=400)

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

        output_filename = f"{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        frame_counts = {}
        frame_idx = 0
        max_count = 0
        total_infer_ms = 0.0

        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', format='ffmpeg', pixelformat='yuv420p', macro_block_size=1)

        min_count = float('inf')
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t_frame = time.perf_counter()
                results = model.predict(frame, imgsz=640, conf=0.20, verbose=False, device=INFER_DEVICE)
                total_infer_ms += (time.perf_counter() - t_frame) * 1000
                
                # Custom Filtering
                raw_boxes = results[0].boxes.data.cpu().numpy()
                filtered_boxes = filter_detections(raw_boxes, dist_thresh=40)
                
                count = len(filtered_boxes)
                frame_counts[frame_idx] = count
                if count > max_count:
                    max_count = count
                if count < min_count:
                    min_count = count

                for box in filtered_boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cls = int(box[5])
                    label = model.names[cls]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(rgb)
                frame_idx += 1
        finally:
            cap.release()
            writer.close()
            os.remove(input_path)

        # Compute final analytics
        if min_count == float('inf'):
            min_count = 0
        total_frames = frame_idx
        all_counts = list(frame_counts.values())
        avg_count = sum(all_counts) / len(all_counts) if all_counts else 0.0
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        avg_infer_ms = round(total_infer_ms / max(frame_idx, 1), 1)
        print(f"[INFER] Video: {frame_idx} frames, avg {avg_infer_ms} ms/frame on {DEVICE.upper()}")

        # Send Telegram alert only when min count drops below threshold
        sms_error = None
        sms_sent = False
        print(f"[INFO] Video: min={min_count}, max={max_count}, avg={avg_count:.1f}, frames={total_frames}, threshold={threshold}")
        if min_count < threshold:
            print("[ALERT] Video: triggering absconding alert")
            sms_sent, sms_error = _send_telegram_alert(
                _build_video_alert_msg(min_count, max_count, avg_count, total_frames, fps, threshold)
            )
        else:
            print("[INFO] Video: count remained above threshold — no alert")

        base_url = str(request.base_url).rstrip('/')
        return JSONResponse(content={
            "video_url": f"{base_url}/static/outputs/{output_filename}?v={uuid.uuid4().hex}",
            "fps": fps,
            "frame_counts": frame_counts,
            "sms_error": sms_error,
            "sms_sent": sms_sent,
            "threshold": threshold,
            "infer_device": DEVICE,
            "avg_infer_ms": avg_infer_ms,
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

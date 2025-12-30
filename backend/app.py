from fastapi import FastAPI, UploadFile, File, Request
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

# Configurable threshold
ALERT_THRESHOLD = 100

# Output directory
# Output directory
OUTPUT_DIR = "../frontend/static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load environment variables
load_dotenv()

# Load YOLO model
model = YOLO("best.pt")

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

# Telegram alert function
def _send_telegram_alert_if_configured(body: str) -> Optional[str]:
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return None
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {"chat_id": chat_id, "text": body}
        response = requests.post(url, data=data)
        if response.status_code != 200:
            return f"Telegram send failed: {response.text}"
        return None
    except Exception as e:
        return f"Telegram error: {e}"

# Root endpoint
@app.get("/")
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "../frontend/index.html")
    return FileResponse(index_path, media_type="text/html")

@app.get("/style.css")
async def style():
    return FileResponse(os.path.join(os.path.dirname(__file__), "../frontend/style.css"), media_type="text/css")

@app.get("/script.js")
async def script():
    return FileResponse(os.path.join(os.path.dirname(__file__), "../frontend/script.js"), media_type="application/javascript")

# Image prediction
@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model.predict(img, imgsz=640, conf=0.25, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = model.names[cls]
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        count = len(results[0].boxes)
        filename = f"{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, img)

        # Telegram alert if count > threshold
        sms_error = None
        sms_sent = False
        if count > ALERT_THRESHOLD:
            err = _send_telegram_alert_if_configured(f"🐝 Alert! Bee count = {count} (image)")
            sms_error = err
            sms_sent = err is None

        base_url = str(request.base_url).rstrip('/')
        return JSONResponse(content={
            "image_url": f"{base_url}/static/outputs/{filename}?v={uuid.uuid4().hex}",
            "count": count,
            "sms_error": sms_error,
            "sms_sent": sms_sent,
            "threshold": ALERT_THRESHOLD
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Video prediction
@app.post("/predict_video/")
async def predict_video(request: Request, file: UploadFile = File(...)):
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

        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', format='ffmpeg', output_params=['-pix_fmt','yuv420p'])

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)
                count = len(results[0].boxes)
                frame_counts[frame_idx] = count
                if count > max_count:
                    max_count = count

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls)
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

        # Telegram alert if max_count > threshold
        sms_error = None
        sms_sent = False
        if max_count > ALERT_THRESHOLD:
            err = _send_telegram_alert_if_configured(f"🐝 Alert! Bee count peak = {max_count} (video)")
            sms_error = err
            sms_sent = err is None

        base_url = str(request.base_url).rstrip('/')
        return JSONResponse(content={
            "video_url": f"{base_url}/static/outputs/{output_filename}?v={uuid.uuid4().hex}",
            "fps": fps,
            "frame_counts": frame_counts,
            "sms_error": sms_error,
            "sms_sent": sms_sent,
            "threshold": ALERT_THRESHOLD
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

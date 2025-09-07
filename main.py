from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import tempfile
import os
import uuid

# Load YOLO model
model = YOLO("best.pt")

app = FastAPI()

# Serve static files (videos, frontend assets)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return {"message": "Honey Bee Detection API is running!"}

@app.post("/predict_video/")
async def predict_video(file: UploadFile = File(...)):
    # Generate unique file names
    video_id = str(uuid.uuid4())
    input_path = f"static/{video_id}_input.mp4"
    output_path = f"static/{video_id}_output.mp4"

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Process video
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        num_boxes = len(results[0].boxes)
        annotated_frame = results[0].plot(labels=False, line_width=2)

        cv2.putText(
            annotated_frame,
            f"Count: {num_boxes}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        out.write(annotated_frame)

    cap.release()
    out.release()

    # Return video URL (frontend can play it)
    video_url = f"/static/{video_id}_output.mp4"
    return JSONResponse({"video_url": video_url})

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import tempfile
import os

# Load YOLOv8 model once
model = YOLO("best.pt")

app = FastAPI()

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict_video/")
async def predict_video(file: UploadFile = File(...)):
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(await file.read())
        input_path = temp.name

    output_path = input_path.replace(".mp4", "_output.mp4")

    # Read video
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

    return FileResponse(output_path, media_type="video/mp4", filename="output.mp4")

# Bee Detection Web App

This is a FastAPI-based web application that uses a YOLO model (`best.pt`) to detect bees in images and videos. It provides a web interface for uploading files and viewing predictions.

## Features

- **Image Prediction**: Upload an image to detect bees and get a count.
- **Video Prediction**: Upload a video to process frames and track bee counts over time.
- **Telegram Alerts**: Sends an alert if the bee count exceeds a configured threshold.

## Setup

1.  **Clone the repository**.
2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Environment Variables**:
    Copy `.env.example` to `.env` and fill in your Telegram credentials:
    ```
    TELEGRAM_TOKEN=your_token_here
    TELEGRAM_CHAT_ID=your_chat_id_here
    ```

## Running the App

Run the application locally using the provided batch script or directly with uvicorn:

```bash
uvicorn app:app --reload
```

or on Windows:

```bash
run_locally.bat
```

## files Description
- `app.py`: Main application logic.
- `best.pt`: Pre-trained YOLO model weights.
- `index.html`: Frontend interface.

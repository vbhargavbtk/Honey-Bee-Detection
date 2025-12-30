# 🐝 Honey Bee Detection in Videos

**Precision Beekeeping Powered by AI & Computer Vision**

This project is a state-of-the-art web application designed to automate the detection and monitoring of honeybees using advanced deep learning models (YOLO). By analyzing images and videos, it aids researchers, farmers, and ecologists in tracking bee populations, identifying hive health trends, and ensuring global food security through effective pollination monitoring.

---

## 📂 Project Structure

*   **`frontend/`**: Contains the user interface code (HTML, CSS, JS).
*   **`backend/`**: Contains the FastAPI application (`app.py`), the YOLO model (`best.pt`), and Python dependencies.

## 🌟 Key Features

*   **Real-Time Detection**: Instantly identifies honeybees in uploaded images and video feeds with high accuracy.
*   **Automated Counting**: Provides precise counts of bees per frame.
*   **Video Analysis**: Detailed frame-by-frame analysis generating processed video with bounding boxes.
*   **Smart Alerts**: Integrated **Telegram Notification System** that sends instant alerts if bee activity exceeds thresholds.
*   **Interactive Dashboard**: A modern, responsive web interface.

## 🛠️ Technology Stack

*   **Backend**: Python, FastAPI
*   **AI/ML**: YOLO (You Only Look Once), OpenCV, NumPy
*   **Frontend**: HTML5, CSS3, JavaScript
*   **Utilities**: `imageio`, `python-dotenv`

## 📝 How to Use
1.  **Upload**: Select an image or video file.
2.  **Analyze**: Click "Analyze Now".
3.  **View Results**: See detected bees with bounding boxes and counts.

# 🐝 Honey Bee Detection in Videos

**Precision Beekeeping Powered by AI & Computer Vision**

This project is a state-of-the-art web application designed to automate the detection and monitoring of honeybees using advanced deep learning models (YOLO). By analyzing images and videos , it aids researchers, farmers, and ecologists in tracking bee populations, identifying hive health trends, and ensuring global food security through effective pollination monitoring.

---

## 🌟 Key Features

*   **Real-Time Detection**: Instantly identifies honeybees in uploaded images and video feeds with high accuracy.
*   **Automated Counting**: Provides precise counts of bees per frame, helping to estimate colony activity levels.
*   **Video Analysis**: detailed frame-by-frame analysis of video footage, generating a processed video with bounding boxes and labels.
*   **Smart Alerts**: Integrated **Telegram Notification System** that sends instant alerts to your phone if bee activity exceeds pre-defined thresholds.
*   **Interactive Dashboard**: A modern, responsive web interface for easy file uploads and result visualization.

## 🛠️ Technology Stack

*   **Backend**: Python, FastAPI
*   **AI/ML**: YOLO (You Only Look Once) via `ultralytics`, OpenCV, NumPy
*   **Frontend**: HTML5, CSS3, JavaScript
*   **Utilities**: `imageio` for video processing, `python-dotenv` for security.

### How to Use
1.  **Upload**: Click "Choose File" to select an image (`.jpg`, `.png`) or video (`.mp4`) of a beehive.
2.  **Analyze**: Click "Analyze Now". The AI will process the file in the background.
3.  **View Results**: 
    *   **Images**: See the image with drawn bounding boxes and the total bee count.
    *   **Videos**: Watch the processed video playback with real-time tracking.

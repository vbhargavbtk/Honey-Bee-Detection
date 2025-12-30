# 🐝 Bee Detection & Monitoring Platform

**Precision Beekeeping Powered by AI & Computer Vision**

This project is a state-of-the-art web application designed to automate the detection and monitoring of honeybees using advanced deep learning models (YOLO). By analyzing images and videos in real-time, it aids researchers, farmers, and ecologists in tracking bee populations, identifying hive health trends, and ensuring global food security through effective pollination monitoring.

---

## 🌟 Key Features

*   **Real-Time Detection**: Instantly identifies honeybees in uploaded images and video feeds with high accuracy.
*   **Automated Counting**: Provides precise counts of bees per frame, helping to estimate colony activity levels.
*   **Video Analysis**: detailed frame-by-frame analysis of video footage, generating a processed video with bounding boxes and labels.
*   **Smart Alerts**: Integrated **Telegram Notification System** that sends instant alerts to your phone if bee activity exceeds or drops below configured thresholds.
*   **Interactive Dashboard**: A modern, responsive web interface for easy file uploads and result visualization.

## 🛠️ Technology Stack

*   **Backend**: Python, FastAPI
*   **AI/ML**: YOLO (You Only Look Once) via `ultralytics`, OpenCV, NumPy
*   **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
*   **Utilities**: `imageio` for video processing, `python-dotenv` for security.

## 🚀 Getting Started

Follow these steps to set up the project locally on your machine.

### Prerequisites
*   Python 3.8 or higher
*   Git

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/vbhargavbtk/Honey-Bee-Detection.git
    cd Honey-Bee-Detection
    ```

2.  **Create a Virtual Environment**
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    This project uses environment variables to keep sensitive credentials safe.
    *   Duplicate the `.env.example` file and rename it to `.env`.
    *   Open `.env` and add your Telegram Bot Token and Chat ID (optional, for alerts).
    ```ini
    TELEGRAM_TOKEN=your_token_here
    TELEGRAM_CHAT_ID=your_chat_id_here
    ```

## 🏃‍♂️ Usage

### Running Locally
You can start the server using the provided batch script (Windows) or standard Uvicorn commands.

**Option 1: Windows Batch Script**
```bash
run_locally.bat
```

**Option 2: Terminal**
```bash
uvicorn app:app --reload
```

Once running, open your browser and navigate to:
👉 **http://127.0.0.1:8000**

### How to Use
1.  **Upload**: Click "Choose File" to select an image (`.jpg`, `.png`) or video (`.mp4`) of a beehive or flower patch.
2.  **Analyze**: Click "Analyze Now". The AI will process the file in the background.
3.  **View Results**: 
    *   **Images**: See the image with drawn bounding boxes and the total bee count.
    *   **Videos**: Watch the processed video playback with real-time tracking.

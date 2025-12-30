# 🐝 Honey Bee Detection in Videos

**Precision Beekeeping Powered by AI & Computer Vision**

This project is a state-of-the-art web application designed to automate the detection and monitoring of honeybees using advanced deep learning models (YOLO). By analyzing images and videos, it aids researchers, farmers, and ecologists in tracking bee populations, identifying hive health trends, and ensuring global food security through effective pollination monitoring.

---

## 📂 Project Structure

*   **`frontend/`**: Contains the user interface code (HTML, CSS, JS) and static assets.
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

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   Git

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/vbhargavbtk/Honey-Bee-Detection.git
    cd Honey-Bee-Detection
    ```

2.  **Backend Setup**
    Navigate to the backend directory and set up the environment:
    ```bash
    cd backend
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
    *   Create a `.env` file in the `backend/` directory.
    *   Add your Telegram credentials (optional):
    ```ini
    TELEGRAM_TOKEN=your_token_here
    TELEGRAM_CHAT_ID=your_chat_id_here
    ```

## 🏃‍♂️ Running the App

1.  Make sure you are inside the `backend` directory and your virtual environment is active.
2.  Start the server:

    ```bash
    uvicorn app:app --reload
    ```

3.  Open your browser and navigate to:
    👉 **http://127.0.0.1:8000**

## 📝 How to Use
1.  **Upload**: Select an image or video file.
2.  **Analyze**: Click "Analyze Now".
3.  **View Results**: See detected bees with bounding boxes and counts.

# 🐝 Honey Bee Detection in Videos

**Precision Beekeeping Powered by AI & Computer Vision**

This project is a state-of-the-art web application designed to automate the detection and monitoring of honeybees using advanced deep learning models (YOLO). By analyzing images and videos, it aids researchers, farmers, and ecologists in tracking bee populations, identifying hive health trends, and ensuring global food security through effective pollination monitoring.

---

## 📂 Project Structure

*   **`frontend/`**: Contains the user interface code (HTML, CSS, JS).
*   **`backend/`**: Contains the FastAPI application (`app.py`), the YOLO model (`best.pt`), and Python dependencies.

## 🌟 Key Features

*   **Real-Time Detection**: Instantly identifies honeybees in uploaded images and video feeds with high accuracy.
*   **Colony Absconding Detection**: Automatically detects **low bee activity** (below configurable threshold) to alert beekeepers of potential colony abandonment.
*   **Detailed Analytics**: Tracks **Peak Bee Count** and **Minimum Bee Count** throughout videos to identify sudden drops in population.
*   **Smart Telegram Alerts**: Sends instant notifications for crucial events (e.g., "🐝 Alert! Low bee activity detected").
*   **Interactive Dashboard**: A modern, responsive web interface with real-time analysis status and results.

## ⚙️ Configuration

To enable Telegram notifications, create a `.env` file in the `backend/` directory:

```env
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
# Optional: Set custom threshold (Default: 20)
# MIN_BEE_THRESHOLD=20
```

## 🛠️ Technology Stack

*   **Backend**: Python, FastAPI
*   **AI/ML**: YOLO (You Only Look Once), OpenCV, NumPy
*   **Frontend**: HTML5, CSS3, JavaScript
*   **Utilities**: `imageio`, `python-dotenv`

## 📝 How to Use
1.  **Upload**: Select an image or video file.
2.  **Analyze**: Click "Analyze Now".
3.  **View Results**: See detected bees with bounding boxes and counts.

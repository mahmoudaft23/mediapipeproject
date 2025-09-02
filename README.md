# 🏋️ One-Arm Row Pose Tracking API

This project is a **Flask-based backend service** that uses **OpenCV** and **MediaPipe** to track arm row exercises via a webcam.  
It provides **real-time video streaming** with repetition counting, feedback on elbow position, and motivational messages.  

---

## ✨ Features
- 📹 Real-time pose detection using **MediaPipe Pose**  
- 🔄 Tracks **left and right arm repetitions** separately  
- ⏱️ Built-in timer display on the video feed  
- 🖼️ Live video stream served via Flask  
- 💬 Feedback text for **elbow positioning**  
- 🎯 Motivational messages every 5 reps  
- 🌐 REST API endpoints to start, stream, and stop the exercise  

---

## ⚙️ Requirements
Make sure you have **Python 3.8+** installed.

Install dependencies:
```bash
pip install flask flask-cors opencv-python mediapipe numpy


## 📡 API Endpoints

| Endpoint            | Method | Description                                       |
|---------------------|--------|---------------------------------------------------|
| `/video_feed`       | GET    | Live MJPEG video stream with tracking overlay     |
| `/start_one_arm_row`| GET    | Starts the exercise tracking in a background thread|
| `/stop_exercise`    | GET    | Stops the exercise and returns total rep count     |

---

## 📖 Project Structure





---

## 🛠️ Technologies Used

- **Flask** – Lightweight web framework  
- **Flask-CORS** – Handle cross-origin requests  
- **OpenCV** – Computer vision library for image/video processing  
- **MediaPipe** – Pose detection & landmark tracking  
- **NumPy** – Math operations for angle calculation  

---

## 🎯 Future Improvements

- ✅ Add user authentication for personalized tracking  
- ✅ Save exercise data to a database  
- ✅ Support more exercise types beyond one-arm row  
- ✅ Frontend integration with React or mobile app  

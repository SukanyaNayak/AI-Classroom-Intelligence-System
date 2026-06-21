# 🎓 AI Classroom Intelligence System

> A real-time AI-powered classroom monitoring system built with **YOLOv8**, **OpenCV**, and **Streamlit** that detects students, monitors teacher presence, tracks raised hands, and analyses classroom accessibility.

---

## 📸 System Overview

```
📷 Input Source (Webcam / CCTV / Video / Image)
         │
         ▼
🧠 YOLOv8 Detection Engine
         │
         ├── 👥 Student Count
         ├── 👨‍🏫 Teacher Presence
         ├── ✋ Raised Hand Detection (Pose Estimation)
         ├── 💻 Instrument Detection (Laptop, Phone, Bag...)
         └── 🗺️ Seat Heatmap
         │
         ▼
📊 Streamlit Dashboard → 📁 Attendance Logs → 🎥 Session Recording
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 👥 Student counting | Counts students in real time per frame |
| 👨‍🏫 Teacher detection | Detects if teacher is present or absent in the front zone |
| ✋ Raised hand alerts | Notifies teacher when a student raises their hand with position info |
| 💻 Instrument detection | Detects laptops, smartphones, bags, books, bottles and more |
| 🗺️ Seat heatmap | Visual map of which classroom zones are most occupied |
| 📈 Attendance history | Timestamped log with charts, peak/average stats |
| 🎯 Engagement scoring | Scores each seat zone based on occupancy over session |
| 🎥 Session recording | Records annotated video of each detection session |
| 🔐 Login system | Role-based access (admin / teacher / viewer) |
| 📁 CSV export | Download attendance and hand raise logs |
| 🌐 Remote access | Accessible from any device via ngrok HTTPS tunnel |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10 | Core language |
| YOLOv8 (Ultralytics) | Object detection and pose estimation |
| OpenCV | Frame processing and annotation |
| Streamlit | Web dashboard UI |
| Plotly | Interactive charts and heatmaps |
| Pandas | Data handling and CSV logging |
| ngrok | Remote HTTPS tunnel for network access |

---

## 📁 Project Structure

```
AI-Classroom-Intelligence-System/
│
├── app.py                  ← Main Streamlit application
├── detector.py             ← YOLOv8 student/teacher/instrument detection
├── pose_detector.py        ← YOLOv8 pose estimation for raised hands
├── analytics.py            ← Heatmap, seat occupancy, engagement scoring
├── logger.py               ← Attendance CSV, video recording, logging
├── auth.py                 ← Login, register, password management
├── config.py               ← Settings, class IDs, seat zones
│
├── data/                   ← Generated at runtime (gitignored)
│   ├── attendance_log.csv
│   ├── users.json
│   └── recordings/
│
├── requirements.txt        ← Python dependencies
├── start_classroom_ai.bat  ← One-click Windows startup script
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10+
- Webcam or IP camera (optional)
- Windows / Linux / Mac
- Internet connection (for first-time YOLO model download)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/SukanyaNayak/AI-Classroom-Intelligence-System.git
cd AI-Classroom-Intelligence-System
```

---

### Step 2 — Create virtual environment

```bash
python -m venv classroom_env

# Windows
classroom_env\Scripts\activate

# Mac / Linux
source classroom_env/bin/activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** YOLOv8 models (`yolov8n.pt` and `yolov8n-pose.pt`) download automatically on first run. This requires an internet connection and takes 1-2 minutes.

---

### Step 4 — Run the app

```bash
streamlit run app.py
```

Open your browser and go to:
```
http://localhost:8501
```

---

## 🔐 Default Login Credentials

| Username | Password | Role |
|---|---|---|
| admin | admin123 | Admin |

> ⚠️ **Change your password immediately after first login** via the sidebar Change Password panel.

---

## 📷 Input Sources

| Source | How to use |
|---|---|
| **Webcam** | Select Webcam → press ▶ Start |
| **Upload video** | Select Upload video → upload MP4/AVI/MOV → press ▶ Start |
| **Upload image** | Select Upload image → upload JPG/PNG → detects instantly |
| **RTSP URL** | Select RTSP URL → enter camera stream URL → press ▶ Start |

### RTSP URL format for CCTV cameras:
```
rtsp://username:password@camera_ip:554/stream
```

---

## 🌐 Remote Access via ngrok

To access the app from any device anywhere:

**Terminal 1 — Start the app:**
```bash
streamlit run app.py --server.port 8501
```

**Terminal 2 — Start ngrok tunnel:**
```bash
.\ngrok http 8501
```

Copy the `https://xxxx.ngrok-free.app` URL and open it on any device.

> 💡 Get a free permanent URL at [dashboard.ngrok.com/domains](https://dashboard.ngrok.com/domains) so the link never changes.

---

## 🏫 Classroom Setup Guide

### Recommended hardware
- Any PC or laptop running Windows 10/11
- Webcam or existing CCTV/IP camera
- School WiFi or LAN connection

### Optimal camera placement
```
                    📷 Camera (ceiling mount)
                          │
                          ▼
    ┌─────────────────────────────────────────┐
    │  [Teacher Zone]  │    [Student Area]    │
    │   Left 30%       │      Right 70%       │
    │                  │  🪑🪑🪑🪑🪑🪑🪑🪑  │
    │   👨‍🏫           │  🪑🪑🪑🪑🪑🪑🪑🪑  │
    │                  │  🪑🪑🪑🪑🪑🪑🪑🪑  │
    └─────────────────────────────────────────┘
```

> The left 30% of the frame is the **Teacher Zone**. Anyone detected there is labelled as Teacher. Adjust `TEACHER_ZONE` in `config.py` if your camera is positioned differently.

---

## 🎛️ Settings Guide

| Setting | What it does |
|---|---|
| **Confidence threshold** | Minimum YOLO confidence to count a detection (0.4 recommended) |
| **Detect raised hands** | Enables YOLOv8 pose estimation for hand raise detection |
| **Enable logging** | Saves attendance data to CSV on every frame |
| **Input source** | Choose between Webcam, Video, Image, or RTSP |

---

## 📊 Dashboard Sections

| Section | Description |
|---|---|
| 📹 Live Feed | Annotated video with bounding boxes and labels |
| 📊 Live Metrics | Student count, hands raised, active zones |
| 🔔 Hand Raise Alerts | Real-time notifications with student position |
| 📋 Hand Raise History | Timestamped log of all hand raise events |
| 💻 Instrument Chart | Bar chart of detected items per session |
| 🗺️ Seat Heatmap | 10x10 grid showing presence density |
| 📈 Attendance History | Line chart of student count over time |
| 👥 Engagement Scores | Per-seat occupancy percentage scoring |
| 🎥 Session Recordings | Download or delete recorded sessions |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| App starts then immediately stops | Wait 2 minutes — YOLO models are loading |
| `localhost:8501` not opening | Check terminal for Python errors, run `streamlit run app.py` directly |
| Webcam not detected | Use `cv2.VideoCapture(0, cv2.CAP_DSHOW)` — already set in code |
| Other devices can't connect | Use ngrok instead of local network on school/corporate WiFi |
| Blinking video feed | Already fixed with `@st.fragment` — reduce `run_every` if still slow |
| Hand raise not detecting | Lower confidence threshold to 0.3, ensure good lighting |
| `KeyError: student_count` | Delete `data/attendance_log.csv` and restart |

---

## 🚀 Future Improvements

- [ ] Face recognition for automatic attendance by name
- [ ] Emotion detection (happy, confused, bored)
- [ ] Multi-camera support for multiple classrooms
- [ ] SMS/Email alerts when teacher is absent too long
- [ ] PDF report generation per session
- [ ] Mobile app version
- [ ] Integration with school management systems

---

## 👥 User Roles

| Role | Permissions |
|---|---|
| **Admin** | Full access, user management, all features |
| **Teacher** | Detection, logging, recordings, hand raise alerts |
| **Viewer** | View dashboard and history only, no controls |

---

## 📝 License

This project is for educational purposes.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection and pose estimation
- [Streamlit](https://streamlit.io) — web dashboard framework
- [OpenCV](https://opencv.org) — computer vision and frame processing
- [ngrok](https://ngrok.com) — secure tunneling for remote access

---

## 📞 Contact

Built as part of an AI classroom intelligence project.

> ⭐ If you found this useful, give it a star on GitHub!

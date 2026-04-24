# 🚗 Driver Drowsiness Detection System

A real-time driver drowsiness detection system using **Computer Vision** and **Deep Learning** to prevent road accidents caused by driver fatigue.

## 📌 Features

- **Real-time eye state detection** using a custom CNN model
- **Eye Aspect Ratio (EAR)** monitoring via Dlib 68-point facial landmarks
- **Mouth Aspect Ratio (MAR)** for yawn detection
- **Adaptive preprocessing** — works under varying lighting conditions (day/night)
- **Personalized calibration** — adjusts EAR threshold per user
- **Temporal smoothing** — rolling averages & majority voting to reduce false alarms
- **Audio alarm** with hysteresis-based triggering
- **Telegram alerts** with snapshot & location on drowsiness events

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| Deep Learning | PyTorch (Custom CNN) |
| Face Detection | Dlib (68-point landmarks) |
| Computer Vision | OpenCV (CLAHE, Gamma Correction) |
| Audio | Pygame |
| Alerts | Telegram Bot API |

## 📂 Project Structure

```
├── main.py                  # Main detection loop & DrowsinessDetector class
├── model.py                 # CNN architecture (DrowsinessCNN)
├── train.py                 # Model training script
├── config.py                # Thresholds & configuration constants
├── alert_system.py          # Telegram notification system
├── data_collector.py        # Dataset collection utility
├── download_dataset.py      # Dataset downloader
├── create_alarm.py          # Alarm sound generator
├── requirements.txt         # Python dependencies
├── drowsines_model.pth      # Trained CNN weights
├── alarm.wav                # Alarm sound file
└── haarcascade_*.xml        # Haar cascade classifiers
```

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- Webcam

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/driver-drowsiness-detection.git
cd driver-drowsiness-detection

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Download Required Files

The following large files are **not included** in the repository and must be downloaded separately:

1. **Shape Predictor** (`shape_predictor_68_face_landmarks.dat`) — Auto-downloaded on first run from [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
2. **LBF Model** (`lbfmodel.yaml`) — Download from [OpenCV Extra](https://github.com/kurnianggoro/GSOC2017/blob/master/data/lbfmodel.yaml)

## 🚀 Usage

```bash
python main.py
```

1. **Calibration Phase** — Look at the camera normally for ~5 seconds
2. **Detection Phase** — The system monitors your eyes and mouth in real-time
3. Press **`q`** to quit

## 🧠 How It Works

1. **Adaptive Preprocessing** — Gamma correction + CLAHE normalize lighting
2. **Face & Landmark Detection** — Dlib detects face and 68 facial landmarks
3. **EAR Calculation** — Eye Aspect Ratio measures eye openness geometrically
4. **CNN Classification** — Custom CNN classifies eye crops as Open/Closed
5. **Ensemble Scoring** — EAR + CNN results combined with weighted scoring
6. **Yawn Detection** — MAR monitors mouth openness for yawns
7. **Alarm Trigger** — Score exceeds threshold → audio alarm + Telegram alert

## 📄 License

This project is developed as an **8th Semester Final Year Project** for academic purposes.

## 👨‍💻 Author

**Rajdeepsinh**

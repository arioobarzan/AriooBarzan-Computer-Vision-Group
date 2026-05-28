# Anti-Slouch Alarm

Real-time posture monitor and slouch alarm using your webcam.

Pressing 'c' calibrates your upright sitting posture. After calibration the
script continuously tracks shoulder position, head posture, and screen
proximity. If you slouch, lean forward, or get too close to the monitor for
more than 3 seconds, a red warning overlay appears and an audible beep plays.

## Demo

[![Anti-Slouch Alarm Demo](https://img.youtube.com/vi/66qplDZo6a0/maxresdefault.jpg)](https://www.youtube.com/watch?v=66qplDZo6a0)

## How It Works

| Metric | What It Tracks | Bad Posture Sign |
|--------|---------------|------------------|
| **Shoulder Y** | Vertical position of shoulders | Drops → slouching |
| **Head/Shoulder** | Ear-to-shoulder vertical distance | Shrinks → head slouches forward |
| **Eye Distance** | Inter-ocular (eye-to-eye) distance | Grows → face too close to screen |

All three are compared against your personal baseline captured during calibration.

## Usage

```bash
pip install -r requirements.txt
python anti_slouch_alarm.py
```

1. Sit upright with good posture facing the webcam.
2. Press **`c`** to calibrate (your current posture becomes the baseline).
3. Work normally. The banner stays green while your posture is good.
4. If you slouch for more than 3 seconds the screen flashes red and a beep sounds.
5. Press **`r`** to recalibrate at any time, **`q`** to quit.

## Model File

This project needs the MediaPipe Pose Landmarker model. Place it at
`Sample Codes/common/models/pose_landmarker.task`.

```bash
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile "Sample Codes/common/models/pose_landmarker.task"

# macOS / Linux
wget -O "Sample Codes/common/models/pose_landmarker.task" \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

If you prefer the full (heavier) model, replace `pose_landmarker_lite` with
`pose_landmarker` in the URL above.

## Requirements

- Python 3.8+
- Webcam
- MediaPipe Pose Landmarker model file (see above)

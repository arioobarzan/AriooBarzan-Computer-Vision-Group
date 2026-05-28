# Fingertip Detection with MediaPipe

Real-time fingertip detection using MediaPipe Hand Landmarker. Detects 5 fingertips per hand and draws colored circles on each one. When both hands are detected, matching fingertips are connected with gradient lines (thick & bright in the middle, thin & saturated at the ends).

## Demo

[![HandTask Demo](https://img.youtube.com/vi/TezkHG0_X_E/maxresdefault.jpg)](https://www.youtube.com/shorts/TezkHG0_X_E)

- 5 colored circles on each fingertip (thumb, index, middle, ring, pinky)
- Gradient lines connecting matching fingertips between two hands
- Smooth glow effect: thick & bright center, thin & saturated ends
- No skeleton, no extra lines — just fingertips

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)

## Installation

```bash
pip install opencv-python mediapipe numpy
```

## Model File

Download the MediaPipe Hand Landmarker model and place it in the project directory:

```bash
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "hand_landmarker.task"

# macOS / Linux
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Usage

```bash
python fingertips_mediapipe.py
```

Press `q` to quit.

## Fingertip Colors

| Finger | Color   |
|--------|---------|
| Thumb  | Blue    |
| Index  | Green   |
| Middle | Red     |
| Ring   | Cyan    |
| Pinky  | Magenta |

## Landmark Indices

```
4  = Thumb tip
8  = Index finger tip
12 = Middle finger tip
16 = Ring finger tip
20 = Pinky tip
```

## License

MIT

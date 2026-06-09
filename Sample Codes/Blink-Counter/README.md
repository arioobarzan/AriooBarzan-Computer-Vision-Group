# Blink Counter

Count your eye blinks in real time using the **Eye Aspect Ratio (EAR)**
formula on MediaPipe Face Mesh landmarks.

## How It Works

The EAR formula measures how open each eye is by comparing the vertical
distance between eyelids to the horizontal width of the eye:

```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```

- **EAR > 0.20** → eye is open
- **EAR < 0.20** → eye is closing/closed

A state machine with frame-level debounce prevents the counter from
incrementing multiple times during a single long blink.

## Usage

```bash
pip install -r requirements.txt
python blink_counter.py
```

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **R** | Reset counter |

## Visual Feedback

- **Counter** — large green number, top-left
- **EAR value** — current averaged EAR, top-centre
- **EAR progress bar** — fills green (open) / red (closed), bottom
- **Threshold marker** — orange line on the bar at 0.20
- **Eye-landmark dots** — blue dots on the 12 EAR points
- **Green flash + "BLINK!"** on every detected blink

## Requirements

- Python 3.8+
- Webcam
- MediaPipe Face Landmarker model (`face_landmarker.task`) at `Sample Codes/common/models/`

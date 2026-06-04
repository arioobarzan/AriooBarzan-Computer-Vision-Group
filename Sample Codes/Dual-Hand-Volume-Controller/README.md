# Dual-Hand Volume Controller

Control your Windows master volume by moving your hands in the air.

**Move hands apart → volume up.** **Move hands close → volume down.**
No knobs, no keyboards — just gesture.

## Demo

[![Dual-Hand Volume Controller Demo](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

*Replace `VIDEO_ID` with your YouTube video ID after uploading a demo.*

## How It Works

1. **MediaPipe Hands** detects both hands and tracks the wrist position of each.
2. The **Euclidean distance** between the two wrists is normalised by the frame diagonal.
3. Press **`c`** to calibrate: spread your hands wide — that span becomes 100 % volume.
4. The distance is mapped linearly to **0–100 %** and a moving-average filter smooths out jitter.
5. The **Windows Core Audio API** (via pure `ctypes`, no extra dependencies) sets the system master volume in real time.

## Usage

```bash
pip install -r requirements.txt
python dual_hand_volume_controller.py
```

| Key | Action |
|-----|--------|
| **C** | Calibrate max distance (spread hands wide first) |
| **Q** | Quit |

## Visual Feedback

- **Volume bar** (right side) — fills green (high) to red (low)
- **Connection line** between wrists — thickness and colour reflect volume
- **Distance & percentage** label at the midpoint
- **Status banner** at top — green when active, blue when waiting

## Requirements

- Python 3.8+
- Windows 10 / 11 (Core Audio API)
- Webcam
- MediaPipe Hand Landmarker model (shared at `Sample Codes/common/models/`)

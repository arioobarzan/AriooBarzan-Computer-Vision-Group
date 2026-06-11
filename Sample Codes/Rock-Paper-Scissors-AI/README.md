# Rock-Paper-Scissors AI

Play Rock-Paper-Scissors against the computer using your webcam and
hand gestures.

## How It Works

A **3-second countdown** runs automatically. When it hits zero, your
hand gesture is captured, the AI picks randomly, and the winner is
announced. The scoreboard keeps track across rounds.

| Gesture | Rule | Finger State |
|---------|------|-------------|
| **Rock** | Beats Scissors | All 5 fingers folded |
| **Paper** | Beats Rock | All 5 fingers extended |
| **Scissors** | Beats Paper | Only index + middle extended |

## Usage

```bash
pip install -r requirements.txt
python rock_paper_scissors_ai.py
```

| Key | Action |
|-----|--------|
| **Q** | Quit |

Show your hand clearly to the camera. The current detected gesture is
shown at the top of the screen during the countdown.

## Requirements

- Python 3.8+
- Webcam
- MediaPipe Hand Landmarker model (shared at `Sample Codes/common/models/`)

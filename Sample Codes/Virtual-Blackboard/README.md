# Virtual Blackboard

Draw on a digital canvas using hand gestures in the air — no mouse, no
stylus, just your fingers.

| Gesture | Mode | Action |
|---------|------|--------|
| Index finger only | **Draw** | White ink on black canvas |
| Index + middle fingers | **Erase** | Thick black wiper clears content |
| Any other gesture | **Idle** | No marks, stroke resets |

## How It Works

1. **MediaPipe Hands** tracks 21 landmarks in real time.
2. Finger extension is checked by comparing fingertip vs PIP-joint
   Y positions — if the tip is above the joint, the finger is "up".
3. A **state machine** classifies your gesture on every frame:
   - Index only → **DRAW** (continuous white line)
   - Index + middle → **ERASE** (thick black circle)
   - Anything else → **IDLE** (stroke resets, no stray marks)
4. Drawing happens on a persistent black NumPy canvas shown in a separate
   window.

## Usage

```bash
pip install -r requirements.txt
python virtual_blackboard.py
```

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **C** | Clear canvas |

Two windows open: the webcam feed (with gesture feedback) and the
black canvas where your drawing appears.

## Requirements

- Python 3.8+
- Webcam
- MediaPipe Hand Landmarker model (shared at `Sample Codes/common/models/`)

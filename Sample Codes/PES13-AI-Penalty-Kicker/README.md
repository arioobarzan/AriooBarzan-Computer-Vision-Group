# PES 2013 AI Penalty Kicker

Control penalty shootouts in PES 2013 using hand gestures via webcam.

## How It Works

| Action | Gesture | Key |
|--------|---------|-----|
| **Aim Left** | Hand in left zone | Left Arrow held |
| **Aim Centre / Up** | Hand in centre zone | Up Arrow held |
| **Aim Right** | Hand in right zone | Right Arrow held |
| **Fill Power** | Close hand into a fist | 'A' held |
| **Shoot** | Open palm | 'A' released |

The direction key is **always held** while your hand is visible — it never
drops. The aim zone is determined by the wrist X position (stable, not
fingertip jitter).

Keys are injected via both `SendInput` (scan codes) and `keybd_event` for
maximum game compatibility.

## Usage

```bash
pip install -r requirements.txt
python pes_penalty_kicker.py
```

| Key | Action |
|-----|--------|
| **Q** | Quit |

## Visual HUD

- 3 coloured zones (LEFT, UP, RIGHT) — the active zone is highlighted darker
- 5 coloured fingertip dots + wrist dot with real-time extension ratios
- Bottom status bar shows current state and which keys are held

## Requirements

- Python 3.8+
- Windows (uses Win32 keyboard APIs)
- Webcam
- MediaPipe Hand Landmarker model (shared at `Sample Codes/common/models/`)

# Jedi Force Push

Thrust your palm toward the camera and unleash a shockwave — Jedi style.

## How It Works

The **wrist-to-middle-MCP distance** serves as a depth proxy: it grows when your
hand moves closer to the camera. A short rolling-average baseline is maintained,
and when the current depth **spikes** significantly above that baseline, a
"Force Push" is triggered.

## Effects

| Effect | Description |
|--------|-------------|
| **Shockwave rings** | 4 concentric blue-white rings expand from the palm |
| **Sparks** | 30 particles radiate outward from the push centre |
| **Screen shake** | 10-frame random jitter with decay for impact feel |
| **Cooldown** | 1.5 s between pushes to prevent spam |

## Usage

```bash
pip install -r requirements.txt
python jedi_force_push.py
```

| Key | Action |
|-----|--------|
| **Q** | Quit |

Open your palm facing the camera, then quickly thrust it forward.
The depth bar at the top right shows your current hand distance.

## Requirements

- Python 3.8+
- Webcam
- MediaPipe Hand Landmarker model (shared at `Sample Codes/common/models/`)

# Computer Vision Group

**A curated collection** — open-source computer vision projects, sample code, articles, papers, and learning resources.

---

## Repository Structure

```
.
├── Sample Codes/              # Ready-to-run CV projects
│   ├── common/                #   Shared utilities (camera, drawing, MediaPipe helpers)
│   ├── HandTask/              #   Fingertip detection with gradient hand connections
│   ├── Hand-Clap-Counter/     #   Real-time hand-clap counter via webcam
│   ├── Anti-Slouch-Alarm/     #   Posture monitor with slouch alarm
│   └── Virtual-Drum-Kit/      #   Play drums in the air with your fingers
├── Articles/                  # (coming) Blog posts, tutorials, and write-ups
├── Books/                     # (coming) Book notes, summaries, and references
├── Papers/                    # (coming) Paper reviews and implementations
├── README.md
└── .gitignore
```

## Sample Codes

Each project in `Sample Codes/` is self-contained with its own `README.md`, `requirements.txt`, and source code.

| Project | Description | Tech |
|---------|-------------|------|
| [**HandTask**](Sample%20Codes/HandTask/) | Detect 5 fingertips per hand. Connect matching fingers between two hands with glowing gradient lines. | MediaPipe Tasks API, OpenCV, NumPy |
| [**Hand Clap Counter**](Sample%20Codes/Hand-Clap-Counter/) | Count hand claps in real time via webcam. Live counter overlay with visual feedback. | MediaPipe Hands, OpenCV |
| [**Anti-Slouch Alarm**](Sample%20Codes/Anti-Slouch-Alarm/) | Real-time posture monitor. Calibrate your upright posture, get alerted when you slouch or lean too close to the screen. | MediaPipe Pose, OpenCV |
| [**Virtual Drum Kit**](Sample%20Codes/Virtual-Drum-Kit/) | Play drums in the air. Tap circular pads with your index fingers to trigger sounds. Built-in synth sounds + support for custom WAV files. | MediaPipe Hands, Pygame, OpenCV |

### Shared Utilities (`Sample Codes/common/`)

| Module | Purpose |
|--------|---------|
| `camera.py` | `WebcamManager` — context-managed webcam capture with mirror flip |
| `drawing.py` | `draw_gradient_line` — smooth glow-gradient line renderer |
| `mediapipe_utils.py` | Cache clearing, GPU warning suppression, model-path resolution |

## Getting Started

### Prerequisites

- Python 3.8+
- A working webcam

### Running a Project

```bash
cd "Sample Codes/<project-folder>"
pip install -r requirements.txt
python <main-script>.py
```

Press `q` to quit any running project.

### Model Files

Projects using MediaPipe Landmarkers need model files in `Sample Codes/common/models/`.

| Model | File | Status |
|-------|------|--------|
| Hand Landmarker | `hand_landmarker.task` | Included |
| Pose Landmarker | `pose_landmarker.task` | Download below |

**Download Pose Landmarker (required for Anti-Slouch Alarm):**

```bash
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile "Sample Codes/common/models/pose_landmarker.task"

# macOS / Linux
wget -O "Sample Codes/common/models/pose_landmarker.task" \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## Contributing

This repository is open to contributions:

1. Create a new folder under the appropriate section (e.g., `Sample Codes/`, `Articles/`)
2. Include a `README.md`, `requirements.txt` (if code), and your source files
3. Use the shared `Sample Codes/common/` utilities where applicable
4. Open a Pull Request

Keep all comments, docstrings, and documentation in English.

## License

MIT

---

**Made for the Computer Vision community**

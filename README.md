# Computer Vision Group

**Open-source computer vision projects** — real-time hand tracking, gesture recognition, object detection, and more. Every project is self-contained, well-documented, and ready to run.

---

## Repository Structure

```
.
├── common/                   # Shared utilities (camera, drawing, MediaPipe helpers)
├── HandTask/                 # Fingertip detection with gradient hand connections
├── Hand-Clap-Counter/        # Real-time hand-clap counter via webcam
└── README.md
```

Each project lives in its own folder with its own `README.md`, `requirements.txt`, and source code.

## Projects

| Project | Description | Tech |
|---------|-------------|------|
| [**HandTask**](HandTask/) | Detect 5 fingertips per hand. Connect matching fingers between two hands with glowing gradient lines. | MediaPipe Tasks API, OpenCV, NumPy |
| [**Hand Clap Counter**](Hand-Clap-Counter/) | Count hand claps in real time via webcam. Live counter overlay with visual feedback. | MediaPipe Hands, OpenCV |

## Shared Utilities (`common/`)

| Module | Purpose |
|--------|---------|
| `camera.py` | `WebcamManager` — context-managed webcam capture with mirror flip |
| `drawing.py` | `draw_gradient_line` — smooth glow-gradient line renderer |
| `mediapipe_utils.py` | Cache clearing and GPU warning suppression for MediaPipe |

## Getting Started

### Prerequisites

- Python 3.8+
- A working webcam

### Running a Project

```bash
cd <project-folder>
pip install -r requirements.txt
python <main-script>.py
```

Press `q` to quit any running project.

### HandTask — Additional Setup

The HandTask project requires the MediaPipe Hand Landmarker model file. Download it into the project folder:

```bash
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "HandTask/hand_landmarker.task"

# macOS / Linux
wget -P HandTask/ https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Contributing

This repository is open to contributions. If you have a computer vision project you would like to add:

1. Create a new folder with your project name
2. Include a `README.md`, `requirements.txt`, and your source code
3. Use the shared `common/` utilities where applicable
4. Open a Pull Request

Keep all comments, docstrings, and documentation in English.

## License

MIT

---

**Made for the Computer Vision community**

# GNM Face Tracker

Real-time face tracking using Google's **GNM Head** parametric 3D model and
MediaPipe Face Mesh.

## How It Works

| Stage | Description |
|-------|-------------|
| **1. Identity Fit** | You sit with a neutral expression.  MediaPipe detects 478 face landmarks; we map 68 of them to GNM's standard facial landmarks and optimise the 253 identity parameters via L-BFGS-B. |
| **2. Real-time Tracking** | Each frame, MediaPipe landmarks are aligned to the fitted identity, and the residual is projected onto GNM's 383 expression blendshapes via regularised least squares.  The deformed mesh is rendered side-by-side with the webcam. |

### Pipeline

```
Webcam → MediaPipe Face Mesh → 478 3D Landmarks
                                   │
                    ┌──────────────┴──────────────┐
                    │  Procrustes similarity align │  → map to GNM space
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │  Subtract identity contribution          │
              │  Regularised least squares → expression  │
              └────────────────────┬────────────────────┘
                                   │
                        ┌──────────┴──────────┐
                        │   GNM Forward Pass   │  identity + expression
                        └──────────┬──────────┘
                                   │
                         17,821-vertex Mesh
```

## Requirements

- Python 3.10+
- **GNM Head** — clone and install from [google/GNM](https://github.com/google/GNM):

```bash
git clone https://github.com/google/GNM.git
cd GNM/gnm/shape
pip install -e .
```

- Project dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

| Key | Action |
|-----|--------|
| **Space** | Capture neutral face to fit identity |
| **R** | Re-fit identity |
| **F** | Toggle fullscreen GNM mesh view |
| **Q** | Quit |

## GNM Model

| Parameter | Dims | Description |
|-----------|------|-------------|
| **Identity** | 253 | 170 head + 80 teeth + 3 eyeball shape |
| **Expression** | 383 | 100 L-eye + 100 R-eye + 150 lower face + 32 tongue + 1 iris |
| **Pose** | 4 joints × 3 | Neck, head, L/R eyeball rotation (axis-angle) |

The mesh has **17,821 vertices** covering skin, eyeballs, teeth, and tongue
in a single unified topology.

## Architecture Notes

- Identity fitting uses 3 iterations of alternating similarity-transform
  estimation and L-BFGS-B optimisation, since MediaPipe's coordinate space
  (normalised image + metric depth) differs from GNM's world space (metres).
- The expression regressor is a pre-computed (383 × 204) matrix that solves
  the regularised least-squares problem in closed form — < 1 ms per frame.
- Mesh rendering uses a flat-shaded software rasteriser (NumPy + OpenCV
  fillPoly) — no GPU needed for the overlay.

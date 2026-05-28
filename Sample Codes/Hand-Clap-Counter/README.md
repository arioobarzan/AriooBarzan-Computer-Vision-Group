# Hand Clap Counter

Real-time hand-clap counter using your webcam and MediaPipe. Detects both hands, tracks when they come together, and keeps a live count displayed on screen.

## Demo

[![Hand Clap Counter Demo](https://img.youtube.com/vi/XfMXq6csxyQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=XfMXq6csxyQ)

- Live clap counter with large on-screen number
- Green flash effect on every detected clap
- Cooldown bar to prevent double-counting
- Hand-count indicator (shows how many hands are visible)

## Requirements

- Python 3.8+
- Webcam

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python clap_counter.py
```

Clap your hands in front of the camera. Press `q` to quit.

## How It Works

1. **MediaPipe Hands** detects up to 2 hands and extracts 21 landmarks per hand.
2. The **wrist distance** (landmark 0 of each hand) is calculated every frame.
3. When the distance drops below `CLAP_DISTANCE_THRESHOLD` (default: `0.12`), a clap is registered.
4. A **cooldown** (25 frames) and **separation requirement** (8 frames) prevent counting the same clap multiple times.
5. Visual feedback — green flash, skeleton overlay, and counter — is rendered in real time.

## Configuration

Tune these constants at the top of `clap_counter.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CLAP_DISTANCE_THRESHOLD` | `0.12` | Normalised wrist distance that triggers a clap (lower = closer) |
| `COOLDOWN_FRAMES` | `25` | Minimum frames between consecutive claps |
| `SEPARATION_FRAMES` | `8` | Frames hands must stay apart before next clap |

## License

MIT

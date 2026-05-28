# Virtual Drum Kit

Play drums in the air using your webcam and index fingers. Four circular
drum pads are overlaid on the video feed. Tap inside a pad with either
index finger to trigger the corresponding sound.

Synthesised placeholder sounds are built in — just run and play. Drop
real `.wav` files into the project folder to replace them.

## Drum Pads

| Pad | Colour | Default Sound |
|-----|--------|---------------|
| **Snare** | Orange | Noise burst with low-frequency body |
| **Hi-Hat** | Blue | Short high-frequency hiss |
| **Tom** | Green | Mid-range hit |
| **Crash** | Pink | Bright crash cymbal sweep |

## Usage

```bash
pip install -r requirements.txt
python virtual_drum_kit.py
```

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **D** | Toggle hand skeleton overlay (debug) |

### Using Your Own Sounds

Drop `.wav` files named `snare.wav`, `hihat.wav`, `tom.wav`, and
`crash.wav` into the project folder. They are loaded automatically on
startup. If a file is missing the built-in synth fallback is used.

## How It Works

1. **MediaPipe Hands** tracks 21 landmarks per hand in real time.
2. The **index-finger tip** (landmark 8) of each hand is tested against
   every drum pad's circular ROI.
3. When a fingertip enters a pad that is **not already triggered** and
   has **no active cooldown**, the corresponding sound plays.
4. An **anti-spam** state machine prevents machine-gun re-triggers:
   - The finger must **leave** the pad before it can be struck again.
   - An 18-frame **cooldown** (~0.5 s) adds an extra guard.

## Requirements

- Python 3.8+
- Webcam
- Hand Landmarker model file at `Sample Codes/common/models/hand_landmarker.task`

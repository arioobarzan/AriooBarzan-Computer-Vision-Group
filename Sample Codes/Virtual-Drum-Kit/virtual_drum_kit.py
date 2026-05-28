"""
Virtual Drum Kit with MediaPipe Hands
======================================
Play drums in the air using your index fingers and a webcam.

Four circular drum pads are overlaid on the video feed.  Tap your index
finger inside a pad to trigger the corresponding sound.  Built-in
synthesised sounds let you play immediately — drop real ``.wav`` files
into the project folder to replace them.

Press 'q' to quit, 'd' to toggle debug skeleton.
"""

import io
import os
import platform
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

# Allow importing from the repo root (Sample Codes/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (
    WebcamManager,
    clear_mediapipe_cache,
    get_model_path,
    HAND_MODEL_FILENAME,
    suppress_gpu_warnings,
)

# ---------------------------------------------------------------------------
# Environment & model
# ---------------------------------------------------------------------------
clear_mediapipe_cache()
suppress_gpu_warnings()
MODEL_PATH = get_model_path(HAND_MODEL_FILENAME)

# ---------------------------------------------------------------------------
# MediaPipe setup (Tasks API)
# ---------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INDEX_TIP = 8                     # MediaPipe hand landmark index for index-finger tip
COOLDOWN_FRAMES = 18              # frames (~0.5 s at 30 fps) before a pad can re-trigger
PAD_RADIUS_FRAC = 0.11            # drum-pad radius as fraction of min(frame_w, frame_h)
PAD_Y_FRAC = 0.20                 # vertical centre of pads relative to frame height
SAMPLE_RATE = 22050

_IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# Audio player (zero external dependencies)
# ---------------------------------------------------------------------------
class _Sound:
    """A playable sound backed by in-memory WAV bytes."""

    def __init__(self, wav_bytes: bytes) -> None:
        self._data = wav_bytes
        if not _IS_WINDOWS:
            # Write to a temp file so afplay / aplay can read it
            self._tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            self._tmp.write(wav_bytes)
            self._tmp.close()
            self._path = self._tmp.name

    def play(self) -> None:
        """Play the sound asynchronously."""
        if _IS_WINDOWS:
            import winsound
            winsound.PlaySound(self._data, winsound.SND_MEMORY | winsound.SND_ASYNC)
        else:
            cmd = ["afplay", self._path] if platform.system() == "Darwin" else ["aplay", "-q", self._path]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def __del__(self) -> None:
        if not _IS_WINDOWS and hasattr(self, "_path"):
            try:
                os.unlink(self._path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Synthesised drum sounds
# ---------------------------------------------------------------------------
def _make_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert a float numpy array (-1..1) to WAV bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _synth_snare(sr: int = SAMPLE_RATE) -> bytes:
    """Short noise burst with a low-frequency body (snare-like)."""
    dur = 0.15
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.normal(0, 0.6, len(t))
    tone = np.sin(2 * np.pi * 180 * t)
    env = np.exp(-t * 25)
    return _make_wav_bytes((noise * 0.65 + tone * 0.35) * env, sr)


def _synth_hihat(sr: int = SAMPLE_RATE) -> bytes:
    """Very short high-frequency hiss (hi-hat)."""
    dur = 0.06
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.normal(0, 0.8, len(t))
    tone = np.sin(2 * np.pi * 8000 * t)
    env = np.exp(-t * 80)
    return _make_wav_bytes((noise * 0.5 + tone * 0.5) * env, sr)


def _synth_cymbal(sr: int = SAMPLE_RATE) -> bytes:
    """Longer bright noise sweep (crash cymbal)."""
    dur = 0.45
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.normal(0, 0.7, len(t))
    tone = np.sin(2 * np.pi * (6000 + 2000 * np.exp(-t * 8)) * t)
    env = np.exp(-t * 5)
    return _make_wav_bytes((noise * 0.55 + tone * 0.45) * env, sr)


# ---------------------------------------------------------------------------
# DrumPad dataclass
# ---------------------------------------------------------------------------
@dataclass
class DrumPad:
    """A single drum pad: its ROI, colour, sound, and trigger state."""

    name: str
    color: tuple[int, int, int]               # BGR
    sound_file: str                            # e.g. "snare.wav"
    synth_func: object                         # callable → bytes

    # Runtime state (managed by VirtualDrumKit)
    cx: int = 0
    cy: int = 0
    radius: int = 0
    triggered: bool = False
    cooldown: int = 0


# ---------------------------------------------------------------------------
# VirtualDrumKit
# ---------------------------------------------------------------------------
class VirtualDrumKit:
    """Real-time virtual drum kit driven by hand tracking."""

    def __init__(self, camera_id: int = 0, cam_width: int = 1280, cam_height: int = 720):
        # Webcam
        self.cam = WebcamManager(camera_id=camera_id, width=cam_width, height=cam_height)

        # MediaPipe
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.detector = HandLandmarker.create_from_options(options)

        # Drum pads
        self.pads: list[DrumPad] = []
        self._init_pads()

        # Debug toggle
        self.show_skeleton = False
        self.frame_idx = 0

    # ------------------------------------------------------------------
    # Pad setup
    # ------------------------------------------------------------------
    def _init_pads(self) -> None:
        """Create the drum pads with loaded sounds."""
        definitions = [
            ("Snare", (255, 100, 50),  "snare.wav",  _synth_snare),
            ("HiHat", (50, 220, 255),  "hihat.wav",  _synth_hihat),
            ("Tom",   (50, 255, 120),  "tom.wav",    _synth_snare),
            ("Crash", (255, 50, 200),  "crash.wav",  _synth_cymbal),
        ]
        for name, col, filename, synth in definitions:
            pad = DrumPad(name=name, color=col, sound_file=filename, synth_func=synth)
            self._load_sound(pad)
            self.pads.append(pad)

    def _load_sound(self, pad: DrumPad) -> None:
        """Load a WAV file if it exists, otherwise create a synthesised fallback."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        wav_path = os.path.join(script_dir, pad.sound_file)
        if os.path.exists(wav_path):
            with open(wav_path, "rb") as f:
                pad._sound = _Sound(f.read())
        else:
            pad._sound = _Sound(pad.synth_func())

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _layout_pads(self, h: int, w: int) -> None:
        n = len(self.pads)
        r = int(min(w, h) * PAD_RADIUS_FRAC)
        cy = int(h * PAD_Y_FRAC)
        spacing = w / (n + 1)
        for i, pad in enumerate(self.pads):
            pad.cx = int(spacing * (i + 1))
            pad.cy = cy
            pad.radius = r

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------
    @staticmethod
    def _finger_in_pad(fx: int, fy: int, pad: DrumPad) -> bool:
        dx = fx - pad.cx
        dy = fy - pad.cy
        return (dx * dx + dy * dy) < (pad.radius * pad.radius)

    # ------------------------------------------------------------------
    # Update (called every frame)
    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._layout_pads(h, w)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(mp_image, self.frame_idx)
        self.frame_idx += 1

        # Collect fingertip positions
        fingertips: list[tuple[int, int]] = []
        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                tip = hand_lms[INDEX_TIP]
                fingertips.append((int(tip.x * w), int(tip.y * h)))

        # Collision + anti-spam trigger logic
        for pad in self.pads:
            finger_inside = any(self._finger_in_pad(fx, fy, pad) for fx, fy in fingertips)

            if finger_inside and not pad.triggered and pad.cooldown == 0:
                pad._sound.play()
                pad.triggered = True
                pad.cooldown = COOLDOWN_FRAMES

            if not finger_inside:
                pad.triggered = False

            if pad.cooldown > 0:
                pad.cooldown -= 1

        # Draw
        self._draw_pads(frame)
        self._draw_fingertips(frame, fingertips)
        if self.show_skeleton and result.hand_landmarks:
            self._draw_skeleton(frame, result.hand_landmarks, h, w)
        self._draw_hud(frame, h, w)

        return frame

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_pads(self, frame: np.ndarray) -> None:
        for pad in self.pads:
            alpha = 0.55 if pad.triggered else (0.35 + 0.20 * (1 - pad.cooldown / COOLDOWN_FRAMES) if pad.cooldown > 0 else 0.30)
            overlay = frame.copy()
            cv2.circle(overlay, (pad.cx, pad.cy), pad.radius, pad.color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (pad.cx, pad.cy), pad.radius, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

            (tw, th), _ = cv2.getTextSize(pad.name, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            cv2.putText(frame, pad.name, (pad.cx - tw // 2, pad.cy + th // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_fingertips(self, frame: np.ndarray,
                         fingertips: list[tuple[int, int]]) -> None:
        for fx, fy in fingertips:
            cv2.circle(frame, (fx, fy), 12, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (fx, fy), 12, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_skeleton(self, frame: np.ndarray, hand_landmarks_list, h: int, w: int) -> None:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]
        for hand_lms in hand_landmarks_list:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for a, b in connections:
                cv2.line(frame, pts[a], pts[b], (180, 180, 180), 1, cv2.LINE_AA)
            for x, y in pts:
                cv2.circle(frame, (x, y), 2, (200, 200, 200), -1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, h: int, w: int) -> None:
        cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.putText(frame, "Virtual Drum Kit — tap the pads with your index fingers",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, "Q=quit  D=toggle skeleton",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        y_off = h - 8
        x_cursor = w - 40
        for pad in reversed(self.pads):
            status = "RDY" if (not pad.triggered and pad.cooldown == 0) else ("CD" if pad.cooldown > 0 else "HIT")
            label = f"{pad.name}:{status}"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            x_cursor -= tw + 25
            cv2.putText(frame, label, (x_cursor, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        if self.show_skeleton:
            cv2.putText(frame, "SKEL", (w - 50, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def release(self) -> None:
        self.detector.close()
        self.cam.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        self.cam.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Model: {MODEL_PATH}")
    print("=" * 55)
    print("Virtual Drum Kit — MediaPipe Hands")
    print("  Tap the coloured pads with your index fingers.")
    print("  q = quit   |   d = toggle skeleton overlay")
    print("=" * 55)

    with VirtualDrumKit() as kit:
        while True:
            success, frame = kit.cam.read()
            if not success:
                break

            frame = kit.update(frame)
            cv2.imshow("Virtual Drum Kit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                kit.show_skeleton = not kit.show_skeleton

    print("Done.")


if __name__ == "__main__":
    main()

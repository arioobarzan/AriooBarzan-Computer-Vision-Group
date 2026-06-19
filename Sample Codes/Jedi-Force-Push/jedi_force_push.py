"""
Jedi Force Push with MediaPipe Hands
======================================
Thrust your palm toward the camera to unleash a shockwave!

- Tracks hand depth via wrist-to-MCP distance.
- Detects a sudden forward push (velocity spike on the distance signal).
- Triggers expanding concentric shockwave rings + screen shake.
- 1.5 s cooldown prevents machine-gun re-triggers.

Press 'q' to quit.
"""

import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# Allow importing from the repo root (Sample Codes/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (
    WebcamManager,
    clear_mediapipe_cache,
    get_model_path,
    suppress_gpu_warnings,
)

# ---------------------------------------------------------------------------
# Environment & model
# ---------------------------------------------------------------------------
clear_mediapipe_cache()
suppress_gpu_warnings()
MODEL_PATH = get_model_path()

# ---------------------------------------------------------------------------
# MediaPipe setup (Tasks API)
# ---------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
WRIST = 0
MIDDLE_MCP = 9

# ---------------------------------------------------------------------------
# Push-detection parameters
# ---------------------------------------------------------------------------
VELOCITY_WINDOW = 6         # frames used for baseline averaging
VELOCITY_THRESHOLD = 0.015  # normalised-distance delta that triggers push
COOLDOWN_SECS = 1.5         # minimum seconds between pushes

# ---------------------------------------------------------------------------
# VFX parameters
# ---------------------------------------------------------------------------
SHOCKWAVE_COUNT = 4         # concentric rings per push
SHOCKWAVE_MAX_RADIUS = 1.2  # fraction of frame diagonal
SHOCKWAVE_LIFETIME = 0.8    # seconds per ring
SHAKE_FRAMES = 10           # screen-shake duration in frames
SHAKE_MAX_OFFSET = 14       # max pixel jitter per axis
SPARK_COUNT = 30            # number of spark particles per push

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
GREEN = (0, 255, 80)
BLUE_CYAN = (255, 200, 80)
WHITE = (255, 255, 255)
DARK = (30, 30, 30)
GREY = (160, 160, 160)

JEDI_BLUE = (255, 180, 60)   # shockwave core colour
JEDI_WHITE = (255, 240, 210)  # shockwave outer colour


# ===================================================================
# FX data classes
# ===================================================================
@dataclass
class Shockwave:
    """A single expanding ring spawned on push."""
    cx: int
    cy: int
    radius: float = 0.0
    alpha: float = 1.0
    max_radius: float = 0.0
    lifetime: float = SHOCKWAVE_LIFETIME
    born: float = 0.0


@dataclass
class Spark:
    """A particle flying outward from the push centre."""
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple


# ===================================================================
# JediForcePush
# ===================================================================
class JediForcePush:
    """
    Detect a palm thrust toward the camera and render Jedi-style VFX.

    Depth is estimated from the wrist-to-middle-MCP distance (normalised).
    A velocity spike (delta above recent baseline) triggers the push.
    """

    def __init__(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self._detector = HandLandmarker.create_from_options(options)

        # Depth signal history for velocity computation
        self._depth_history: deque[float] = deque(maxlen=VELOCITY_WINDOW)
        self._last_push_time: float = 0.0

        # Active effects
        self._shockwaves: list[Shockwave] = []
        self._sparks: list[Spark] = []
        self._shake_frames: int = 0
        self._shake_seed: int = 0

        self._frame_idx: int = 0
        self._palm_center: Optional[tuple[int, int]] = None
        self._current_depth: float = 0.0
        self._push_this_frame: bool = False

    # ── context manager ───────────────────────────────────────────
    def __enter__(self) -> "JediForcePush":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "_detector"):
            self._detector.close()

    # ── depth signal ──────────────────────────────────────────────
    def _compute_depth(self, landmarks) -> float:
        """
        Wrist-to-middle-MCP distance, normalised.
        Increases when the hand moves toward the camera.
        """
        w = landmarks[WRIST]
        m = landmarks[MIDDLE_MCP]
        return np.hypot(m.x - w.x, m.y - w.y)

    # ── push detection ────────────────────────────────────────────
    def _detect_push(self, depth: float, now: float) -> bool:
        """
        Compare current depth against recent baseline.
        A sudden forward thrust = depth spike above the rolling average.
        """
        self._depth_history.append(depth)
        if len(self._depth_history) < VELOCITY_WINDOW:
            return False

        # Baseline = average of the window, excluding the current frame
        hist = list(self._depth_history)
        baseline = sum(hist[:-1]) / (len(hist) - 1)
        delta = depth - baseline

        # Must also respect cooldown
        if now - self._last_push_time < COOLDOWN_SECS:
            return False

        return delta > VELOCITY_THRESHOLD

    # ── spawn VFX ─────────────────────────────────────────────────
    def _trigger_push(self, cx: int, cy: int, now: float,
                      frame_diag: float) -> None:
        """Spawn shockwaves, sparks, and screen shake."""
        self._last_push_time = now
        self._shake_frames = SHAKE_FRAMES
        self._shake_seed = random.randint(0, 10000)

        # Shockwave rings
        for i in range(SHOCKWAVE_COUNT):
            sw = Shockwave(
                cx=cx, cy=cy,
                radius=0.0,
                max_radius=frame_diag * (0.15 + 0.25 * i),
                lifetime=SHOCKWAVE_LIFETIME + i * 0.12,
                born=now,
                alpha=1.0,
            )
            self._shockwaves.append(sw)

        # Sparks
        for _ in range(SPARK_COUNT):
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(3, 15)
            color = (
                random.randint(180, 255),   # B
                random.randint(100, 220),   # G
                random.randint(40, 120),    # R
            )
            spark = Spark(
                x=float(cx), y=float(cy),
                vx=np.cos(angle) * speed,
                vy=np.sin(angle) * speed,
                life=random.uniform(0.3, 0.9),
                max_life=0.0,
                color=color,
            )
            spark.max_life = spark.life
            self._sparks.append(spark)

    # ── update effects ────────────────────────────────────────────
    def _update_effects(self, now: float) -> None:
        """Age shockwaves, sparks, and shake counter."""
        # Shockwaves
        surviving = []
        for sw in self._shockwaves:
            elapsed = now - sw.born
            if elapsed >= sw.lifetime:
                continue
            progress = elapsed / sw.lifetime
            sw.radius = sw.max_radius * progress
            sw.alpha = 1.0 - progress
            surviving.append(sw)
        self._shockwaves = surviving

        # Sparks
        surviving = []
        for sp in self._sparks:
            sp.life -= (now - sp.born) if hasattr(sp, 'born') else 0.016
            if hasattr(sp, 'born'):
                sp.life -= (now - sp.born)
                sp.born = now
            else:
                sp.life -= 0.016
            sp.x += sp.vx
            sp.y += sp.vy
            if sp.life > 0:
                surviving.append(sp)
        # Note: sparks handled differently — we just use frame-based decay
        # Reset and redo properly below in draw method.

    # ── draw VFX ──────────────────────────────────────────────────
    @staticmethod
    def _draw_glow_circle(frame: np.ndarray, cx: int, cy: int,
                          radius: int, alpha: float, color: tuple) -> None:
        """Draw a single fading circle on an overlay."""
        overlay = np.zeros_like(frame)
        cv2.circle(overlay, (cx, cy), radius, color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0, 0, dst=frame)

    def _draw_shockwaves(self, frame: np.ndarray) -> None:
        """Render all active shockwave rings."""
        for sw in self._shockwaves:
            r = int(sw.radius)
            # Outer glow (thick, lower alpha)
            alpha_outer = sw.alpha * 0.4
            self._draw_glow_circle(frame, sw.cx, sw.cy, r,
                                   alpha_outer, JEDI_BLUE)
            # Inner bright ring (thin, higher alpha)
            self._draw_glow_circle(frame, sw.cx, sw.cy, r,
                                   sw.alpha * 0.85, JEDI_WHITE)

    def _apply_screen_shake(self, frame: np.ndarray) -> np.ndarray:
        """Shift the frame by a random offset for impact feel."""
        if self._shake_frames <= 0:
            return frame

        h, w = frame.shape[:2]
        # Decay intensity over the shake duration
        intensity = self._shake_frames / SHAKE_FRAMES
        max_offset = int(SHAKE_MAX_OFFSET * intensity)

        rng = random.Random(self._shake_seed + self._shake_frames)
        dx = rng.randint(-max_offset, max_offset)
        dy = rng.randint(-max_offset, max_offset)

        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shaken = cv2.warpAffine(frame, matrix, (w, h),
                                borderMode=cv2.BORDER_REPLICATE)
        return shaken

    # ── HUD ───────────────────────────────────────────────────────
    def _draw_hud(self, frame: np.ndarray, h: int, w: int, now: float) -> None:
        """Status banner + cooldown indicator."""
        # Top banner
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 58), DARK, -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, dst=frame)

        # Cooldown remaining
        cd_remain = max(0.0, COOLDOWN_SECS - (now - self._last_push_time))
        cd_text = f"Cooldown: {cd_remain:.1f}s" if cd_remain > 0 else "READY"
        cd_color = GREY if cd_remain > 0 else GREEN
        cv2.putText(frame, cd_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cd_color, 2, cv2.LINE_AA)

        # Depth indicator
        cv2.putText(frame, f"Depth: {self._current_depth:.4f}",
                    (w - 250, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1, cv2.LINE_AA)

        # Push prompt
        if self._shake_frames > 0:
            cv2.putText(frame, "FORCE PUSH!",
                        (w // 2 - 140, h // 2 - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 2.5, JEDI_WHITE, 4, cv2.LINE_AA)

        # Help
        cv2.putText(frame, "Q = quit  |  Thrust your palm toward the camera!",
                    (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, GREY, 1, cv2.LINE_AA)

    # ── update ────────────────────────────────────────────────────
    def update(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        diag = np.hypot(w, h)
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, self._frame_idx)
        self._frame_idx += 1

        self._push_this_frame = False
        self._palm_center = None

        # ── Hand detection ──
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            lms = result.hand_landmarks[0]

            # Palm centre (in pixels, for VFX origin)
            wrist = lms[WRIST]
            mcp = lms[MIDDLE_MCP]
            self._palm_center = (int((wrist.x + mcp.x) / 2 * w),
                                 int((wrist.y + mcp.y) / 2 * h))

            # Depth signal
            self._current_depth = self._compute_depth(lms)

            # Push detection
            if self._detect_push(self._current_depth, now):
                self._push_this_frame = True
                self._trigger_push(*self._palm_center, now, diag)
        else:
            self._current_depth = 0.0

        # ── Age effects ──
        self._update_effects(now)
        if self._shake_frames > 0:
            self._shake_frames -= 1

        # ── Update sparks (frame-based) ──
        surviving_s = []
        for sp in self._sparks:
            sp.life -= 0.025
            sp.x += sp.vx
            sp.y += sp.vy
            if sp.life > 0:
                surviving_s.append(sp)
        self._sparks = surviving_s

        # ── Draw VFX ──
        self._draw_shockwaves(frame)

        # Sparks
        for sp in self._sparks:
            alpha_spark = sp.life / sp.max_life
            try:
                xi, yi = int(sp.x), int(sp.y)
                if 0 <= xi < w and 0 <= yi < h:
                    cv2.circle(frame, (xi, yi), 1 + int(3 * alpha_spark),
                               sp.color, -1, cv2.LINE_AA)
            except Exception:
                pass

        # Screen shake
        frame = self._apply_screen_shake(frame)

        # Palm-centre dot
        if self._palm_center is not None:
            cv2.circle(frame, self._palm_center, 8, BLUE_CYAN, -1, cv2.LINE_AA)
            cv2.circle(frame, self._palm_center, 8, WHITE, 1, cv2.LINE_AA)

        # ── HUD ──
        self._draw_hud(frame, h, w, now)
        return frame


# ===================================================================
def main() -> None:
    print(f"Model: {MODEL_PATH}")
    print("=" * 52)
    print("Jedi Force Push -- MediaPipe Hands")
    print("  Thrust your open palm toward the camera to push!")
    print("  Shockwave rings + sparks + screen shake.")
    print("  q = quit")
    print("=" * 52)

    with JediForcePush() as jedi, \
         WebcamManager(camera_id=0, width=1280, height=720) as cam:

        while True:
            success, frame = cam.read()
            if not success:
                break

            frame = jedi.update(frame)
            cv2.imshow("Jedi Force Push", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("May the Force be with you.")


if __name__ == "__main__":
    main()

"""
Virtual Blackboard with MediaPipe Hands
=========================================
Draw directly on a dimmed webcam feed using hand gestures.

Gestures:
  - **Index finger only**  -> DRAW  (white line)
  - **Index + middle**     -> ERASE (thick wiper)
  - **Anything else**      -> IDLE  (no marks)

Single-window design: the webcam is darkened and drawing appears on top.

Press 'q' to quit, 'c' to clear all ink.
"""

import os
import sys
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
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

THUMB_IP = 3
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

# ---------------------------------------------------------------------------
# Finger-detection parameters
# ---------------------------------------------------------------------------
FINGER_UP_Y_MARGIN = 0.06   # tip must be clearly above PIP to count as "up"
THUMB_CLOSED_DIST = 0.12    # thumb tip near index MCP -> thumb is folded

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
DRAW_COLOR = (255, 255, 255)
DRAW_THICKNESS = 4
ERASE_RADIUS = 30

# Face brightness: how dim the webcam gets (0 = black, 1 = full)
WEBCAM_DIM = 0.4

# Colours (BGR)
GREEN = (0, 255, 80)
RED = (0, 50, 255)
ORANGE = (0, 165, 255)
GREY = (180, 180, 180)
WHITE = (255, 255, 255)

STATE_DRAW = "DRAW"
STATE_ERASE = "ERASE"
STATE_IDLE = "IDLE"


# ===================================================================
class VirtualBlackboard:
    """Gesture-driven drawing directly on the webcam feed."""

    def __init__(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self._detector = HandLandmarker.create_from_options(options)

        # Persistent ink layer (black = nothing, white = drawn)
        self._ink: Optional[np.ndarray] = None

        self._prev_pos: Optional[tuple[int, int]] = None
        self._state = STATE_IDLE
        self._frame_idx = 0

        # Debounce
        self._pending_state: Optional[str] = None
        self._pending_count = 0
        self._debounce_frames = 2

    # ── context manager ───────────────────────────────────────────
    def __enter__(self) -> "VirtualBlackboard":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "_detector"):
            self._detector.close()

    # ── finger detection ──────────────────────────────────────────
    @staticmethod
    def _is_finger_up(tip_lm, pip_lm) -> bool:
        return tip_lm.y < pip_lm.y - FINGER_UP_Y_MARGIN

    @staticmethod
    def _is_thumb_closed(tip_lm, index_mcp_lm) -> bool:
        dx = tip_lm.x - index_mcp_lm.x
        dy = tip_lm.y - index_mcp_lm.y
        return (dx * dx + dy * dy) < THUMB_CLOSED_DIST ** 2

    def _classify_gesture(self, landmarks) -> tuple[str, dict]:
        lms = landmarks
        thumb_closed = self._is_thumb_closed(lms[THUMB_TIP], lms[INDEX_MCP])
        index_up = self._is_finger_up(lms[INDEX_TIP], lms[INDEX_PIP])
        middle_up = self._is_finger_up(lms[MIDDLE_TIP], lms[MIDDLE_PIP])
        ring_up = self._is_finger_up(lms[RING_TIP], lms[RING_PIP])
        pinky_up = self._is_finger_up(lms[PINKY_TIP], lms[PINKY_PIP])

        states = {
            "thumb": "closed" if thumb_closed else "open",
            "index": "up" if index_up else "down",
            "middle": "up" if middle_up else "down",
            "ring": "up" if ring_up else "down",
            "pinky": "up" if pinky_up else "down",
        }

        if (index_up and not middle_up and not ring_up
                and not pinky_up and thumb_closed):
            return STATE_DRAW, states

        if (index_up and middle_up and not ring_up
                and not pinky_up and thumb_closed):
            return STATE_ERASE, states

        return STATE_IDLE, states

    def _debounce_state(self, raw_state: str) -> str:
        if raw_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = raw_state
            self._pending_count = 1
        if self._pending_count >= self._debounce_frames:
            return raw_state
        return self._state

    # ── ink layer ─────────────────────────────────────────────────
    def _ensure_ink(self, h: int, w: int) -> None:
        if self._ink is None:
            self._ink = np.zeros((h, w, 3), dtype=np.uint8)

    def clear_ink(self) -> None:
        if self._ink is not None:
            self._ink[:] = 0

    # ── drawing helpers ───────────────────────────────────────────
    @staticmethod
    def _to_px(lm, w: int, h: int) -> tuple[int, int]:
        return int(lm.x * w), int(lm.y * h)

    def _draw_stroke(self, p: tuple[int, int]) -> None:
        if self._prev_pos is not None:
            cv2.line(self._ink, self._prev_pos, p,
                     DRAW_COLOR, DRAW_THICKNESS, cv2.LINE_AA)
        else:
            cv2.circle(self._ink, p, DRAW_THICKNESS,
                       DRAW_COLOR, -1, cv2.LINE_AA)
        self._prev_pos = p

    def _erase_stroke(self, p: tuple[int, int]) -> None:
        cv2.circle(self._ink, p, ERASE_RADIUS, (0, 0, 0), -1, cv2.LINE_AA)
        self._prev_pos = None

    def _reset_stroke(self) -> None:
        self._prev_pos = None

    # ── HUD ───────────────────────────────────────────────────────
    def _draw_hud(self, frame: np.ndarray, finger_states: dict,
                  index_px: Optional[tuple[int, int]],
                  middle_px: Optional[tuple[int, int]],
                  h: int, w: int) -> None:

        # Status banner
        if self._state == STATE_DRAW:
            b_color, b_text, t_color = GREEN, "Drawing", GREEN
        elif self._state == STATE_ERASE:
            b_color, b_text, t_color = RED, "Erasing", RED
        else:
            b_color, b_text, t_color = GREY, "Idle", GREY

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 50), b_color, -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, dst=frame)
        cv2.putText(frame, b_text, (w // 2 - 80, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)

        # Fingertip dots
        if index_px is not None:
            cv2.circle(frame, index_px, 12, t_color, -1, cv2.LINE_AA)
            cv2.circle(frame, index_px, 12, WHITE, 2, cv2.LINE_AA)
        if middle_px is not None:
            cv2.circle(frame, middle_px, 12, t_color, -1, cv2.LINE_AA)
            cv2.circle(frame, middle_px, 12, WHITE, 2, cv2.LINE_AA)

        # Finger-state panel (top-right)
        px_, py_ = w - 135, 58
        cv2.rectangle(frame, (px_ - 4, py_ - 4),
                      (px_ + 128, py_ + 120), (20, 20, 20), -1)
        for i, name in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
            fs = finger_states.get(name, "?")
            dc = GREEN if fs in ("up", "open") else (ORANGE if fs == "closed" else RED)
            y = py_ + 14 + i * 22
            cv2.circle(frame, (px_ + 10, y), 5, dc, -1, cv2.LINE_AA)
            cv2.putText(frame, f"{name}: {fs}", (px_ + 22, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)

        # Help
        cv2.putText(frame, "Q = quit  |  C = clear",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, GREY, 1, cv2.LINE_AA)

    # ── update ────────────────────────────────────────────────────
    def update(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._ensure_ink(h, w)

        # Dim the webcam
        dimmed = cv2.addWeighted(frame, WEBCAM_DIM, np.zeros_like(frame), 0, 0)

        # Composite ink onto dimmed feed
        ink_mask = self._ink.astype(bool)
        dimmed[ink_mask] = self._ink[ink_mask]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, self._frame_idx)
        self._frame_idx += 1

        index_px = None
        middle_px = None
        finger_states: dict = {}
        prev_state = self._state

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            lms = result.hand_landmarks[0]
            index_px = self._to_px(lms[INDEX_TIP], w, h)
            middle_px = self._to_px(lms[MIDDLE_TIP], w, h)

            raw_state, finger_states = self._classify_gesture(lms)
            self._state = self._debounce_state(raw_state)

            if self._state != prev_state:
                self._reset_stroke()

            if self._state == STATE_DRAW:
                self._draw_stroke(index_px)
            elif self._state == STATE_ERASE:
                ep = ((index_px[0] + middle_px[0]) // 2,
                      (index_px[1] + middle_px[1]) // 2)
                self._erase_stroke(ep)
            else:
                self._reset_stroke()
        else:
            self._state = STATE_IDLE
            self._reset_stroke()

        self._draw_hud(dimmed, finger_states, index_px, middle_px, h, w)
        return dimmed


# ===================================================================
def main() -> None:
    print(f"Model: {MODEL_PATH}")
    print("=" * 45)
    print("Virtual Blackboard -- MediaPipe Hands")
    print("  Index finger only  -> DRAW")
    print("  Index + middle      -> ERASE")
    print("  Any other gesture   -> IDLE")
    print("  q = quit   |   c = clear")
    print("=" * 45)

    with VirtualBlackboard() as board, \
         WebcamManager(camera_id=0, width=1280, height=720) as cam:

        while True:
            success, frame = cam.read()
            if not success:
                break

            output = board.update(frame)
            cv2.imshow("Virtual Blackboard", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                board.clear_ink()

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

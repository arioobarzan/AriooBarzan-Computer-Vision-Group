"""
Virtual Blackboard with MediaPipe Hands
=========================================
Draw on a digital canvas using hand gestures in the air.

Gestures (detected via robust finger-extension logic with margin):
  - **Index finger only**  -> DRAW  (white line on black canvas)
  - **Index + middle**     -> ERASE (thick black circle wipes content)
  - **Anything else**      -> IDLE  (no drawing, resets stroke)

Two windows open:
  1. Webcam feed with gesture feedback + finger-state debug panel
  2. Full-screen black canvas where drawing appears

Press 'q' to quit, 'c' to clear the canvas.
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
# Tips
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# MCP joints (base knuckles)
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

# PIP / IP joints
THUMB_IP = 3
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

# ---------------------------------------------------------------------------
# Finger-detection parameters (normalised, 0..1)
# ---------------------------------------------------------------------------
FINGER_UP_Y_MARGIN = 0.06   # tip must be at least 6% of frame height above PIP
THUMB_CLOSED_DIST = 0.12    # thumb tip near index MCP -> thumb is folded

# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------
DRAW_COLOR = (255, 255, 255)   # white ink
ERASE_COLOR = (0, 0, 0)        # black (canvas background)
DRAW_THICKNESS = 5
ERASE_RADIUS = 35

# Feedback colours (BGR)
GREEN = (0, 255, 80)
RED = (0, 50, 255)
ORANGE = (0, 165, 255)
GREY = (180, 180, 180)
WHITE = (255, 255, 255)

# ---------------------------------------------------------------------------
# Gesture state machine
# ---------------------------------------------------------------------------
STATE_DRAW = "DRAW"
STATE_ERASE = "ERASE"
STATE_IDLE = "IDLE"


# ===================================================================
# VirtualBlackboard
# ===================================================================
class VirtualBlackboard:
    """
    Gesture-driven drawing canvas.

    - Detects which fingers are extended using tip-vs-PIP with margin.
    - Thumb uses proximity to index MCP (folded thumb = near index base).
    - State machine with 2-frame debounce prevents flicker.
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

        # Canvas (created on first frame)
        self._canvas: Optional[np.ndarray] = None
        self._canvas_h = 0
        self._canvas_w = 0

        # Drawing state
        self._prev_pos: Optional[tuple[int, int]] = None
        self._state = STATE_IDLE
        self._frame_idx = 0

        # 2-frame debounce: state must be same for N frames to take effect
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

    # ── finger-extension logic (robust, with margin) ──────────────
    @staticmethod
    def _is_finger_up(tip_lm, pip_lm) -> bool:
        """
        A finger is extended when its tip is clearly above the PIP joint.
        We require a margin (FINGER_UP_Y_MARGIN) to prevent flicker
        when the finger is partially bent.
        """
        return tip_lm.y < pip_lm.y - FINGER_UP_Y_MARGIN

    @staticmethod
    def _is_finger_down(tip_lm, pip_lm) -> bool:
        """
        A finger is folded/closed when its tip is below or at the
        same level as its PIP joint.
        """
        return tip_lm.y >= pip_lm.y

    @staticmethod
    def _is_thumb_closed(tip_lm, index_mcp_lm) -> bool:
        """
        Thumb is folded (closed) when its tip sits close to the
        index-finger MCP joint (the base of the index finger).
        Uses a normalised Euclidean distance threshold.
        """
        dx = tip_lm.x - index_mcp_lm.x
        dy = tip_lm.y - index_mcp_lm.y
        return (dx * dx + dy * dy) < THUMB_CLOSED_DIST ** 2

    def _classify_gesture(self, landmarks) -> tuple[str, dict]:
        """
        Determine gesture from finger-extension state.

        Returns (state, finger_states_dict) where finger_states
        is useful for debug overlay.
        """
        lms = landmarks
        thumb_closed = self._is_thumb_closed(lms[THUMB_TIP], lms[INDEX_MCP])
        index_up = self._is_finger_up(lms[INDEX_TIP], lms[INDEX_PIP])
        middle_up = self._is_finger_up(lms[MIDDLE_TIP], lms[MIDDLE_PIP])
        ring_up = self._is_finger_up(lms[RING_TIP], lms[RING_PIP])
        pinky_up = self._is_finger_up(lms[PINKY_TIP], lms[PINKY_PIP])

        finger_states = {
            "thumb": "closed" if thumb_closed else "open",
            "index": "up" if index_up else "down",
            "middle": "up" if middle_up else "down",
            "ring": "up" if ring_up else "down",
            "pinky": "up" if pinky_up else "down",
        }

        # DRAW: index only, all others down/folded
        if (index_up and not middle_up and not ring_up
                and not pinky_up and thumb_closed):
            return STATE_DRAW, finger_states

        # ERASE: index + middle, ring/pinky down, thumb closed
        if (index_up and middle_up and not ring_up
                and not pinky_up and thumb_closed):
            return STATE_ERASE, finger_states

        return STATE_IDLE, finger_states

    # ── debounced state transition ─────────────────────────────────
    def _debounce_state(self, raw_state: str) -> str:
        """Require *raw_state* to be seen N consecutive frames before
           actually switching from the current ``_state``."""
        if raw_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = raw_state
            self._pending_count = 1

        if self._pending_count >= self._debounce_frames:
            return raw_state
        return self._state

    # ── canvas management ─────────────────────────────────────────
    def _ensure_canvas(self, h: int, w: int) -> None:
        """Create the blackboard if not yet initialised."""
        if self._canvas is None:
            self._canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self._canvas_h, self._canvas_w = h, w

    def clear_canvas(self) -> None:
        """Reset the canvas to pure black."""
        if self._canvas is not None:
            self._canvas[:] = 0

    # ── landmark helpers ──────────────────────────────────────────
    @staticmethod
    def _to_px(lm, w: int, h: int) -> tuple[int, int]:
        return int(lm.x * w), int(lm.y * h)

    # ── drawing on canvas ─────────────────────────────────────────
    def _draw_stroke(self, p: tuple[int, int]) -> None:
        """Draw a white line (or dot) from the previous point to *p*."""
        if self._prev_pos is not None:
            cv2.line(self._canvas, self._prev_pos, p,
                     DRAW_COLOR, DRAW_THICKNESS, cv2.LINE_AA)
        else:
            # First frame in DRAW: draw a dot so user sees instant feedback
            cv2.circle(self._canvas, p, DRAW_THICKNESS,
                       DRAW_COLOR, -1, cv2.LINE_AA)
        self._prev_pos = p

    def _erase_stroke(self, p: tuple[int, int]) -> None:
        """Draw a thick black circle at *p* to wipe white ink."""
        cv2.circle(self._canvas, p, ERASE_RADIUS, ERASE_COLOR, -1, cv2.LINE_AA)
        self._prev_pos = None   # eraser does not connect strokes

    def _reset_stroke(self) -> None:
        """Break the drawing line so next DRAW starts fresh."""
        self._prev_pos = None

    # ── webcam overlay ────────────────────────────────────────────
    def _draw_overlay(self, frame: np.ndarray, lms,
                      finger_states: dict,
                      index_tip_px: tuple[int, int],
                      middle_tip_px: Optional[tuple[int, int]],
                      h: int, w: int) -> None:
        """Render status banner, fingertip indicators, and debug panel."""

        # ── Status banner (top) ──
        if self._state == STATE_DRAW:
            banner_color, banner_text = GREEN, "Status: Drawing"
            tip_color = GREEN
        elif self._state == STATE_ERASE:
            banner_color, banner_text = RED, "Status: Erasing"
            tip_color = RED
        else:
            banner_color, banner_text = GREY, "Status: Idle"
            tip_color = GREY

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), banner_color, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)
        cv2.putText(frame, banner_text, (w // 2 - 150, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)

        # ── Active fingertip circles ──
        cv2.circle(frame, index_tip_px, 14, tip_color, -1, cv2.LINE_AA)
        cv2.circle(frame, index_tip_px, 14, WHITE, 2, cv2.LINE_AA)

        if middle_tip_px is not None:
            cv2.circle(frame, middle_tip_px, 14, tip_color, -1, cv2.LINE_AA)
            cv2.circle(frame, middle_tip_px, 14, WHITE, 2, cv2.LINE_AA)

        # ── Debug panel: finger states (top-right) ──
        panel_w = 150
        panel_h = 160
        panel_x = w - panel_w - 10
        panel_y = 70

        cv2.rectangle(frame,
                      (panel_x - 2, panel_y - 2),
                      (panel_x + panel_w + 2, panel_y + panel_h + 2),
                      (20, 20, 20), -1)
        cv2.rectangle(frame,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (80, 80, 80), 1)

        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        for i, name in enumerate(finger_names):
            fstate = finger_states.get(name, "?")
            if fstate in ("up", "open"):
                dot_color = GREEN
            elif fstate == "closed":
                dot_color = ORANGE
            else:
                dot_color = RED

            y = panel_y + 18 + i * 26
            cv2.circle(frame, (panel_x + 14, y), 6, dot_color, -1, cv2.LINE_AA)
            cv2.putText(frame, f"{name}: {fstate}",
                        (panel_x + 28, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

        # ── Help text (bottom) ──
        cv2.putText(frame, "Q = quit  |  C = clear canvas",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, GREY, 1, cv2.LINE_AA)

    # ── update loop ───────────────────────────────────────────────
    def update(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Process one frame: detect hand, classify gesture, draw/erase
        on canvas, render overlays.

        Returns:
            (annotated_webcam_frame, canvas_frame)
        """
        h, w = frame.shape[:2]
        self._ensure_canvas(h, w)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, self._frame_idx)
        self._frame_idx += 1

        index_tip_px: Optional[tuple[int, int]] = None
        middle_tip_px: Optional[tuple[int, int]] = None
        lms = None
        finger_states: dict = {}
        prev_state = self._state

        # ── Hand detected ──
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            lms = result.hand_landmarks[0]

            index_tip_px = self._to_px(lms[INDEX_TIP], w, h)
            middle_tip_px = self._to_px(lms[MIDDLE_TIP], w, h)

            raw_state, finger_states = self._classify_gesture(lms)
            self._state = self._debounce_state(raw_state)

            # State change -> break stroke continuity
            if self._state != prev_state:
                self._reset_stroke()

            # Execute action
            if self._state == STATE_DRAW:
                self._draw_stroke(index_tip_px)
            elif self._state == STATE_ERASE:
                # Eraser uses midpoint between index & middle tips
                eraser_pos = ((index_tip_px[0] + middle_tip_px[0]) // 2,
                              (index_tip_px[1] + middle_tip_px[1]) // 2)
                self._erase_stroke(eraser_pos)
            else:
                self._reset_stroke()
        else:
            self._state = STATE_IDLE
            self._reset_stroke()

        # ── Overlays ──
        if index_tip_px is not None:
            self._draw_overlay(frame, lms, finger_states,
                               index_tip_px, middle_tip_px, h, w)
        else:
            # No hand visible — show idle banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), GREY, -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)
            cv2.putText(frame, "Status: Idle (no hand)", (w // 2 - 180, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
            cv2.putText(frame, "Q = quit  |  C = clear canvas",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, GREY, 1, cv2.LINE_AA)

        return frame, self._canvas


# ===================================================================
# main
# ===================================================================
def main() -> None:
    """Run the virtual blackboard loop."""
    print(f"Model: {MODEL_PATH}")
    print("=" * 50)
    print("Virtual Blackboard -- MediaPipe Hands")
    print("  Index finger only  -> DRAW  (white ink)")
    print("  Index + middle      -> ERASE (thick wiper)")
    print("  Any other gesture   -> IDLE  (no marks)")
    print("  q = quit   |   c = clear canvas")
    print("=" * 50)

    with VirtualBlackboard() as board, \
         WebcamManager(camera_id=0, width=1280, height=720) as cam:

        while True:
            success, frame = cam.read()
            if not success:
                break

            webcam_frame, canvas_frame = board.update(frame)

            # Show both windows
            cv2.imshow("Virtual Blackboard - Webcam", webcam_frame)
            cv2.imshow("Virtual Blackboard - Canvas", canvas_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                board.clear_canvas()
                print("[OK] Canvas cleared.")

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

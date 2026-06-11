"""
Rock-Paper-Scissors AI with MediaPipe Hands
=============================================
Play Rock-Paper-Scissors against the computer using your webcam.

- Show your hand gesture to the camera.
- A 3-second countdown runs automatically.
- At "SHOOT!" your gesture is captured, the AI picks randomly,
  and the winner is announced.

Gestures (rule-based finger-extension logic):
  Rock     — all 5 fingers folded
  Paper    — all 5 fingers extended
  Scissors — only index + middle extended, others folded

Press 'q' to quit.
"""

import os
import random
import sys
import time
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

THUMB_IP = 3
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

INDEX_MCP = 5

# ---------------------------------------------------------------------------
# Finger-detection thresholds
# ---------------------------------------------------------------------------
FINGER_UP_Y_MARGIN = 0.06    # tip must be at least 6 % of frame height above PIP
THUMB_CLOSED_DIST = 0.12     # thumb tip near index MCP -> thumb is folded

# ---------------------------------------------------------------------------
# Game timing (seconds)
# ---------------------------------------------------------------------------
COUNTDOWN_SECS = 3.0
RESULT_SECS = 2.5

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
GREEN = (0, 255, 80)
RED = (0, 50, 255)
BLUE = (255, 130, 0)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
GREY = (180, 180, 180)
DARK = (30, 30, 30)

GESTURES = ["Rock", "Paper", "Scissors"]
RULES = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}


# ===================================================================
# RockPaperScissorsAI
# ===================================================================
class RockPaperScissorsAI:
    """
    Real-time Rock-Paper-Scissors game.

    - Classifies hand gesture from finger-extension state.
    - Auto countdown → capture at "SHOOT!" → AI picks → result.
    - Persistent scoreboard.
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

        # Scoreboard
        self.user_score = 0
        self.ai_score = 0

        # Round state
        self._round_start = time.time()
        self._phase = "countdown"          # countdown | shoot | result
        self._phase_start = time.time()
        self._user_gesture: str = "?"
        self._ai_choice: str = "?"
        self._result: str = ""
        self._captured = False

        self._frame_idx = 0
        self._current_gesture: str = "?"

    # ── context manager ───────────────────────────────────────────
    def __enter__(self) -> "RockPaperScissorsAI":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "_detector"):
            self._detector.close()

    # ── finger detection ──────────────────────────────────────────
    @staticmethod
    def _is_finger_up(tip, pip) -> bool:
        """Finger is extended when tip is clearly above PIP joint."""
        return tip.y < pip.y - FINGER_UP_Y_MARGIN

    @staticmethod
    def _is_finger_down(tip, pip) -> bool:
        """Finger is folded when tip is at or below PIP joint."""
        return tip.y >= pip.y

    @staticmethod
    def _is_thumb_closed(tip, index_mcp) -> bool:
        """Thumb is folded when tip sits near index-finger MCP."""
        dx = tip.x - index_mcp.x
        dy = tip.y - index_mcp.y
        return (dx * dx + dy * dy) < THUMB_CLOSED_DIST ** 2

    # ── gesture classification ────────────────────────────────────
    def _classify(self, landmarks) -> str:
        lms = landmarks
        thumb_closed = self._is_thumb_closed(lms[THUMB_TIP], lms[INDEX_MCP])
        index_up = self._is_finger_up(lms[INDEX_TIP], lms[INDEX_PIP])
        middle_up = self._is_finger_up(lms[MIDDLE_TIP], lms[MIDDLE_PIP])
        ring_up = self._is_finger_up(lms[RING_TIP], lms[RING_PIP])
        pinky_up = self._is_finger_up(lms[PINKY_TIP], lms[PINKY_PIP])

        # All 4 main fingers extended + thumb out = Paper
        if index_up and middle_up and ring_up and pinky_up and not thumb_closed:
            return "Paper"

        # Only index + middle extended, others folded = Scissors
        if index_up and middle_up and not ring_up and not pinky_up and thumb_closed:
            return "Scissors"

        # All fingers folded = Rock
        if (not index_up and not middle_up and not ring_up
                and not pinky_up and thumb_closed):
            return "Rock"

        return "?"

    # ── game logic ────────────────────────────────────────────────
    def _determine_winner(self, user: str, ai: str) -> str:
        if user == ai:
            return "TIE!"
        if RULES.get(user) == ai:
            return "YOU WIN!"
        return "AI WINS!"

    # ── HUD ───────────────────────────────────────────────────────
    def _draw_hud(self, frame: np.ndarray, h: int, w: int) -> None:
        """Render scoreboard, countdown / result, and gesture indicators."""

        # -- Top banner (scoreboard) --
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 80), DARK, -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, dst=frame)

        # User score (left)
        cv2.putText(frame, "YOU", (30, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)
        cv2.putText(frame, str(self.user_score), (48, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, GREEN, 3, cv2.LINE_AA)

        # AI score (right)
        cv2.putText(frame, "AI", (w - 120, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
        cv2.putText(frame, str(self.ai_score), (w - 105, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, RED, 3, cv2.LINE_AA)

        # Current gesture (centre of banner)
        g_color = GREEN if self._current_gesture != "?" else GREY
        cv2.putText(frame, f"Gesture: {self._current_gesture}",
                    (w // 2 - 110, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, g_color, 2, cv2.LINE_AA)

        # -- Countdown / Result (centre of frame) --
        elapsed = time.time() - self._phase_start

        if self._phase == "countdown":
            remaining = max(0, COUNTDOWN_SECS - elapsed)
            num = int(remaining) + 1
            if num > 0:
                count_text = str(num)
                cv2.putText(frame, count_text,
                            (w // 2 - 60, h // 2 + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 6, WHITE, 8, cv2.LINE_AA)
        elif self._phase == "shoot":
            # "SHOOT!" appears briefly
            cv2.putText(frame, "SHOOT!",
                        (w // 2 - 220, h // 2 + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 5, BLUE, 7, cv2.LINE_AA)
        elif self._phase == "result":
            # Show user vs AI side-by-side
            cv2.putText(frame, f"YOU: {self._user_gesture}",
                        (w // 2 - 380, h // 2 - 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, GREEN, 4, cv2.LINE_AA)
            cv2.putText(frame, f"AI: {self._ai_choice}",
                        (w // 2 + 30, h // 2 - 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, RED, 4, cv2.LINE_AA)

            # Result (large, centre)
            if self._result == "YOU WIN!":
                r_color = GREEN
            elif self._result == "AI WINS!":
                r_color = RED
            else:
                r_color = ORANGE
            cv2.putText(frame, self._result,
                        (w // 2 - 220, h // 2 + 40),
                        cv2.FONT_HERSHEY_DUPLEX, 3, r_color, 6, cv2.LINE_AA)

        # -- Bottom help --
        cv2.putText(frame, "Q = quit  |  Show your hand to the camera",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, GREY, 1, cv2.LINE_AA)

    # ── update ────────────────────────────────────────────────────
    def update(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, self._frame_idx)
        self._frame_idx += 1

        now = time.time()

        # ── Gesture detection ──
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            self._current_gesture = self._classify(result.hand_landmarks[0])
        else:
            self._current_gesture = "?"

        # ── Phase machine ──
        elapsed = now - self._phase_start

        if self._phase == "countdown" and elapsed >= COUNTDOWN_SECS:
            self._phase = "shoot"
            self._phase_start = now
            # Capture gesture + AI pick
            self._user_gesture = (self._current_gesture
                                  if self._current_gesture != "?" else "Rock")
            self._ai_choice = random.choice(GESTURES)
            self._result = self._determine_winner(self._user_gesture,
                                                  self._ai_choice)
            if self._result == "YOU WIN!":
                self.user_score += 1
            elif self._result == "AI WINS!":
                self.ai_score += 1

        elif self._phase == "shoot" and elapsed >= 0.5:
            self._phase = "result"
            self._phase_start = now

        elif self._phase == "result" and elapsed >= RESULT_SECS:
            self._phase = "countdown"
            self._phase_start = now

        # ── HUD ──
        self._draw_hud(frame, h, w)
        return frame


# ===================================================================
def main() -> None:
    print(f"Model: {MODEL_PATH}")
    print("=" * 50)
    print("Rock-Paper-Scissors AI -- MediaPipe Hands")
    print("  Show Rock, Paper, or Scissors to the camera.")
    print("  3-second countdown, then gesture is captured.")
    print("  q = quit")
    print("=" * 50)

    with RockPaperScissorsAI() as game, \
         WebcamManager(camera_id=0, width=1280, height=720) as cam:

        while True:
            success, frame = cam.read()
            if not success:
                break

            frame = game.update(frame)
            cv2.imshow("Rock-Paper-Scissors AI", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"Final score -- You: {game.user_score}  |  AI: {game.ai_score}")
    print("Done.")


if __name__ == "__main__":
    main()

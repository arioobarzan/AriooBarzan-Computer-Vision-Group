"""
Rock-Paper-Scissors AI with MediaPipe Hands
=============================================
Play Rock-Paper-Scissors against the computer using your webcam.

- Show your hand gesture to the camera.
- A 3-second countdown runs automatically.
- At "SHOOT!" your gesture is captured, the AI picks randomly,
  and the winner is announced.

Gestures (geometry-based, using fingertip-to-wrist distance ratios):
  Rock     — all 5 fingertips close to wrist (fingers folded)
  Paper    — all 5 fingertips far from wrist (fingers extended)
  Scissors — index + middle far, ring + pinky close, thumb close

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
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
MIDDLE_MCP = 9      # palm-centre reference for normalisation

# The 6 points we draw on the hand
DISPLAY_POINTS = [
    (WRIST,      "Wrist",  (0, 220, 255)),   # Gold
    (THUMB_TIP,  "Thumb",  (255, 0, 0)),     # Blue
    (INDEX_TIP,  "Index",  (0, 255, 0)),     # Green
    (MIDDLE_TIP, "Middle", (0, 0, 255)),     # Red
    (RING_TIP,   "Ring",   (255, 255, 0)),   # Cyan
    (PINKY_TIP,  "Pinky",  (255, 0, 255)),   # Magenta
]

# ---------------------------------------------------------------------------
# Geometry-based finger-extension thresholds
#
# extension_ratio = distance(fingertip, wrist) / palm_size
#   where palm_size = distance(wrist, middle_mcp)
#
# A finger is "extended" when its tip is far enough from the wrist
# relative to the palm size.  This is scale- and orientation-invariant.
# ---------------------------------------------------------------------------
# Per-finger thresholds account for different finger lengths.
# Pinky & ring are shorter → lower ceiling when fully open.
EXTENDED_RATIO = {
    THUMB_TIP: 1.50,
    INDEX_TIP: 1.55,
    MIDDLE_TIP: 1.55,
    RING_TIP: 1.40,
    PINKY_TIP: 1.35,
}
FOLDED_RATIO = 1.25         # any finger below this is definitely folded

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

    - Classifies hand gesture from fingertip-to-wrist distance ratios.
    - Auto countdown -> capture at "SHOOT!" -> AI picks -> result.
    - Persistent scoreboard with 6-point hand overlay.
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
        self._phase = "countdown"
        self._phase_start = time.time()
        self._user_gesture: str = "?"
        self._ai_choice: str = "?"
        self._result: str = ""

        self._frame_idx = 0
        self._current_gesture: str = "?"

        # Last-hand data for HUD drawing
        self._landmarks = None
        self._frame_h = 0
        self._frame_w = 0
        self._ratios: dict[int, float] = {}   # last computed ratios for display

    # ── context manager ───────────────────────────────────────────
    def __enter__(self) -> "RockPaperScissorsAI":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "_detector"):
            self._detector.close()

    # ── geometry ──────────────────────────────────────────────────
    @staticmethod
    def _dist(lm1, lm2) -> float:
        """Normalised Euclidean distance between two landmarks."""
        return np.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

    def _extension_ratio(self, lms, tip_idx: int) -> float:
        """
        Normalised distance from fingertip to wrist.
        palm_size = |wrist -> middle MCP| serves as the reference.

        Returns a value typically in [0.5 (tight fist) ... 2.2 (fully open)].
        """
        palm_size = self._dist(lms[WRIST], lms[MIDDLE_MCP])
        if palm_size < 0.01:
            return 0.0
        return self._dist(lms[tip_idx], lms[WRIST]) / palm_size

    # ── gesture classification (geometry-based) ───────────────────
    def _classify(self, landmarks) -> str:
        lms = landmarks

        # Compute extension ratio for each finger (fingertip-to-wrist / palm-size)
        self._ratios = {
            THUMB_TIP: self._extension_ratio(lms, THUMB_TIP),
            INDEX_TIP: self._extension_ratio(lms, INDEX_TIP),
            MIDDLE_TIP: self._extension_ratio(lms, MIDDLE_TIP),
            RING_TIP: self._extension_ratio(lms, RING_TIP),
            PINKY_TIP: self._extension_ratio(lms, PINKY_TIP),
        }

        # A finger is "up" if above its per-finger threshold
        up = {k: self._ratios[k] >= EXTENDED_RATIO[k] for k in self._ratios}

        # All 5 extended = Paper
        if up[INDEX_TIP] and up[MIDDLE_TIP] and up[RING_TIP] and up[PINKY_TIP] and up[THUMB_TIP]:
            return "Paper"

        # Only index + middle extended, others folded = Scissors
        if (up[INDEX_TIP] and up[MIDDLE_TIP]
                and not up[RING_TIP] and not up[PINKY_TIP]
                and not up[THUMB_TIP]):
            return "Scissors"

        # All 5 folded = Rock
        if all(self._ratios[k] < FOLDED_RATIO for k in self._ratios):
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
    def _draw_hand_points(self, frame: np.ndarray) -> None:
        """Draw the 6 coloured dots + extension-ratio labels on each fingertip."""
        if self._landmarks is None:
            return
        lms = self._landmarks
        h, w = self._frame_h, self._frame_w

        wrist_xy = (int(lms[WRIST].x * w), int(lms[WRIST].y * h))

        # Palm-size reference line (wrist -> middle MCP)
        mcp_xy = (int(lms[MIDDLE_MCP].x * w), int(lms[MIDDLE_MCP].y * h))
        cv2.line(frame, wrist_xy, mcp_xy, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "palm", (mcp_xy[0] + 6, mcp_xy[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)

        # TIP_FINGER_MAP for ratio labels
        tip_names = {THUMB_TIP: "Th", INDEX_TIP: "In", MIDDLE_TIP: "Mi",
                     RING_TIP: "Ri", PINKY_TIP: "Pi"}

        for idx, _label, color in DISPLAY_POINTS:
            x, y = int(lms[idx].x * w), int(lms[idx].y * h)
            # Outer white ring
            cv2.circle(frame, (x, y), 12, WHITE, 2, cv2.LINE_AA)
            # Filled coloured dot
            cv2.circle(frame, (x, y), 9, color, -1, cv2.LINE_AA)

            # Show extension ratio next to each fingertip
            if idx in self._ratios:
                r = self._ratios[idx]
                thresh = EXTENDED_RATIO.get(idx, 1.55)
                up = r >= thresh
                r_color = GREEN if up else RED
                r_text = f"{r:.2f}"
                cv2.putText(frame, r_text, (x + 16, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, r_color, 1, cv2.LINE_AA)

        # Connect fingertips to wrist with thin guide lines
        for tip_idx in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
            tip_xy = (int(lms[tip_idx].x * w), int(lms[tip_idx].y * h))
            cv2.line(frame, wrist_xy, tip_xy, (100, 100, 100), 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, h: int, w: int) -> None:
        """Scoreboard, countdown / result, hand points."""

        # -- Top banner (scoreboard) --
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 80), DARK, -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, dst=frame)

        cv2.putText(frame, "YOU", (30, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)
        cv2.putText(frame, str(self.user_score), (48, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, GREEN, 3, cv2.LINE_AA)

        cv2.putText(frame, "AI", (w - 120, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
        cv2.putText(frame, str(self.ai_score), (w - 105, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, RED, 3, cv2.LINE_AA)

        g_color = GREEN if self._current_gesture != "?" else GREY
        cv2.putText(frame, f"Gesture: {self._current_gesture}",
                    (w // 2 - 110, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, g_color, 2, cv2.LINE_AA)

        # -- Countdown / Result (centre) --
        elapsed = time.time() - self._phase_start

        if self._phase == "countdown":
            remaining = max(0, COUNTDOWN_SECS - elapsed)
            num = int(remaining) + 1
            if num > 0:
                cv2.putText(frame, str(num),
                            (w // 2 - 60, h // 2 + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 6, WHITE, 8, cv2.LINE_AA)
        elif self._phase == "shoot":
            cv2.putText(frame, "SHOOT!",
                        (w // 2 - 220, h // 2 + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 5, BLUE, 7, cv2.LINE_AA)
        elif self._phase == "result":
            cv2.putText(frame, f"YOU: {self._user_gesture}",
                        (w // 2 - 380, h // 2 - 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, GREEN, 4, cv2.LINE_AA)
            cv2.putText(frame, f"AI: {self._ai_choice}",
                        (w // 2 + 30, h // 2 - 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, RED, 4, cv2.LINE_AA)

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
        self._frame_h, self._frame_w = h, w

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, self._frame_idx)
        self._frame_idx += 1

        now = time.time()

        # ── Gesture detection ──
        self._landmarks = None
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            self._landmarks = result.hand_landmarks[0]
            self._current_gesture = self._classify(self._landmarks)
        else:
            self._current_gesture = "?"

        # ── Phase machine ──
        elapsed = now - self._phase_start

        if self._phase == "countdown" and elapsed >= COUNTDOWN_SECS:
            self._phase = "shoot"
            self._phase_start = now
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

        # ── Draw hand points + HUD ──
        self._draw_hand_points(frame)
        self._draw_hud(frame, h, w)
        return frame


# ===================================================================
def main() -> None:
    print(f"Model: {MODEL_PATH}")
    print("=" * 50)
    print("Rock-Paper-Scissors AI -- MediaPipe Hands")
    print("  Show Rock, Paper, or Scissors to the camera.")
    print("  3-second countdown, then gesture is captured.")
    print("  6 coloured dots show tracked points on your hand.")
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

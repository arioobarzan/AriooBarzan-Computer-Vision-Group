"""
PES 2013 AI Penalty Kicker with MediaPipe Hands
==================================================
Control penalty shootouts in PES 2013 using hand gestures via webcam.

- Move your HAND to LEFT / CENTER / RIGHT zone -> aims in that direction.
  The key is HELD DOWN as long as your hand stays in the zone.
- Close your hand into a FIST -> hold 'A' (power bar fills).
- Open palm (all 5 fingers) -> release 'A' (lock power & SHOOT).

Uses **pydirectinput** for hardware-level keyboard simulation via
DirectInput scan codes — reliable for games like PES 2013.

Press 'q' to quit.
"""

import os
import sys
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pydirectinput

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
# MediaPipe setup
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
MIDDLE_MCP = 9

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# ---------------------------------------------------------------------------
# Finger-extension thresholds
# ---------------------------------------------------------------------------
EXTENDED_RATIO = {
    THUMB_TIP: 1.50,
    INDEX_TIP: 1.55,
    MIDDLE_TIP: 1.55,
    RING_TIP: 1.40,
    PINKY_TIP: 1.35,
}

# ---------------------------------------------------------------------------
# Aiming zones  (fraction of frame width)
# ---------------------------------------------------------------------------
ZONE_LEFT_MAX = 0.35
ZONE_RIGHT_MIN = 0.65

# key names for pydirectinput
ZONE_KEY = {"LEFT": "left", "CENTER": "up", "RIGHT": "right"}

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
GREEN = (0, 255, 80)
RED = (0, 50, 255)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
GREY = (180, 180, 180)
DARK = (30, 30, 30)

ZONE_COLORS = {
    "LEFT":   (120, 70, 30),
    "CENTER": (70, 120, 30),
    "RIGHT":  (30, 70, 120),
}


# ===================================================================
class PESPenaltyController:
    """
    Hand-gesture penalty-kick controller.

    - Direction key HELD DOWN while hand is in zone (always, every gesture).
    - Fist (all folded)  -> keyDown('a')  (power bar fill).
    - Open palm           -> keyUp('a')    (lock power, shoot).
    - No hand / out of frame -> release ALL keys.
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

        # key state tracking (prevents key-spam)
        self._dir_held: str = ""          # "" | "LEFT" | "CENTER" | "RIGHT"
        self._a_held: bool = False

        # visual
        self._state: str = "IDLE"
        self._zone: str = ""
        self._is_fist: bool = False
        self._ratios: dict = {}
        self._landmarks = None
        self._frame_h = self._frame_w = 0
        self._frame_idx = 0

    # ── context manager ───────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._release_all()
        if hasattr(self, "_detector"):
            self._detector.close()

    # ── key control (pydirectinput) ───────────────────────────────
    def _release_all(self):
        """Release direction key + 'a', track state."""
        if self._dir_held:
            pydirectinput.keyUp(ZONE_KEY[self._dir_held])
            self._dir_held = ""
        if self._a_held:
            pydirectinput.keyUp("a")
            self._a_held = False

    def _hold_direction(self, zone: str):
        """Hold arrow key for *zone*.  Release previous if changed."""
        if zone == self._dir_held:
            return
        # release old
        if self._dir_held:
            pydirectinput.keyUp(ZONE_KEY[self._dir_held])
        # press new
        pydirectinput.keyDown(ZONE_KEY[zone])
        self._dir_held = zone

    def _set_a(self, down: bool):
        """Press or release 'a' — idempotent."""
        if down == self._a_held:
            return
        if down:
            pydirectinput.keyDown("a")
        else:
            pydirectinput.keyUp("a")
        self._a_held = down

    # ── gesture ───────────────────────────────────────────────────
    @staticmethod
    def _dist(lm1, lm2):
        return np.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

    def _extension_ratio(self, lms, tip_idx):
        palm = self._dist(lms[WRIST], lms[MIDDLE_MCP])
        if palm < 0.01:
            return 0.0
        return self._dist(lms[tip_idx], lms[WRIST]) / palm

    def _classify(self, lms):
        ratios = {t: self._extension_ratio(lms, t) for t in FINGERTIPS}
        up = {t: ratios[t] >= EXTENDED_RATIO[t] for t in ratios}
        all_down = all(not up[t] for t in ratios)
        above_one = sum(1 for r in ratios.values() if r > 1.0)
        open_palm = above_one >= 4
        return ratios, all_down, open_palm

    # ── zone ─────────────────────────────────────────────────────
    @staticmethod
    def _get_zone(x_norm):
        if x_norm <= ZONE_LEFT_MAX:
            return "LEFT"
        if x_norm >= ZONE_RIGHT_MIN:
            return "RIGHT"
        return "CENTER"

    # ── HUD ───────────────────────────────────────────────────────
    def _draw_hud(self, frame):
        h, w = self._frame_h, self._frame_w
        lw = int(w * ZONE_LEFT_MAX)
        rx = int(w * ZONE_RIGHT_MIN)

        # -- zone overlays (active = darker) --
        for zname, x0, x1 in [("LEFT", 0, lw), ("CENTER", lw, rx),
                               ("RIGHT", rx, w)]:
            active = self._zone == zname
            alpha = 0.35 if active else 0.08
            ov = frame.copy()
            cv2.rectangle(ov, (x0, 0), (x1, h), ZONE_COLORS[zname], -1)
            cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, dst=frame)

        cv2.line(frame, (lw, 0), (lw, h), WHITE, 1, cv2.LINE_AA)
        cv2.line(frame, (rx, 0), (rx, h), WHITE, 1, cv2.LINE_AA)
        for z, x0, x1 in [("LEFT", 0, lw), ("CENTER", lw, rx),
                           ("RIGHT", rx, w)]:
            cx = (x0 + x1) // 2
            off = 35 if z == "LEFT" else (20 if z == "CENTER" else 40)
            cv2.putText(frame, z, (cx - off, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        WHITE if self._zone == z else GREY,
                        2 if self._zone == z else 1, cv2.LINE_AA)

        # -- bottom status --
        ph = 85
        ov = frame.copy()
        cv2.rectangle(ov, (0, h - ph), (w, h), DARK, -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, dst=frame)

        if self._is_fist:
            st, sc = "SHOOTING: HOLDING A", ORANGE
            sub = "Fist -> power bar filling..."
        elif self._state == "AIMING":
            st, sc = f"AIMING: {self._zone}", GREEN
            sub = f"Direction held: {self._zone} arrow"
        else:
            st, sc = "STATE: IDLE", GREY
            sub = "Show hand to camera"

        cv2.putText(frame, st, (20, h - ph + 35),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, sc, 2, cv2.LINE_AA)
        cv2.putText(frame, sub, (20, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)

        # -- hand points --
        if self._landmarks is not None:
            lms = self._landmarks
            tip_colors = {
                THUMB_TIP: (255, 0, 0), INDEX_TIP: (0, 255, 0),
                MIDDLE_TIP: (0, 0, 255), RING_TIP: (255, 255, 0),
                PINKY_TIP: (255, 0, 255),
            }
            for t_idx, color in tip_colors.items():
                x, y = int(lms[t_idx].x * w), int(lms[t_idx].y * h)
                cv2.circle(frame, (x, y), 10, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 10, WHITE, 2, cv2.LINE_AA)
            wx, wy = int(lms[WRIST].x * w), int(lms[WRIST].y * h)
            cv2.circle(frame, (wx, wy), 8, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (wx, wy), 8, WHITE, 2, cv2.LINE_AA)

            # ratio debug
            tags = [(THUMB_TIP, "Th"), (INDEX_TIP, "In"), (MIDDLE_TIP, "Mi"),
                    (RING_TIP, "Ri"), (PINKY_TIP, "Pi")]
            for t_idx, tag in tags:
                r = self._ratios.get(t_idx, 0)
                c = GREEN if r >= EXTENDED_RATIO[t_idx] else RED
                cv2.putText(frame, f"{tag}:{r:.2f}",
                            (w - 145, 100 + tags.index((t_idx, tag)) * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)

        cv2.putText(frame, "Q = quit", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)

    # ── update ────────────────────────────────────────────────────
    def update(self, frame):
        h, w = frame.shape[:2]
        self._frame_h, self._frame_w = h, w

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, self._frame_idx)
        self._frame_idx += 1

        self._landmarks = None

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            lms = result.hand_landmarks[0]
            self._landmarks = lms
            self._ratios, all_down, open_palm = self._classify(lms)

            # zone from wrist X (stable)
            self._zone = self._get_zone(lms[WRIST].x)

            # ALWAYS hold direction while hand visible
            self._hold_direction(self._zone)

            # fist -> hold A  |  open palm -> release A
            self._is_fist = all_down
            if all_down:
                self._state = "SHOOT"
                self._set_a(True)
            elif open_palm:
                self._state = "RELEASE"
                self._set_a(False)
            else:
                self._state = "AIMING"
                self._set_a(False)
        else:
            self._state = "IDLE"
            self._zone = ""
            self._is_fist = False
            self._release_all()

        self._draw_hud(frame)
        return frame


# ===================================================================
def main():
    print(f"Model: {MODEL_PATH}")
    print("=" * 55)
    print("PES 2013 AI Penalty Kicker -- MediaPipe Hands + pydirectinput")
    print("  Hand in LEFT / CENTER / RIGHT -> aims that way (key held)")
    print("  Fist (all folded)             -> hold 'A' (power)")
    print("  Open palm                     -> release 'A' (shoot!)")
    print("  Keys released when hand leaves frame.")
    print("  q = quit")
    print("=" * 55)

    with PESPenaltyController() as ctrl, \
         WebcamManager(camera_id=0, width=1280, height=720) as cam:
        while True:
            success, frame = cam.read()
            if not success:
                break
            frame = ctrl.update(frame)
            cv2.imshow("PES 2013 Penalty Kicker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

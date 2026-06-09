"""
Blink Counter with MediaPipe Face Mesh
========================================
Real-time eye-blink detector and counter via webcam.

Uses the **Eye Aspect Ratio (EAR)** formula on MediaPipe Face Mesh
landmarks to measure eye openness. A state machine with frame-level
debounce prevents double-counting on long blinks.

  EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

Press 'q' to quit, 'r' to reset the counter.
"""

import os
import sys
from collections import deque
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# Allow importing from the repo root (Sample Codes/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (
    FACE_MODEL_FILENAME,
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
MODEL_PATH = get_model_path(FACE_MODEL_FILENAME)

# ---------------------------------------------------------------------------
# MediaPipe setup (Tasks API -- Face Landmarker)
# ---------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ---------------------------------------------------------------------------
# Eye landmark indices (MediaPipe Face Mesh, 468-point topology)
#
# Each eye uses 6 points for the EAR formula:
#   p1 -- left corner, p2 -- upper-left, p3 -- upper-right,
#   p4 -- right corner, p5 -- lower-right, p6 -- lower-left
# ---------------------------------------------------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]    # p1 … p6
RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # p1 … p6

# ---------------------------------------------------------------------------
# Blink-detection parameters
# ---------------------------------------------------------------------------
EAR_THRESHOLD = 0.20     # EAR below this → eye considered closed
EAR_CLOSE_FRAMES = 2      # consecutive frames EAR must stay low before closure
EAR_OPEN_FRAMES = 2       # consecutive frames EAR must stay high to confirm re-open

# Smoothing
EAR_SMOOTH_WINDOW = 5     # moving-average window on raw EAR

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
GREEN = (0, 255, 80)
RED = (0, 50, 255)
BLUE = (255, 130, 0)
WHITE = (255, 255, 255)
GREY = (180, 180, 180)
DARK = (30, 30, 30)


# ===================================================================
# BlinkCounter
# ===================================================================
class BlinkCounter:
    """
    Eye-blink detector based on Eye Aspect Ratio (EAR).

    State machine:
      OPEN → (EAR drops) → CLOSING → (N frames low) → CLOSED
      CLOSED → (EAR rises) → OPENING → (N frames high) → OPEN → blink++
    """

    def __init__(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._detector = FaceLandmarker.create_from_options(options)

        # Blink count + state
        self.blink_count: int = 0
        self._ear: float = 1.0
        self._eye_closed: bool = False
        self._close_frames: int = 0
        self._open_frames: int = 0

        # EAR smoothing buffer
        self._ear_buffer: deque[float] = deque(maxlen=EAR_SMOOTH_WINDOW)

        # Tracking
        self._frame_idx: int = 0
        self._landmarks_all = None

    # ── context manager ───────────────────────────────────────────
    def __enter__(self) -> "BlinkCounter":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "_detector"):
            self._detector.close()

    # ── EAR calculation ───────────────────────────────────────────
    @staticmethod
    def _calc_ear(landmarks, indices, w: int, h: int) -> float:
        """
        Eye Aspect Ratio for the eye defined by *indices* (list of 6 ints).

                     p2         p3
                     o-----------o
                     |           |
                     |     eye   |
                     |           |
        p1  o--------o-----------o--------o  p4
                     |           |
                     |           |
                     o-----------o
                     p6         p5

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            pts.append((lm.x * w, lm.y * h))

        p1, p2, p3, p4, p5, p6 = pts

        # Vertical eye-open distances
        v1 = np.hypot(p2[0] - p6[0], p2[1] - p6[1])
        v2 = np.hypot(p3[0] - p5[0], p3[1] - p5[1])

        # Horizontal eye width
        h_dist = np.hypot(p1[0] - p4[0], p1[1] - p4[1])

        if h_dist == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h_dist)

    # ── blink state machine ───────────────────────────────────────
    def _update_blink_state(self, ear: float) -> bool:
        """
        Process one EAR reading through the blink state machine.

        Returns True on the exact frame a new blink is registered.
        """
        blinked = False

        if ear < EAR_THRESHOLD:
            # Eye is closing / closed
            if not self._eye_closed:
                self._close_frames += 1
                if self._close_frames >= EAR_CLOSE_FRAMES:
                    self._eye_closed = True
                    self._open_frames = 0
            else:
                self._close_frames = EAR_CLOSE_FRAMES  # keep saturated
        else:
            # Eye is open
            if self._eye_closed:
                self._open_frames += 1
                if self._open_frames >= EAR_OPEN_FRAMES:
                    self._eye_closed = False
                    self._close_frames = 0
                    blinked = True
                    self.blink_count += 1
            else:
                self._close_frames = 0

        return blinked

    # ── smooth EAR ────────────────────────────────────────────────
    def _smooth_ear(self, raw: float) -> float:
        self._ear_buffer.append(raw)
        return sum(self._ear_buffer) / len(self._ear_buffer)

    # ── drawing ───────────────────────────────────────────────────
    def _draw_hud(self, frame: np.ndarray, blinked: bool,
                  h: int, w: int) -> None:
        """Render counter, EAR bar, and eye-landmark dots."""

        # -- Top banner --
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 70), DARK, -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, dst=frame)

        # Blink counter
        cv2.putText(frame, "Blinks", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREY, 2, cv2.LINE_AA)
        cv2.putText(frame, str(self.blink_count), (30, 68),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, GREEN, 4, cv2.LINE_AA)

        # EAR value
        ear_color = RED if self._eye_closed else GREEN
        cv2.putText(frame, f"EAR: {self._ear:.3f}", (w // 2 - 80, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, ear_color, 2, cv2.LINE_AA)

        # Blink flash
        if blinked:
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (w, h), GREEN, -1)
            cv2.addWeighted(flash, 0.15, frame, 0.85, 0, dst=frame)
            cv2.putText(frame, "BLINK!", (w // 2 - 80, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 2.5, GREEN, 4, cv2.LINE_AA)

        # -- EAR progress bar (bottom-centre) --
        bar_w, bar_h = 300, 18
        bar_x, bar_y = w // 2 - bar_w // 2, h - 50

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), DARK, -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)

        # Filled portion (EAR of 0.3 → full bar)
        fill = min(1.0, self._ear / 0.30)
        fill_w = int(bar_w * fill)
        bar_color = RED if self._ear < EAR_THRESHOLD else GREEN
        cv2.rectangle(frame, (bar_x, bar_y + 2),
                      (bar_x + fill_w, bar_y + bar_h - 2), bar_color, -1)

        # Threshold marker line
        th_x = bar_x + int(bar_w * EAR_THRESHOLD / 0.30)
        cv2.line(frame, (th_x, bar_y - 6), (th_x, bar_y + bar_h + 6),
                 (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "thresh", (th_x - 35, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, "Open <--- EAR ---> Closed",
                    (bar_x + 40, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)

        # -- Eye-landmark dots --
        for idx in LEFT_EYE + RIGHT_EYE:
            lm = self._landmarks_all[idx] if self._landmarks_all else None
            if lm:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, BLUE, -1, cv2.LINE_AA)

        # -- Quit/reset hints --
        cv2.putText(frame, "Q = quit  |  R = reset",
                    (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, GREY, 1, cv2.LINE_AA)

    # ── update ────────────────────────────────────────────────────
    def update(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, self._frame_idx)
        self._frame_idx += 1

        blinked = False
        self._landmarks_all = None

        if result.face_landmarks and len(result.face_landmarks) > 0:
            lms = result.face_landmarks[0]
            self._landmarks_all = lms

            left_ear = self._calc_ear(lms, LEFT_EYE, w, h)
            right_ear = self._calc_ear(lms, RIGHT_EYE, w, h)
            raw_ear = (left_ear + right_ear) / 2.0

            self._ear = self._smooth_ear(raw_ear)
            blinked = self._update_blink_state(self._ear)
        else:
            self._ear = 1.0

        self._draw_hud(frame, blinked, h, w)
        return frame


# ===================================================================
def main() -> None:
    print(f"Model: {MODEL_PATH}")
    print("=" * 45)
    print("Blink Counter -- MediaPipe Face Mesh + EAR")
    print("  q = quit   |   r = reset counter")
    print("=" * 45)

    with BlinkCounter() as counter, \
         WebcamManager(camera_id=0, width=1280, height=720) as cam:

        while True:
            success, frame = cam.read()
            if not success:
                break

            frame = counter.update(frame)
            cv2.imshow("Blink Counter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                counter.blink_count = 0

    cv2.destroyAllWindows()
    print(f"Total blinks: {counter.blink_count}")
    print("Done.")


if __name__ == "__main__":
    main()

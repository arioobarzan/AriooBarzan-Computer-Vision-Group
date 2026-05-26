"""
Fingertip Detection with MediaPipe
===================================
Detects 5 fingertips per hand using MediaPipe Hand Landmarker
and draws colored circles on each fingertip.
When both hands are detected, matching fingertips are connected
with gradient lines (thick & bright in the middle, thin & saturated at the ends).

Landmark indices for fingertips:
    4  = Thumb tip
    8  = Index finger tip
    12 = Middle finger tip
    16 = Ring finger tip
    20 = Pinky tip

Requires: hand_landmarker.task in the same directory.
Download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""

import os
import sys

import cv2
import mediapipe as mp
import numpy as np

# Allow importing from the repo root (common/ package)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (
    WebcamManager,
    clear_mediapipe_cache,
    draw_gradient_line,
    get_model_path,
    suppress_gpu_warnings,
)

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
clear_mediapipe_cache()
suppress_gpu_warnings()

MODEL_PATH = get_model_path()
print(f"Model: {MODEL_PATH}")
print("Ready.\n")

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ---------------------------------------------------------------------------
# Fingertip landmark indices and their colors (BGR)
# ---------------------------------------------------------------------------
FINGERTIPS = {
    4:  (255, 0, 0),     # Blue   — Thumb
    8:  (0, 255, 0),     # Green  — Index
    12: (0, 0, 255),     # Red    — Middle
    16: (255, 255, 0),   # Cyan   — Ring
    20: (255, 0, 255),   # Magenta — Pinky
}


def draw_fingertips(frame: np.ndarray, result) -> None:
    """Draw colored circles on fingertips + gradient lines between matching fingers."""
    if not result.hand_landmarks:
        return

    h, w = frame.shape[:2]

    # Collect fingertip coordinates for each detected hand
    all_hands_tips = []

    for hand_landmarks in result.hand_landmarks:
        tips = {}
        for tip_idx in FINGERTIPS:
            lm = hand_landmarks[tip_idx]
            x, y = int(lm.x * w), int(lm.y * h)
            tips[tip_idx] = (x, y)
        all_hands_tips.append(tips)

    # Connect matching fingertips between two hands with gradient lines
    if len(all_hands_tips) == 2:
        for tip_idx, color in FINGERTIPS.items():
            p1 = all_hands_tips[0][tip_idx]
            p2 = all_hands_tips[1][tip_idx]
            draw_gradient_line(frame, p1, p2, color, max_thickness=14, min_thickness=3)

    # Draw fingertip circles
    for tips in all_hands_tips:
        for tip_idx, (x, y) in tips.items():
            color = FINGERTIPS[tip_idx]
            cv2.circle(frame, (x, y), 16, (255, 255, 255), 1)      # outer glow ring
            cv2.circle(frame, (x, y), 13, color, -1)                # solid colored circle
            cv2.circle(frame, (x, y), 13, (255, 255, 255), 2)       # white border


def main() -> None:
    """Run the fingertip detection loop."""
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    with WebcamManager(camera_id=0, width=1280, height=720) as cam:
        print("=" * 50)
        print("Fingertip Detection — MediaPipe")
        print("Colored circles on fingertips + gradient lines")
        print("Press 'q' to quit.")
        print("=" * 50)

        frame_idx = 0
        with HandLandmarker.create_from_options(options) as detector:
            while True:
                success, frame = cam.read()
                if not success:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(mp_image, frame_idx)
                frame_idx += 1

                draw_fingertips(frame, result)

                cv2.imshow("Fingertips - MediaPipe", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

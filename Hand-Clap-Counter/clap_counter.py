"""
Hand Clap Counter with MediaPipe
=================================
Real-time hand-clap counter via webcam.
Tracks 6 corresponding point pairs (5 fingertips + palm center) between
two hands. A clap is registered when at least 4 of the 6 pairs are closer
than the threshold — meaning both palms and fingers meet.

Press 'q' to quit.
"""

import os
import sys

import cv2
import mediapipe as mp

# Allow importing from the repo root (common/ package)
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
# Clap-detection parameters
# ---------------------------------------------------------------------------
# 6 corresponding point pairs: 5 fingertips + palm center (middle MCP)
PAIR_LANDMARKS = [4, 8, 12, 16, 20, 9]   # thumb, index, middle, ring, pinky tips + palm

PAIR_DISTANCE_THRESHOLD = 0.15  # normalised distance — a pair closer than this counts
MIN_PAIRS_CLOSE = 4             # at least this many pairs must be close to register a clap
COOLDOWN_FRAMES = 25            # min frames between consecutive claps
SEPARATION_FRAMES = 8           # hands must be apart this many frames before next clap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _count_close_pairs(lms1, lms2) -> int:
    """Return how many of the PAIR_LANDMARKS point-pairs are closer than threshold."""
    close = 0
    for idx in PAIR_LANDMARKS:
        dx = lms1[idx].x - lms2[idx].x
        dy = lms1[idx].y - lms2[idx].y
        if (dx * dx + dy * dy) ** 0.5 < PAIR_DISTANCE_THRESHOLD:
            close += 1
    return close


def _draw_hud(frame, clap_count: int, clap_this_frame: bool,
              num_hands: int, cooldown_counter: int) -> None:
    """Render counter, flash effect, hand status, and cooldown bar."""
    h, w = frame.shape[:2]

    # Dark transparent banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    # Label
    cv2.putText(frame, "Claps", (w // 2 - 62, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    # Counter
    cv2.putText(frame, str(clap_count), (w // 2 - 40, 95),
                cv2.FONT_HERSHEY_DUPLEX, 3.5, (0, 255, 100), 5, cv2.LINE_AA)

    # Green flash
    if clap_this_frame:
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), -1)
        cv2.addWeighted(flash, 0.2, frame, 0.8, 0, dst=frame)
        cv2.putText(frame, "CLAP!", (w // 2 - 100, h // 2 + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 0), 4, cv2.LINE_AA)

    # Hand status
    status_color = (0, 255, 0) if num_hands == 2 else (0, 165, 255)
    cv2.putText(frame, f"Hands: {num_hands}/2", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    # Quit hint
    cv2.putText(frame, "Press Q to quit", (w - 240, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    # Cooldown bar
    if cooldown_counter > 0:
        bar_w = int(200 * cooldown_counter / COOLDOWN_FRAMES)
        cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 38), (255, 140, 0), -1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the hand-clap counting loop."""
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    clap_count = 0
    cooldown_counter = 0
    separation_counter = 0
    hands_were_close = False

    print(f"Model: {MODEL_PATH}")
    print("=" * 50)
    print("Hand Clap Counter — MediaPipe")
    print(f"Tracking {len(PAIR_LANDMARKS)} point pairs, need {MIN_PAIRS_CLOSE}+ close to count.")
    print("Press 'q' to quit.")
    print("=" * 50)

    with WebcamManager(camera_id=0, width=1280, height=720) as cam:
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

                clap_this_frame = False

                # --- Clap detection (requires exactly 2 hands) ---
                if result.hand_landmarks and len(result.hand_landmarks) == 2:
                    lms0 = result.hand_landmarks[0]
                    lms1 = result.hand_landmarks[1]
                    close_pairs = _count_close_pairs(lms0, lms1)

                    if cooldown_counter > 0:
                        cooldown_counter -= 1
                    if separation_counter > 0:
                        separation_counter -= 1

                    close = close_pairs >= MIN_PAIRS_CLOSE

                    if (close and not hands_were_close
                            and cooldown_counter == 0
                            and separation_counter == 0):
                        clap_count += 1
                        clap_this_frame = True
                        cooldown_counter = COOLDOWN_FRAMES

                    if not close and separation_counter == 0:
                        separation_counter = SEPARATION_FRAMES

                    hands_were_close = close

                # --- HUD ---
                num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
                _draw_hud(frame, clap_count, clap_this_frame, num_hands, cooldown_counter)

                cv2.imshow("Hand Clap Counter", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()
    print(f"Total claps counted: {clap_count}")
    print("Done.")


if __name__ == "__main__":
    main()

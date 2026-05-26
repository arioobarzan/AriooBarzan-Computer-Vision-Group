"""
Hand Clap Counter with MediaPipe
=================================
Real-time hand-clap counter via webcam.
Detects when both hands clap by checking proximity of corresponding
landmark pairs (5 fingertips + palm centre). A clap is registered
when >= MIN_PAIRS_CLOSE of the pairs are closer than the threshold.

Press 'q' to quit.
"""

import os
import sys
import time

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
# Landmark pairs used for clap detection
#   4=thumb tip, 8=index tip, 12=middle tip, 16=ring tip, 20=pinky tip, 9=palm
# ---------------------------------------------------------------------------
PAIR_INDICES = [4, 8, 12, 16, 20, 9]

# ---------------------------------------------------------------------------
# Clap-detection parameters
# ---------------------------------------------------------------------------
PAIR_DISTANCE_THRESHOLD = 0.10  # normalised distance — a pair closer than this counts
MIN_PAIRS_CLOSE = 3             # at least this many pairs must be close to register a clap
COOLDOWN_MS = 500               # min milliseconds between consecutive claps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _count_close_pairs(lms1, lms2) -> int:
    """Return how many of the landmark pairs are closer than threshold."""
    close = 0
    for idx in PAIR_INDICES:
        dx = lms1[idx].x - lms2[idx].x
        dy = lms1[idx].y - lms2[idx].y
        if (dx * dx + dy * dy) ** 0.5 < PAIR_DISTANCE_THRESHOLD:
            close += 1
    return close


def _draw_hud(frame, clap_count: int, clap_this_frame: bool,
              num_hands: int, close_pairs: int, cooldown_remaining_ms: float) -> None:
    """Render counter, flash, status, and cooldown bar."""
    h, w = frame.shape[:2]

    # Dark transparent banner at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    # Label
    cv2.putText(frame, "Claps", (w // 2 - 62, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2, cv2.LINE_AA)
    # Counter
    cv2.putText(frame, str(clap_count), (w // 2 - 40, 90),
                cv2.FONT_HERSHEY_DUPLEX, 3.2, (0, 255, 100), 5, cv2.LINE_AA)

    # Green flash on clap
    if clap_this_frame:
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), -1)
        cv2.addWeighted(flash, 0.2, frame, 0.8, 0, dst=frame)
        cv2.putText(frame, "CLAP!", (w // 2 - 100, h // 2 + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 0), 4, cv2.LINE_AA)

    # Hand status (bottom-left)
    status_color = (0, 255, 0) if num_hands == 2 else (0, 165, 255)
    cv2.putText(frame, f"Hands: {num_hands}/2", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    # Close-pairs indicator
    pairs_color = (0, 255, 0) if close_pairs >= MIN_PAIRS_CLOSE else (180, 180, 180)
    cv2.putText(frame, f"Pairs close: {close_pairs}/{len(PAIR_INDICES)}  (need {MIN_PAIRS_CLOSE})",
                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pairs_color, 1, cv2.LINE_AA)

    # Quit hint (bottom-right)
    cv2.putText(frame, "Press Q to quit", (w - 240, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    # Cooldown bar
    if cooldown_remaining_ms > 0:
        bar_w = int(200 * cooldown_remaining_ms / COOLDOWN_MS)
        cv2.rectangle(frame, (w - 220, h - 45), (w - 220 + bar_w, h - 33), (255, 140, 0), -1)


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
    last_clap_time = 0.0
    hands_were_close = False

    print(f"Model: {MODEL_PATH}")
    print("=" * 50)
    print("Hand Clap Counter — MediaPipe")
    print(f"Tracking {len(PAIR_INDICES)} point pairs, need {MIN_PAIRS_CLOSE}+ close to count.")
    print(f"Threshold: {PAIR_DISTANCE_THRESHOLD}, cooldown: {COOLDOWN_MS}ms")
    print("Press 'q' to quit.")
    print("=" * 50)

    with WebcamManager(camera_id=0, width=1280, height=720) as cam:
        frame_idx = 0
        with HandLandmarker.create_from_options(options) as detector:
            while True:
                success, frame = cam.read()
                if not success:
                    break

                now = time.time()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(mp_image, frame_idx)
                frame_idx += 1

                clap_this_frame = False
                close_pairs = 0

                # --- Clap detection (requires exactly 2 hands) ---
                if result.hand_landmarks and len(result.hand_landmarks) == 2:
                    lms0 = result.hand_landmarks[0]
                    lms1 = result.hand_landmarks[1]
                    close_pairs = _count_close_pairs(lms0, lms1)

                    elapsed_ms = (now - last_clap_time) * 1000
                    close = close_pairs >= MIN_PAIRS_CLOSE

                    if close and not hands_were_close and elapsed_ms >= COOLDOWN_MS:
                        clap_count += 1
                        clap_this_frame = True
                        last_clap_time = now

                    hands_were_close = close

                # --- HUD ---
                num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
                cooldown_remaining_ms = max(0.0, COOLDOWN_MS - (now - last_clap_time) * 1000)
                _draw_hud(frame, clap_count, clap_this_frame, num_hands, close_pairs, cooldown_remaining_ms)

                cv2.imshow("Hand Clap Counter", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()
    print(f"Total claps counted: {clap_count}")
    print("Done.")


if __name__ == "__main__":
    main()

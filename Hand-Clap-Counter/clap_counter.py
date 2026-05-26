"""
Hand Clap Counter with MediaPipe
=================================
Real-time hand-clap counter via webcam.

Visualisation:
  - 5 fingertip pairs + palm-centre pair, each pair its own colour
  - Lines connecting corresponding points between the two hands
  - Normalised distance drawn on every connecting line

Detection:
  - A clap is registered when >= 4 of the 6 pairs are closer than the threshold.

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
# Point-pair definitions
#   (landmark_idx, label, BGR colour)
# ---------------------------------------------------------------------------
PAIRS = [
    (4,  "Thumb",  (255, 0, 0)),      # Blue
    (8,  "Index",  (0, 255, 0)),      # Green
    (12, "Middle", (0, 0, 255)),      # Red
    (16, "Ring",   (255, 255, 0)),    # Cyan
    (20, "Pinky",  (255, 0, 255)),    # Magenta
    (9,  "Palm",   (0, 210, 255)),    # Gold
]
PAIR_INDICES = [p[0] for p in PAIRS]

# ---------------------------------------------------------------------------
# Clap-detection parameters
# ---------------------------------------------------------------------------
PAIR_DISTANCE_THRESHOLD = 0.10  # normalised distance — a pair closer than this counts
MIN_PAIRS_CLOSE = 4             # at least this many pairs must be close to register a clap
COOLDOWN_MS = 500               # min milliseconds between consecutive claps

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _landmark_to_px(lm, w: int, h: int) -> tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def _draw_pairs(frame, lms0, lms1, pair_distances, h: int, w: int) -> None:
    """Draw coloured circles, connecting lines, and distance labels for every pair."""
    for i, (lm_idx, label, color) in enumerate(PAIRS):
        p0 = _landmark_to_px(lms0[lm_idx], w, h)
        p1 = _landmark_to_px(lms1[lm_idx], w, h)
        norm_dist = pair_distances[i]

        # Connecting line (dashed-thin when far, solid-thick when close)
        close = norm_dist < PAIR_DISTANCE_THRESHOLD
        thickness = 3 if close else 1
        cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)

        # Circles on both hands
        cv2.circle(frame, p0, 10, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p0, 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, p1, 10, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p1, 10, (255, 255, 255), 2, cv2.LINE_AA)

        # Distance label at midpoint
        mx, my = (p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2
        text = f"{norm_dist:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)

        # Dark background pill behind text
        cv2.rectangle(frame,
                      (mx - tw // 2 - 6, my - th // 2 - 4),
                      (mx + tw // 2 + 6, my + th // 2 + 4),
                      (30, 30, 30), -1)
        cv2.putText(frame, text, (mx - tw // 2, my + th // 2),
                    font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_single_hand_dots(frame, lms, h: int, w: int) -> None:
    """When only one hand is visible, draw its pair-points without lines."""
    for lm_idx, _label, color in PAIRS:
        p = _landmark_to_px(lms[lm_idx], w, h)
        cv2.circle(frame, p, 10, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p, 10, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_hud(frame, clap_count: int, clap_this_frame: bool,
              num_hands: int, close_pairs: int, cooldown_remaining_ms: float) -> None:
    """Render counter, flash, status, and cooldown bar."""
    h, w = frame.shape[:2]

    # Dark transparent banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    # Label
    cv2.putText(frame, "Claps", (w // 2 - 62, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2, cv2.LINE_AA)
    # Counter
    cv2.putText(frame, str(clap_count), (w // 2 - 40, 90),
                cv2.FONT_HERSHEY_DUPLEX, 3.2, (0, 255, 100), 5, cv2.LINE_AA)

    # Green flash
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

    # Close-pairs indicator (bottom-centre-left)
    pairs_color = (0, 255, 0) if close_pairs >= MIN_PAIRS_CLOSE else (180, 180, 180)
    cv2.putText(frame, f"Pairs close: {close_pairs}/{len(PAIRS)}  (need {MIN_PAIRS_CLOSE})",
                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pairs_color, 1, cv2.LINE_AA)

    # Quit hint (bottom-right)
    cv2.putText(frame, "Press Q to quit", (w - 240, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    # Cooldown bar (time-based)
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
    print(f"Tracking {len(PAIRS)} point pairs, need {MIN_PAIRS_CLOSE}+ close to count.")
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

                h, w = frame.shape[:2]
                clap_this_frame = False
                close_pairs = 0

                # --- Clap detection & visualisation (requires 2 hands) ---
                if result.hand_landmarks and len(result.hand_landmarks) == 2:
                    lms0 = result.hand_landmarks[0]
                    lms1 = result.hand_landmarks[1]

                    # Compute normalised distance for every pair
                    pair_distances = []
                    for lm_idx in PAIR_INDICES:
                        dx = lms0[lm_idx].x - lms1[lm_idx].x
                        dy = lms0[lm_idx].y - lms1[lm_idx].y
                        pair_distances.append((dx * dx + dy * dy) ** 0.5)

                    close_pairs = sum(1 for d in pair_distances if d < PAIR_DISTANCE_THRESHOLD)

                    # Draw coloured points and connecting lines
                    _draw_pairs(frame, lms0, lms1, pair_distances, h, w)

                    elapsed_ms = (now - last_clap_time) * 1000
                    close = close_pairs >= MIN_PAIRS_CLOSE

                    if (close and not hands_were_close
                            and elapsed_ms >= COOLDOWN_MS):
                        clap_count += 1
                        clap_this_frame = True
                        last_clap_time = now

                    hands_were_close = close

                elif result.hand_landmarks and len(result.hand_landmarks) == 1:
                    # Single hand: draw dots only, no lines
                    _draw_single_hand_dots(frame, result.hand_landmarks[0], h, w)

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

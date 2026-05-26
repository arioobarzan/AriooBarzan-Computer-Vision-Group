"""
Hand Clap Counter with MediaPipe
=================================
Detects hand claps in real-time using your webcam.
Tracks both hands via MediaPipe and counts a clap each time
the wrists come close together. Displays a live counter overlay
with visual feedback on every clap.

Press 'q' to quit.
"""

import os
import sys

import cv2
import mediapipe as mp

# Allow importing from the repo root (common/ package)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import WebcamManager

# ---------------------------------------------------------------------------
# MediaPipe setup (legacy Solutions API — needed for built-in skeleton drawing)
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

# ---------------------------------------------------------------------------
# Clap-detection parameters
# ---------------------------------------------------------------------------
CLAP_DISTANCE_THRESHOLD = 0.12   # normalized distance (0..1) — wrists closer = clap
COOLDOWN_FRAMES = 25             # min frames between consecutive claps
SEPARATION_FRAMES = 8            # hands must be apart this many frames before next clap

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _wrist_distance(lm1, lm2) -> float:
    """Normalised Euclidean distance between two wrist landmarks."""
    return ((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2) ** 0.5


def _draw_hud(frame, clap_count: int, clap_this_frame: bool,
              num_hands: int, cooldown_counter: int) -> None:
    """Render the on-screen HUD: clap counter, status, flash, cooldown bar."""
    h, w = frame.shape[:2]

    # Dark transparent banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    # Counter label
    cv2.putText(frame, "Claps", (w // 2 - 62, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    # Counter value
    cv2.putText(frame, str(clap_count), (w // 2 - 40, 95),
                cv2.FONT_HERSHEY_DUPLEX, 3.5, (0, 255, 100), 5, cv2.LINE_AA)

    # Green flash on clap
    if clap_this_frame:
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), -1)
        cv2.addWeighted(flash, 0.2, frame, 0.8, 0, dst=frame)
        cv2.putText(frame, "CLAP!", (w // 2 - 100, h // 2 + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 0), 4, cv2.LINE_AA)

    # Hand-count status (bottom-left)
    status_color = (0, 255, 0) if num_hands == 2 else (0, 165, 255)
    cv2.putText(frame, f"Hands: {num_hands}/2", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    # FPS (bottom-right)
    cv2.putText(frame, "Press Q to quit", (w - 240, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    # Cooldown bar
    if cooldown_counter > 0:
        bar_w = int(200 * cooldown_counter / COOLDOWN_FRAMES)
        cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 38), (255, 140, 0), -1)


def main() -> None:
    """Run the hand-clap counting loop."""
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    # Clap-detection state
    clap_count = 0
    cooldown_counter = 0
    separation_counter = 0
    hands_were_close = False

    print("=" * 50)
    print("Hand Clap Counter — MediaPipe")
    print("Clap your hands in front of the webcam.")
    print("Press 'q' to quit.")
    print("=" * 50)

    with WebcamManager(camera_id=0, width=1280, height=720) as cam:
        while True:
            success, frame = cam.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            clap_this_frame = False

            # --- Detection logic (requires exactly 2 hands) ---
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                lm0 = results.multi_hand_landmarks[0].landmark
                lm1 = results.multi_hand_landmarks[1].landmark
                dist = _wrist_distance(lm0[0], lm1[0])   # landmark 0 = wrist

                if cooldown_counter > 0:
                    cooldown_counter -= 1
                if separation_counter > 0:
                    separation_counter -= 1

                close = dist < CLAP_DISTANCE_THRESHOLD

                # A fresh clap: hands just became close, cooldown expired, hands were apart
                if close and not hands_were_close and cooldown_counter == 0 and separation_counter == 0:
                    clap_count += 1
                    clap_this_frame = True
                    cooldown_counter = COOLDOWN_FRAMES

                # Track separation — hands must move apart before next clap
                if not close and separation_counter == 0:
                    separation_counter = SEPARATION_FRAMES

                hands_were_close = close

            # --- Draw hand skeletons ---
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw_styles.get_default_hand_landmarks_style(),
                        mp_draw_styles.get_default_hand_connections_style(),
                    )

            # --- HUD overlay ---
            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            _draw_hud(frame, clap_count, clap_this_frame, num_hands, cooldown_counter)

            cv2.imshow("Hand Clap Counter", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    hands.close()
    cv2.destroyAllWindows()
    print(f"Total claps counted: {clap_count}")
    print("Done.")


if __name__ == "__main__":
    main()

"""
Anti-Slouch Alarm with MediaPipe Pose
======================================
Real-time posture monitor and slouch alarm via webcam.

* Calibration — press 'c' while sitting upright to save your baseline.
* Monitoring — continuously compares shoulder position, head posture,
  and screen proximity against the calibrated baseline.
* Alarm — red overlay + beep when bad posture persists > 3 seconds.

Press 'q' to quit, 'r' to recalibrate.
"""

import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

# Allow importing from the repo root (Sample Codes/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (
    WebcamManager,
    clear_mediapipe_cache,
    get_model_path,
    POSE_MODEL_FILENAME,
    suppress_gpu_warnings,
)

# ---------------------------------------------------------------------------
# Environment & model
# ---------------------------------------------------------------------------
clear_mediapipe_cache()
suppress_gpu_warnings()
MODEL_PATH = get_model_path(POSE_MODEL_FILENAME)

# ---------------------------------------------------------------------------
# MediaPipe setup (Tasks API — Pose Landmarker)
# ---------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe Pose)
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_EYE_OUTER = 2
RIGHT_EYE_OUTER = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# ---------------------------------------------------------------------------
# Posture thresholds (fractional deviation from baseline)
# ---------------------------------------------------------------------------
SLOUCH_SHOULDER_DROP = 0.06     # shoulder Y drops > 6 % of frame height → slouch
SLOUCH_HEAD_FORWARD = 0.70     # ear-to-shoulder ratio < 70 % of baseline → head forward
PROXIMITY_EYE_SCALE = 1.12     # eye distance > 112 % of baseline → too close
BAD_POSTURE_SECONDS = 3.0      # continuous seconds before alarm triggers

# HUD colours (BGR)
GREEN = (0, 200, 80)
ORANGE = (0, 165, 255)
RED = (0, 50, 255)


# ---------------------------------------------------------------------------
# Sound
# ---------------------------------------------------------------------------
def _beep() -> None:
    """Play an audible alert (cross-platform best-effort)."""
    try:
        import winsound
        winsound.Beep(1000, 300)
    except ImportError:
        print("\a", end="", flush=True)


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------
def _landmark_to_xy(lm, w: int, h: int) -> tuple[float, float]:
    """Convert a normalised landmark to pixel coordinates."""
    return lm.x * w, lm.y * h


def _distance(lm1, lm2) -> float:
    """Normalised Euclidean distance between two landmarks."""
    return ((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2) ** 0.5


def _extract_posture_metrics(pose_landmarks, h: int, w: int) -> dict:
    """
    Extract the three posture metrics from a single-pose landmark list.

    Returns a dict with keys: shoulder_y, ear_shoulder_ratio, eye_distance.
    Returns None if required landmarks have low visibility.
    """
    lms = pose_landmarks

    # Visibility check on key landmarks
    key_ids = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_EAR, RIGHT_EAR, LEFT_EYE_OUTER, RIGHT_EYE_OUTER]
    for idx in key_ids:
        if lms[idx].visibility < 0.5:
            return None

    h_norm = h  # for normalisation we use pixel height

    # Shoulder Y (average of left & right shoulder Y in pixels)
    left_sy = lms[LEFT_SHOULDER].y * h
    right_sy = lms[RIGHT_SHOULDER].y * h
    shoulder_y = (left_sy + right_sy) / 2

    # Ear-to-shoulder vertical ratio (head height vs frame)
    left_ear_sy = abs(lms[LEFT_EAR].y - lms[LEFT_SHOULDER].y)
    right_ear_sy = abs(lms[RIGHT_EAR].y - lms[RIGHT_SHOULDER].y)
    ear_shoulder_ratio = (left_ear_sy + right_ear_sy) / 2

    # Eye distance (inter-ocular, normalised)
    eye_distance = _distance(lms[LEFT_EYE_OUTER], lms[RIGHT_EYE_OUTER])

    return {
        "shoulder_y": shoulder_y / h_norm,      # normalised by frame height
        "ear_shoulder_ratio": ear_shoulder_ratio,  # normalised (fraction of frame)
        "eye_distance": eye_distance,            # normalised 0..1
    }


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------
def _draw_hud(frame, status: str, posture_ok: bool, calib: dict | None,
              metrics: dict | None, bad_sec: float, frame_h: int, frame_w: int) -> None:
    """Render overlay: status banner, metrics, timer, calibration prompt."""
    h, w = frame_h, frame_w

    # ---- Top banner ----
    if status == "calibrating":
        banner_color = ORANGE
        banner_text = "PRESS 'C' TO CALIBRATE — sit upright first"
    elif not posture_ok and bad_sec >= BAD_POSTURE_SECONDS:
        banner_color = RED
        banner_text = "!! BAD POSTURE — SIT UP STRAIGHT !!"
    elif not posture_ok:
        banner_color = ORANGE
        banner_text = f"Adjust your posture... ({bad_sec:.1f}s)"
    else:
        banner_color = GREEN
        banner_text = "Posture OK"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), banner_color, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, dst=frame)
    cv2.putText(frame, banner_text, (w // 2 - 300, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # ---- Red flash on active alarm ----
    if not posture_ok and bad_sec >= BAD_POSTURE_SECONDS:
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(flash, 0.15, frame, 0.85, 0, dst=frame)

    # ---- Metrics panel (bottom-left) ----
    panel_y = h - 120
    cv2.rectangle(frame, (10, panel_y), (340, h - 10), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, panel_y), (340, h - 10), (80, 80, 80), 1)

    if calib and metrics:
        lines = [
            f"Shoulder Y : {metrics['shoulder_y']:.3f}  (base: {calib['shoulder_y']:.3f})",
            f"Head/Shldr : {metrics['ear_shoulder_ratio']:.3f}  (base: {calib['ear_shoulder_ratio']:.3f})",
            f"Eye dist   : {metrics['eye_distance']:.3f}  (base: {calib['eye_distance']:.3f})",
        ]
    elif calib:
        lines = ["Calibrated — waiting for pose..."]
    else:
        lines = ["Not calibrated — press 'C'"]

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (20, panel_y + 25 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # ---- Keybindings (bottom-right) ----
    cv2.putText(frame, "Q=quit  C=calibrate  R=recalibrate",
                (w - 400, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    # ---- Pose skeleton ----
    if metrics is not None:
        _draw_pose_overlay(frame, h, w)


# Separate function to keep draw_hud readable
_LAST_POSE_LANDMARKS = None


def _draw_pose_overlay(frame, h: int, w: int) -> None:
    """Draw minimal pose skeleton on frame using the last known landmarks."""
    global _LAST_POSE_LANDMARKS
    lms = _LAST_POSE_LANDMARKS
    if lms is None:
        return

    # Draw key points
    for idx in [NOSE, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_EAR, RIGHT_EAR,
                LEFT_SHOULDER, RIGHT_SHOULDER]:
        x, y = int(lms[idx].x * w), int(lms[idx].y * h)
        cv2.circle(frame, (x, y), 5, (200, 200, 200), -1, cv2.LINE_AA)

    # Draw shoulder line
    ls = (int(lms[LEFT_SHOULDER].x * w), int(lms[LEFT_SHOULDER].y * h))
    rs = (int(lms[RIGHT_SHOULDER].x * w), int(lms[RIGHT_SHOULDER].y * h))
    cv2.line(frame, ls, rs, (200, 200, 200), 2, cv2.LINE_AA)

    # Draw ear-to-shoulder lines
    le = (int(lms[LEFT_EAR].x * w), int(lms[LEFT_EAR].y * h))
    re = (int(lms[RIGHT_EAR].x * w), int(lms[RIGHT_EAR].y * h))
    cv2.line(frame, le, ls, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.line(frame, re, rs, (180, 180, 180), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the anti-slouch monitoring loop."""
    global _LAST_POSE_LANDMARKS

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # State
    calibration: dict | None = None
    status = "calibrating"
    bad_posture_start: float | None = None

    print(f"Model: {MODEL_PATH}")
    print("=" * 55)
    print("Anti-Slouch Alarm — MediaPipe Pose")
    print("  1. Sit upright with good posture.")
    print("  2. Press 'c' to calibrate.")
    print("  3. The alarm triggers if you slouch for > 3 s.")
    print("Press 'c' = calibrate | 'r' = recalibrate | 'q' = quit")
    print("=" * 55)

    with WebcamManager(camera_id=0, width=1280, height=720) as cam:
        frame_idx = 0
        with PoseLandmarker.create_from_options(options) as detector:
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
                metrics = None
                posture_ok = True

                # --- Pose detected ---
                if result.pose_landmarks:
                    pose_lms = result.pose_landmarks[0]
                    _LAST_POSE_LANDMARKS = pose_lms
                    metrics = _extract_posture_metrics(pose_lms, h, w)

                    if metrics and calibration:
                        # Compare against baseline
                        shoulder_dropped = (
                            metrics["shoulder_y"] - calibration["shoulder_y"]
                            > SLOUCH_SHOULDER_DROP
                        )
                        head_forward = (
                            metrics["ear_shoulder_ratio"]
                            < calibration["ear_shoulder_ratio"] * SLOUCH_HEAD_FORWARD
                        )
                        too_close = (
                            metrics["eye_distance"]
                            > calibration["eye_distance"] * PROXIMITY_EYE_SCALE
                        )
                        posture_ok = not (shoulder_dropped or head_forward or too_close)

                # --- Bad-posture timer ---
                if not posture_ok and calibration:
                    if bad_posture_start is None:
                        bad_posture_start = now
                    bad_sec = now - bad_posture_start
                    if bad_sec >= BAD_POSTURE_SECONDS:
                        _beep()
                else:
                    bad_posture_start = None
                    bad_sec = 0.0

                # --- HUD ---
                _draw_hud(frame, status, posture_ok, calibration, metrics,
                          bad_sec if bad_posture_start else 0.0, h, w)

                cv2.imshow("Anti-Slouch Alarm", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in (ord("c"), ord("r")):
                    if metrics:
                        calibration = metrics
                        status = "monitoring"
                        bad_posture_start = None
                        print(f"[OK] Calibrated — shoulder_y={metrics['shoulder_y']:.3f}, "
                              f"ear_shldr={metrics['ear_shoulder_ratio']:.3f}, "
                              f"eye_dist={metrics['eye_distance']:.3f}")

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

"""
Dual-Hand Volume Controller with MediaPipe Hands
==================================================
Control your system master volume by moving your hands apart or together.

- Move hands **apart**  -> volume **up**   (100 %)
- Move hands **close**   -> volume **down**  (0 %)
- Fixed thresholds: index tips < 0.1 apart = mute, > 0.5 apart = max volume.
- Press **'q'** to quit.

Uses MediaPipe Hands (Tasks API) for hand tracking and the native Windows
Core Audio API (via **ctypes**) for system volume control -- **no external
audio dependencies required**.
"""

import ctypes
import os
import sys
from collections import deque
from ctypes import POINTER, byref, c_float, c_uint, c_void_p, cast, windll
from ctypes.wintypes import DWORD
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

# ===================================================================
# Windows Core Audio Volume Control (ctypes -- zero dependencies)
# ===================================================================
# COM IAudioEndpointVolume vtable layout (after IUnknown's 3 methods):
#   3: RegisterControlChangeNotify
#   4: UnregisterControlChangeNotify
#   5: GetChannelCount
#   6: SetMasterVolumeLevel        (dB)
#   7: SetMasterVolumeLevelScalar  (0.0 … 1.0)  ← the one we use
#   8: GetMasterVolumeLevel
#   9: GetMasterVolumeLevelScalar


# ── GUID helper ─────────────────────────────────────────────────
class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", DWORD),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _str_to_guid(s: str) -> _GUID:
    buf = ctypes.create_unicode_buffer(s)
    guid = _GUID()
    windll.ole32.CLSIDFromString(ctypes.byref(buf), ctypes.byref(guid))
    return guid


# ── COM vtable helpers ──────────────────────────────────────────
def _com_call(this: c_void_p, idx: int, restype, *argtypes):
    """Return a ctypes function pointer for vtable method **idx**."""
    vtable = cast(this, POINTER(POINTER(c_void_p)))
    func = cast(vtable[0][idx], ctypes.CFUNCTYPE(restype, c_void_p, *argtypes))
    return func


# ── Acquire IAudioEndpointVolume once ───────────────────────────
_p_vol: Optional[c_void_p] = None


def _init_volume() -> Optional[c_void_p]:
    """Create MMDeviceEnumerator -> default speaker -> IAudioEndpointVolume."""
    try:
        hr = windll.ole32.CoInitialize(None)
        if hr < 0 and hr != 0x00000001:
            return None

        # IMMDeviceEnumerator
        clsid = _str_to_guid("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
        iid_enum = _str_to_guid("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
        p_enum = c_void_p()
        hr = windll.ole32.CoCreateInstance(
            byref(clsid), None, 1, byref(iid_enum), byref(p_enum))
        if hr < 0 or not p_enum:
            return None

        # GetDefaultAudioEndpoint (vtable idx 4)
        get_def = _com_call(p_enum, 4, ctypes.c_long,
                            c_uint, c_uint, POINTER(c_void_p))
        p_dev = c_void_p()
        hr = get_def(p_enum, 0, 0, byref(p_dev))   # eRender, eConsole
        if hr < 0 or not p_dev:
            return None

        # IMMDevice::Activate (vtable idx 3)
        iid_vol = _str_to_guid("{5CDF2C82-841E-4546-9722-0CF74078229A}")
        activate = _com_call(p_dev, 3, ctypes.c_long,
                             POINTER(_GUID), DWORD, c_void_p, POINTER(c_void_p))
        p_vol = c_void_p()
        hr = activate(p_dev, byref(iid_vol), 0, None, byref(p_vol))
        if hr < 0 or not p_vol:
            return None

        return p_vol
    except Exception:
        return None


_p_vol = _init_volume()
_VOLUME_AVAILABLE = _p_vol is not None


def _set_system_volume_scalar(scalar: float) -> None:
    """Set master volume [0.0 … 1.0] via IAudioEndpointVolume."""
    if not _VOLUME_AVAILABLE or _p_vol is None:
        return
    try:
        scalar = max(0.0, min(1.0, scalar))
        set_scalar = _com_call(_p_vol, 7, ctypes.c_long, c_float, c_void_p)
        set_scalar(_p_vol, c_float(scalar), None)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# MediaPipe setup (Tasks API -- Hand Landmarker)
# ---------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ---------------------------------------------------------------------------
# Colour constants (BGR)
# ---------------------------------------------------------------------------
GREEN = (0, 255, 80)
WHITE = (255, 255, 255)
GREY = (180, 180, 180)
DARK = (30, 30, 30)


# ===================================================================
# DualHandVolumeController
# ===================================================================
class DualHandVolumeController:
    """
    Tracks two hands via MediaPipe and maps inter-hand distance to
    system master volume.

    Attributes:
        min_dist: Distance below which volume = 0 % (default 0.10).
        max_dist: Distance above which volume = 100 % (default 0.50).
        volume_smooth: ``deque`` buffer for temporal smoothing.
        volume_pct: Current smoothed volume percentage.
    """

    # ── initialisation ────────────────────────────────────────────
    def __init__(self) -> None:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self._detector = HandLandmarker.create_from_options(options)

        # Fixed distance thresholds (no calibration needed)
        self.min_dist: float = 0.10   # index fingertips nearly touching = 0% volume
        self.max_dist: float = 0.50   # index fingertips wide apart = 100% volume

        # Smoothing -- moving average over last N raw values
        self._smooth_window = 8
        self._volume_buffer: deque[float] = deque(maxlen=self._smooth_window)

        # Current state (updated each frame)
        self.volume_pct: float = 50.0
        self._raw_vol: float = 50.0
        self._distance: float = 0.0
        self._hands_visible = 0
        self._center0: Optional[tuple[float, float]] = None
        self._center1: Optional[tuple[float, float]] = None
        self._frame_idx = 0
        self._running = True

    # ── context manager ───────────────────────────────────────────
    def __enter__(self) -> "DualHandVolumeController":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Release MediaPipe detector."""
        self._running = False
        if hasattr(self, "_detector"):
            self._detector.close()
        # Clean up COM
        try:
            _CoUninitialize()
        except Exception:
            pass

    # ── hand geometry ─────────────────────────────────────────────
    FINGERTIP_IDX = 8  # index-finger tip landmark

    @staticmethod
    def _fingertip(landmarks, w: int, h: int) -> tuple[float, float]:
        """Return pixel (x, y) of the index-finger tip (landmark 8)."""
        lm = landmarks[DualHandVolumeController.FINGERTIP_IDX]
        return lm.x * w, lm.y * h

    @staticmethod
    def _distance_px(p0, p1) -> float:
        """Euclidean distance in pixels."""
        return np.hypot(p1[0] - p0[0], p1[1] - p0[1])

    # ── volume mapping ────────────────────────────────────────────
    def _distance_to_volume(self, dist_norm: float) -> float:
        """Map normalised fingertip distance linearly to volume [0, 100].
        < 0.1 = 0%  |  0.5 = 100%  |  > 0.5 = 100% (clamped)."""
        clamped = max(self.min_dist, min(dist_norm, self.max_dist))
        fraction = ((clamped - self.min_dist)
                    / (self.max_dist - self.min_dist))
        return fraction * 100.0

    def _apply_smoothing(self, raw_vol: float) -> float:
        """Return a temporally smoothed volume (moving average)."""
        self._volume_buffer.append(raw_vol)
        return sum(self._volume_buffer) / len(self._volume_buffer)

    # ── drawing ───────────────────────────────────────────────────
    def _draw_volume_bar(self, frame: np.ndarray,
                         volume_pct: float) -> None:
        """Draw a vertical volume bar on the right side of the frame."""
        h, w = frame.shape[:2]
        bar_x = w - 60
        bar_top = 80
        bar_bottom = h - 80
        bar_w = 36
        bar_h = bar_bottom - bar_top

        # Background track
        cv2.rectangle(frame,
                      (bar_x - 2, bar_top - 2),
                      (bar_x + bar_w + 2, bar_bottom + 2),
                      DARK, -1)
        cv2.rectangle(frame,
                      (bar_x, bar_top), (bar_x + bar_w, bar_bottom),
                      (60, 60, 60), -1)
        cv2.rectangle(frame,
                      (bar_x, bar_top), (bar_x + bar_w, bar_bottom),
                      (100, 100, 100), 1)

        # Filled portion (bottom -> up)
        fill_h = int(bar_h * volume_pct / 100)
        fill_y = bar_bottom - fill_h

        # Gradient colour: red (low) -> yellow -> green (high)
        r = int(255 * (1 - volume_pct / 100))
        g = int(255 * volume_pct / 100)
        bar_color = (0, g, r)

        cv2.rectangle(frame,
                      (bar_x, fill_y), (bar_x + bar_w, bar_bottom),
                      bar_color, -1)

        # Handle cap
        cv2.rectangle(frame,
                      (bar_x - 4, fill_y - 6),
                      (bar_x + bar_w + 4, fill_y),
                      WHITE, -1)

        # Percentage label
        pct_text = f"{int(volume_pct)}%"
        cv2.putText(frame, pct_text,
                    (bar_x - 24, bar_top - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2, cv2.LINE_AA)

        # Tick marks
        for pct in (25, 50, 75):
            ty = bar_bottom - int(bar_h * pct / 100)
            cv2.line(frame,
                     (bar_x, ty), (bar_x + bar_w, ty),
                     (80, 80, 80), 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Render hand centres, connection line, distance, and volume info."""
        h, w = frame.shape[:2]

        # ── Top banner ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 72), DARK, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

        if self._hands_visible != 2:
            status = f"Waiting for 2 hands...  (visible: {self._hands_visible})"
            status_color = (0, 165, 255)
        else:
            status = f"Volume: {int(self.volume_pct)}%"
            status_color = GREEN

        cv2.putText(frame, status,
                    (w // 2 - 320, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2, cv2.LINE_AA)

        # ── Keybinding hints (bottom) ──
        hints = "Q = quit"
        if not _VOLUME_AVAILABLE:
            hints += "   |   (visual-only mode -- audio API unavailable)"
        cv2.putText(frame, hints,
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, GREY, 1, cv2.LINE_AA)

        # ── Hand centres + connection line ──
        if self._center0 is not None and self._center1 is not None:
            c0 = (int(self._center0[0]), int(self._center0[1]))
            c1 = (int(self._center1[0]), int(self._center1[1]))

            # Connecting line (thickness scales with volume)
            thickness = max(2, int(6 * self.volume_pct / 100))
            line_color = (0, int(255 * self.volume_pct / 100),
                          int(255 * (1 - self.volume_pct / 100)))
            cv2.line(frame, c0, c1, line_color, thickness, cv2.LINE_AA)

            # Hand-centre dots
            for c in (c0, c1):
                cv2.circle(frame, c, 14, WHITE, 2, cv2.LINE_AA)
                cv2.circle(frame, c, 10, line_color, -1, cv2.LINE_AA)

            # Distance + volume label at midpoint
            mx, my = (c0[0] + c1[0]) // 2, (c0[1] + c1[1]) // 2
            label = f"Dist: {self._distance:.3f}  |  Vol: {int(self.volume_pct)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame,
                          (mx - tw // 2 - 8, my - th - 12),
                          (mx + tw // 2 + 8, my + 4),
                          DARK, -1)
            cv2.putText(frame, label,
                        (mx - tw // 2, my - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

        # Single-hand dot
        elif self._center0 is not None:
            c = (int(self._center0[0]), int(self._center0[1]))
            cv2.circle(frame, c, 14, WHITE, 2, cv2.LINE_AA)
            cv2.circle(frame, c, 10, (0, 165, 255), -1, cv2.LINE_AA)

        # ── Volume bar ──
        self._draw_volume_bar(frame, self.volume_pct)

    # ── update loop ───────────────────────────────────────────────
    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one frame: run hand detection, compute distance,
        map to volume, smooth, set system volume, draw HUD.

        Returns the annotated frame.
        """
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, self._frame_idx)
        self._frame_idx += 1

        self._hands_visible = (len(result.hand_landmarks)
                               if result.hand_landmarks else 0)

        # ── Two hands -> compute volume ──
        if result.hand_landmarks and len(result.hand_landmarks) == 2:
            lms0 = result.hand_landmarks[0]
            lms1 = result.hand_landmarks[1]

            c0 = self._fingertip(lms0, w, h)
            c1 = self._fingertip(lms1, w, h)
            self._center0 = c0
            self._center1 = c1

            # Normalised distance (pixel distance / frame diagonal)
            diag = np.hypot(w, h)
            self._distance = self._distance_px(c0, c1) / diag

            raw = self._distance_to_volume(self._distance)
            self._raw_vol = raw
            self.volume_pct = self._apply_smoothing(raw)
            _set_system_volume_scalar(self.volume_pct / 100.0)
        else:
            self._center0 = None
            self._center1 = None

            if result.hand_landmarks and len(result.hand_landmarks) == 1:
                lms = result.hand_landmarks[0]
                self._center0 = self._fingertip(lms, w, h)

        # ── Draw everything ──
        self._draw_hud(frame)
        return frame

# ===================================================================
# main
# ===================================================================
def main() -> None:
    """Run the dual-hand volume controller."""
    status = "ACTIVE" if _VOLUME_AVAILABLE else "UNAVAILABLE (visual-only)"
    print(f"Model: {MODEL_PATH}")
    print(f"Audio API: {status}")
    print("=" * 56)
    print("Dual-Hand Volume Controller -- MediaPipe Hands")
    print("  Index fingertips < 0.1 apart -> mute (0%)")
    print("  Index fingertips > 0.5 apart -> max  (100%)")
    print("  q = quit")
    print("=" * 56)

    with DualHandVolumeController() as controller, \
         WebcamManager(camera_id=0, width=1280, height=720) as cam:

        while True:
            success, frame = cam.read()
            if not success:
                break

            frame = controller.update(frame)

            cv2.imshow("Dual-Hand Volume Controller", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

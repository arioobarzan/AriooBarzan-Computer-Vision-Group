"""
MediaPipe helper utilities.
"""

import os
import shutil


def clear_mediapipe_cache() -> None:
    """Remove the MediaPipe cache directory to avoid stale model issues."""
    try:
        mp_cache = os.path.join(os.path.expanduser("~"), ".mediapipe")
        if os.path.exists(mp_cache):
            shutil.rmtree(mp_cache, ignore_errors=True)
    except Exception:
        pass


def suppress_gpu_warnings() -> None:
    """Disable MediaPipe GPU acceleration (avoids warnings on Windows)."""
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

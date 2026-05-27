"""
MediaPipe helper utilities.
"""

import os
import shutil

HAND_MODEL_FILENAME = "hand_landmarker.task"
POSE_MODEL_FILENAME = "pose_landmarker.task"


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


def get_model_path(filename: str = HAND_MODEL_FILENAME) -> str:
    """
    Resolve the path to a MediaPipe model file.

    Looks first in ``common/models/``, then falls back to the caller's
    directory.  Pass ``filename`` to request a non-default model
    (e.g. ``POSE_MODEL_FILENAME``).

    Returns:
        Absolute path to the model file.

    Raises:
        FileNotFoundError: if the model file cannot be found.
    """
    # Repo root = two levels up from this file (Sample Codes/common/mediapipe_utils.py)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shared = os.path.join(repo_root, "common", "models", filename)
    if os.path.exists(shared):
        return shared

    # Fallback: caller's directory
    import inspect
    caller_frame = inspect.currentframe()
    if caller_frame is not None and caller_frame.f_back is not None:
        caller_dir = os.path.dirname(os.path.abspath(caller_frame.f_back.f_code.co_filename))
        local = os.path.join(caller_dir, filename)
        if os.path.exists(local):
            return local

    raise FileNotFoundError(
        f"Model file '{filename}' not found in:\n"
        f"  {shared}\n"
        "Please download it and place it in common/models/."
    )

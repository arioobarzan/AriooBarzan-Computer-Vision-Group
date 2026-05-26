"""
MediaPipe helper utilities.
"""

import os
import shutil

MODEL_FILENAME = "hand_landmarker.task"


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


def get_model_path() -> str:
    """
    Resolve the path to the hand_landmarker.task model file.

    Looks first in ``common/models/`` at the repo root, then falls back
    to the caller's directory.

    Returns:
        Absolute path to the model file.

    Raises:
        FileNotFoundError: if the model file cannot be found.
    """
    # Repo root = two levels up from this file (common/mediapipe_utils.py)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shared = os.path.join(repo_root, "common", "models", MODEL_FILENAME)
    if os.path.exists(shared):
        return shared

    # Fallback: caller's directory
    import inspect
    caller_file = inspect.currentframe()
    if caller_file is not None and caller_file.f_back is not None:
        caller_dir = os.path.dirname(os.path.abspath(caller_file.f_back.f_code.co_filename))
        local = os.path.join(caller_dir, MODEL_FILENAME)
        if os.path.exists(local):
            return local

    raise FileNotFoundError(
        f"Model file '{MODEL_FILENAME}' not found in:\n"
        f"  {shared}\n"
        "Download from:\n"
        "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task\n"
        f"and place it in common/models/"
    )

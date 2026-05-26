"""
Shared utilities for Computer Vision Group projects.

Provides:
    - WebcamManager: context-managed webcam capture with mirror flip
    - draw_gradient_line: smooth gradient line between two points
    - clear_mediapipe_cache: clean up MediaPipe cache directory
"""

from .camera import WebcamManager
from .drawing import draw_gradient_line
from .mediapipe_utils import clear_mediapipe_cache, suppress_gpu_warnings

__all__ = [
    "WebcamManager",
    "draw_gradient_line",
    "clear_mediapipe_cache",
    "suppress_gpu_warnings",
]

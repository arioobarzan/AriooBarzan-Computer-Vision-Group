"""
Webcam capture utility with context-manager support.
"""

import cv2


class WebcamManager:
    """Context-managed webcam capture with mirror flip."""

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        """Open the webcam. Returns True on success."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def read(self) -> tuple[bool, object | None]:
        """Read a frame (mirrored). Returns (success, frame)."""
        if self.cap is None:
            return False, None
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)
        return success, frame

    def release(self) -> None:
        """Release the webcam."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        if not self.open():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        return self

    def __exit__(self, *args) -> None:
        self.release()

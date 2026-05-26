"""
Drawing utilities for computer vision visualization.
"""

import cv2
import numpy as np


def draw_gradient_line(
    frame: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    max_thickness: int = 14,
    min_thickness: int = 3,
    num_segments: int = 50,
) -> None:
    """
    Draw a line with a smooth gradient glow effect.

    Middle of the line: thick and bright (glowing).
    Ends of the line: thin and saturated (original color).

    Args:
        frame: Target image (modified in-place).
        p1, p2: Start and end points.
        color: Base color in BGR format.
        max_thickness: Line thickness at the middle.
        min_thickness: Line thickness at the ends.
        num_segments: Number of line segments (higher = smoother).
    """
    x1, y1 = p1
    x2, y2 = p2
    b, g, r = color

    for i in range(num_segments):
        t1 = i / num_segments
        t2 = (i + 1) / num_segments
        mid = (t1 + t2) / 2

        # Cosine curve: 0 at ends, 1 at middle (smooth bell shape)
        glow = np.sin(np.pi * mid)

        # Color: ends = saturated, middle = bright (closer to white)
        color_factor = glow ** 1.5 * 0.60
        seg_color = (
            int(b + (255 - b) * color_factor),
            int(g + (255 - g) * color_factor),
            int(r + (255 - r) * color_factor),
        )

        # Thickness: thin at ends, thick at middle
        thick_factor = np.sin(np.pi * mid)
        seg_thick = int(min_thickness + (max_thickness - min_thickness) * thick_factor)

        sx1 = int(x1 + (x2 - x1) * t1)
        sy1 = int(y1 + (y2 - y1) * t1)
        sx2 = int(x1 + (x2 - x1) * t2)
        sy2 = int(y1 + (y2 - y1) * t2)

        cv2.line(frame, (sx1, sy1), (sx2, sy2), seg_color, seg_thick)
